// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! io_uring WebSocket Lightstream connection.
//!
//! Wraps a `TcpStream` and handles WebSocket binary framing
//! in the send/recv methods. The HTTP upgrade handshake is done by
//! tungstenite before the raw stream is extracted.
//!
//! Avoids per-frame allocations. The WS header is read into a fixed array
//! on the connection struct. Payloads flow directly into/out of the
//! existing Vec64 encode/decode buffers.

use std::io;

use minarrow::{Field, Vec64};
use tokio_uring::buf::BoundedBuf;
use tokio_uring::net::TcpStream;

use crate::models::codecs::lightstream::LightstreamCodec;
use crate::models::decoders::limits::DecodeLimits;
use crate::models::frames::lightstream_message::{FRAME_HEADER_SIZE, LightstreamMessage};
use crate::models::frames::websocket::{
    self, MAX_HEADER_LEN, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG, fresh_mask_key,
};

use super::buf::UringBuf;

/// Read exactly `len` bytes into a Vec64, starting at `offset`.
async fn read_exact_vec64(
    stream: &TcpStream,
    mut buf: UringBuf,
    mut offset: usize,
    len: usize,
) -> io::Result<UringBuf> {
    let target = offset + len;
    while offset < target {
        let slice = buf.slice(offset..target);
        let (result, slice) = stream.read(slice).await;
        buf = slice.into_inner();
        let n = result?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed during read",
            ));
        }
        offset += n;
    }
    Ok(buf)
}

/// Read exactly `len` bytes into a `Vec<u8>`, starting at `offset`.
async fn read_exact_vec(
    stream: &TcpStream,
    mut buf: Vec<u8>,
    mut offset: usize,
    len: usize,
) -> io::Result<Vec<u8>> {
    let target = offset + len;
    while offset < target {
        let slice = buf.slice(offset..target);
        let (result, slice) = stream.read(slice).await;
        buf = slice.into_inner();
        let n = result?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed during read",
            ));
        }
        offset += n;
    }
    Ok(buf)
}

/// io_uring Lightstream connection over WebSocket.
///
/// Uses a raw TCP stream extracted after the tungstenite handshake.
/// WebSocket binary framing is handled inline with zero extra
/// allocations. Payloads read directly into Vec64 for zero-copy
/// column mapping.
pub struct IoUringWsConnection {
    stream: TcpStream,
    read_codec: LightstreamCodec<Vec64<u8>>,
    write_codec: LightstreamCodec<Vec64<u8>>,
    encode_buf: Vec64<u8>,
    /// Fixed buffer for WS/TLV headers and control payloads. Never reallocated.
    header_buf: Vec<u8>,
    /// Fixed buffer for outbound frame headers and control frames.
    ws_header_out: Vec<u8>,
    eof: bool,
    /// Client role: every outbound frame is masked.
    client: bool,
    limits: DecodeLimits,
}

impl IoUringWsConnection {
    /// Create a connection from a raw `TcpStream`.
    ///
    /// The WebSocket handshake must already be complete. Use
    /// `from_tungstenite` for the typical flow.
    pub fn new(stream: TcpStream, limits: Option<DecodeLimits>) -> Self {
        let limits = limits.unwrap_or_default();
        let control_capacity = (MAX_HEADER_LEN + FRAME_HEADER_SIZE).max(6 + 125);
        let mut header_buf = Vec::with_capacity(control_capacity);
        header_buf.resize(control_capacity, 0);
        let mut ws_header_out = Vec::with_capacity(control_capacity);
        ws_header_out.resize(control_capacity, 0);
        Self {
            stream,
            read_codec: LightstreamCodec::new(Some(limits)),
            write_codec: LightstreamCodec::new(Some(limits)),
            encode_buf: Vec64::with_capacity(0),
            header_buf,
            ws_header_out,
            eof: false,
            client: false,
            limits,
        }
    }

    /// Create a client-side connection from a raw post-handshake stream.
    pub fn new_client(stream: TcpStream, limits: Option<DecodeLimits>) -> Self {
        Self {
            client: true,
            ..Self::new(stream, limits)
        }
    }

    /// Create from a standard library `TcpStream`.
    ///
    /// The WebSocket handshake must already be complete.
    pub fn from_tcp_stream(
        stream: std::net::TcpStream,
        limits: Option<DecodeLimits>,
    ) -> Self {
        Self::new(TcpStream::from_std(stream), limits)
    }

    /// Create a client-side connection from a standard TCP stream after
    /// completing the WebSocket handshake.
    pub fn from_tcp_stream_client(
        stream: std::net::TcpStream,
        limits: Option<DecodeLimits>,
    ) -> Self {
        Self::new_client(TcpStream::from_std(stream), limits)
    }

    /// Register a message type on both halves.
    pub fn register_message(&mut self, name: impl Into<String>) -> u8 {
        let name = name.into();
        let tag = self.write_codec.register_message(name.clone());
        let _ = self.read_codec.register_message(name);
        tag
    }

    /// Register a table type on both halves.
    pub fn register_table(&mut self, name: impl Into<String>, schema: Vec<Field>) -> u8 {
        let name = name.into();
        let tag = self
            .write_codec
            .register_table(name.clone(), schema.clone());
        let _ = self.read_codec.register_table(name, schema);
        tag
    }

    /// Write a WS binary frame header then the payload to the wire.
    ///
    /// Two writes: tiny header first, then payload. No extra allocation.
    async fn ws_write_binary(&mut self, mut payload: UringBuf) -> io::Result<UringBuf> {
        let payload_len = payload.0.len();

        // Write WS header into the fixed output buffer
        let mut ws_header_buf = std::mem::take(&mut self.ws_header_out);
        let header_len = if self.client {
            let key = fresh_mask_key();
            websocket::unmask(&mut payload.0, key);
            websocket::write_masked_binary_header(&mut ws_header_buf, payload_len, key)
        } else {
            websocket::write_binary_header(&mut ws_header_buf, payload_len)
        };

        // Write the WS header bytes
        let header_slice = ws_header_buf.slice(0..header_len);
        let (result, header_slice) = self.stream.write_all(header_slice).await;
        self.ws_header_out = header_slice.into_inner();
        result?;

        // Write the payload
        let (result, payload) = self.stream.write_all(payload).await;
        result?;
        Ok(payload)
    }

    /// Send an opaque message payload by type name.
    pub async fn send(&mut self, name: &str, payload: &[u8]) -> io::Result<()> {
        let tag = self.write_codec.tag_by_name(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown type name '{}'", name),
            )
        })?;
        let frame = self.write_codec.encode_message(tag, payload)?;
        let payload = UringBuf(frame);
        self.ws_write_binary(payload).await?;
        Ok(())
    }

    /// Send an Arrow table by type name.
    pub async fn send_table(
        &mut self,
        name: &str,
        table: impl Into<minarrow::TableV>,
    ) -> io::Result<()> {
        let view: minarrow::TableV = table.into();
        let tag = self.write_codec.tag_by_name(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown type name '{}'", name),
            )
        })?;

        self.write_codec
            .encode_table(tag, &view, &mut self.encode_buf)?;

        let wire_buf = std::mem::replace(&mut self.encode_buf, Vec64::with_capacity(0));
        let UringBuf(returned) = self.ws_write_binary(UringBuf(wire_buf)).await?;

        self.encode_buf = returned;
        self.encode_buf.clear();
        Ok(())
    }

    /// Read the next WS binary frame's payload as a Lightstream message.
    ///
    /// Reads the WS header into the fixed header buffer, then reads
    /// the payload directly into a Vec64. Handles ping/pong/close
    /// control frames transparently.
    pub async fn recv(&mut self) -> Option<io::Result<LightstreamMessage>> {
        if self.eof {
            return None;
        }

        loop {
            // Read enough bytes >=2 for the WS frame header
            // On error, the buffer is consumed by io_uring and lost.
            // We allocate a fresh one next time, as it means the connection is dead.
            let header_buf = std::mem::take(&mut self.header_buf);
            let header_buf = match read_exact_vec(&self.stream, header_buf, 0, 2).await {
                Ok(buf) => buf,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    self.eof = true;
                    return None;
                }
                Err(e) => {
                    self.eof = true;
                    return Some(Err(e));
                }
            };

            // Determine how many more header bytes we need
            let masked = header_buf[1] & 0x80 != 0;
            let len7 = header_buf[1] & 0x7F;
            let extra = match len7 {
                0..=125 => 0usize,
                126 => 2,
                _ => 8, // 127
            } + if masked { 4 } else { 0 };

            // Read the remaining header bytes if needed
            let header_buf = if extra > 0 {
                match read_exact_vec(&self.stream, header_buf, 2, extra).await {
                    Ok(buf) => buf,
                    Err(e) => {
                        self.eof = true;
                        return Some(Err(e));
                    }
                }
            } else {
                header_buf
            };

            // Parse the complete header
            let ws_header = match websocket::parse_header(&header_buf[..2 + extra]) {
                Some(h) => h,
                None => {
                    self.header_buf = header_buf;
                    return Some(Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "failed to parse WebSocket frame header",
                    )));
                }
            };

            self.header_buf = header_buf;
            let payload_len = match usize::try_from(ws_header.payload_len) {
                Ok(len) => len,
                Err(_) => {
                    self.eof = true;
                    return Some(Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "WebSocket payload length exceeds usize",
                    )));
                }
            };
            let max_ws_payload = self
                .limits
                .max_frame_bytes
                .saturating_add(FRAME_HEADER_SIZE);
            if let Err(error) =
                self.limits
                    .check(payload_len, max_ws_payload, "WebSocket frame bytes")
            {
                self.eof = true;
                return Some(Err(error));
            }
            if ws_header.opcode & 0x08 != 0 && (!ws_header.fin || payload_len > 125) {
                self.eof = true;
                return Some(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "fragmented or oversized WebSocket control frame",
                )));
            }

            match ws_header.opcode {
                OPCODE_BINARY => {
                    if payload_len < FRAME_HEADER_SIZE {
                        return Some(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "WebSocket payload too short for TLV header",
                        )));
                    }

                    // Read the 5-byte TLV header from the WS payload into
                    // the reusable header buffer without allocating.
                    let hdr_buf = std::mem::take(&mut self.header_buf);
                    let mut tlv_hdr =
                        match read_exact_vec(&self.stream, hdr_buf, 0, FRAME_HEADER_SIZE).await {
                            Ok(buf) => buf,
                            Err(e) => {
                                self.eof = true;
                                return Some(Err(e));
                            }
                        };

                    // Unmask the TLV header bytes if needed
                    if ws_header.masked {
                        websocket::unmask(&mut tlv_hdr[..FRAME_HEADER_SIZE], ws_header.mask_key);
                    }

                    let tag = tlv_hdr[0];
                    let tlv_payload_len =
                        u32::from_le_bytes(tlv_hdr[1..5].try_into().unwrap()) as usize;
                    self.header_buf = tlv_hdr;

                    if let Err(error) = self.limits.check(
                        tlv_payload_len,
                        self.limits.max_frame_bytes,
                        "TLV frame bytes",
                    ) {
                        self.eof = true;
                        return Some(Err(error));
                    }
                    match FRAME_HEADER_SIZE.checked_add(tlv_payload_len) {
                        Some(consumed) if consumed == payload_len => {}
                        _ => {
                            self.eof = true;
                            return Some(Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "TLV length does not match its WebSocket payload",
                            )));
                        }
                    }

                    // Read the TLV data directly into a Vec64 without copying
                    let mut payload_buf = UringBuf(Vec64::with_capacity(tlv_payload_len));
                    if tlv_payload_len > 0 {
                        payload_buf =
                            match read_exact_vec64(&self.stream, payload_buf, 0, tlv_payload_len)
                                .await
                            {
                                Ok(buf) => buf,
                                Err(e) => {
                                    self.eof = true;
                                    return Some(Err(e));
                                }
                            };

                        // Unmask in place. The mask offset continues from
                        // where the TLV header left off (i.e., FRAME_HEADER_SIZE bytes in).
                        if ws_header.masked {
                            let offset = FRAME_HEADER_SIZE % 4;
                            let shifted_key = [
                                ws_header.mask_key[offset],
                                ws_header.mask_key[(offset + 1) % 4],
                                ws_header.mask_key[(offset + 2) % 4],
                                ws_header.mask_key[(offset + 3) % 4],
                            ];
                            websocket::unmask(&mut payload_buf.0, shifted_key);
                        }
                    }

                    return Some(self.read_codec.decode_frame(tag, payload_buf.0));
                }

                OPCODE_CLOSE => {
                    // Send close response and mark EOF
                    let mut close_buf = std::mem::take(&mut self.ws_header_out);
                    let n = if self.client {
                        websocket::write_masked_close_frame(&mut close_buf, fresh_mask_key())
                    } else {
                        websocket::write_close_frame(&mut close_buf)
                    };
                    let slice = close_buf.slice(0..n);
                    let (_, slice) = self.stream.write_all(slice).await;
                    self.ws_header_out = slice.into_inner();
                    self.eof = true;
                    return None;
                }

                OPCODE_PING => {
                    // Both control buffers are allocated once per connection
                    // and sized for the largest legal masked pong.
                    let mut ping_data = std::mem::take(&mut self.header_buf);
                    if payload_len > 0 {
                        ping_data = match read_exact_vec(
                            &self.stream,
                            ping_data,
                            0,
                            payload_len,
                        )
                        .await
                        {
                            Ok(buf) => buf,
                            Err(e) => {
                                self.eof = true;
                                return Some(Err(e));
                            }
                        };
                    }
                    if ws_header.masked {
                        websocket::unmask(&mut ping_data[..payload_len], ws_header.mask_key);
                    }

                    let mut pong_buf = std::mem::take(&mut self.ws_header_out);
                    let n = if self.client {
                        websocket::write_masked_pong_frame(
                            &mut pong_buf,
                            &ping_data[..payload_len],
                            fresh_mask_key(),
                        )
                    } else {
                        websocket::write_pong_frame(&mut pong_buf, &ping_data[..payload_len])
                    };
                    let slice = pong_buf.slice(0..n);
                    let (result, slice) = self.stream.write_all(slice).await;
                    self.ws_header_out = slice.into_inner();
                    self.header_buf = ping_data;
                    if let Err(error) = result {
                        self.eof = true;
                        return Some(Err(error));
                    }
                    // Loop to read the next frame
                    continue;
                }

                OPCODE_PONG => {
                    if payload_len > 0 {
                        let buf = std::mem::take(&mut self.header_buf);
                        self.header_buf = match read_exact_vec(
                            &self.stream,
                            buf,
                            0,
                            payload_len,
                        )
                        .await
                        {
                            Ok(buf) => buf,
                            Err(error) => {
                                self.eof = true;
                                return Some(Err(error));
                            }
                        };
                    }
                    continue;
                }

                _ => {
                    self.eof = true;
                    return Some(Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported WebSocket opcode",
                    )));
                }
            }
        }
    }

    /// Flush. No-op for io_uring.
    pub async fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Shut down by sending a WebSocket close frame.
    pub async fn shutdown(&mut self) -> io::Result<()> {
        let mut close_buf = std::mem::take(&mut self.ws_header_out);
        let n = if self.client {
            websocket::write_masked_close_frame(&mut close_buf, fresh_mask_key())
        } else {
            websocket::write_close_frame(&mut close_buf)
        };
        let slice = close_buf.slice(0..n);
        let (result, slice) = self.stream.write_all(slice).await;
        self.ws_header_out = slice.into_inner();
        result?;
        self.stream.shutdown(std::net::Shutdown::Write)
    }

    /// Send a protobuf message by type name.
    #[cfg(feature = "protobuf")]
    pub async fn send_proto<M: prost::Message>(&mut self, name: &str, msg: &M) -> io::Result<()> {
        let bytes = msg.encode_to_vec();
        self.send(name, &bytes).await
    }

    /// Send a MessagePack-encoded message by type name.
    #[cfg(feature = "msgpack")]
    pub async fn send_msgpack<M: serde::Serialize>(
        &mut self,
        name: &str,
        msg: &M,
    ) -> io::Result<()> {
        let bytes = super::connection::encode_msgpack(msg)?;
        self.send(name, &bytes).await
    }
}
