// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous WebSocket byte stream adapters
//!
//! Blocking counterparts to [`WsRead`] and [`WsWrite`] using `std::io`.
//! Support WebSocket I/O on dedicated threads without an async runtime.
//!
//! The adapters handle transport I/O using [`WsFrameDecoder`] for decoding
//! and [`WsFrameEncoder`] for encoding. These protocol implementations are
//! shared with the asynchronous adapters, ensuring compatible frame handling.
//! Payload bytes are read directly from the socket into the caller's buffer.
//!
//! [`SyncWsRead`] implements `std::io::Read` for binary payload streams used
//! by table readers. [`SyncWsRead::read_message`] reads complete text or
//! binary messages.
//!
//! [`WsRead`]: crate::models::streams::websocket::WsRead
//! [`WsWrite`]: crate::models::streams::websocket::WsWrite

use std::io::{self, Read, Write};
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

use crate::models::decoders::websocket::{WsDecodeResult, WsFrameDecoder};
use crate::models::encoders::websocket::WsFrameEncoder;

/// Largest buffer growth per read in [`SyncWsRead::read_message`].
///
/// Message buffers grow by the bytes that have arrived, read in steps of at
/// most this size, and never by a payload length the peer declared.
const MESSAGE_READ_CHUNK: usize = 64 * 1024;

/// Blocking WebSocket read adapter over a `std::io::Read` stream.
///
/// Exposes unmasked binary payload bytes through `std::io::Read`, and whole
/// text and binary messages through [`Self::read_message`]. Ping and close
/// frames are handled transparently, with pongs sent immediately through the
/// shared write half.
///
/// The `io::Read` implementation reads through
/// [`WsFrameDecoder::decode_next`], which rejects text frames with
/// `io::ErrorKind::InvalidData`, as [`WsRead`] does.
///
/// [`WsRead`]: crate::models::streams::websocket::WsRead
pub struct SyncWsRead<R, W> {
    inner: R,
    writer: Arc<Mutex<W>>,
    decoder: WsFrameDecoder,
}

impl<R: Read, W: Write> SyncWsRead<R, W> {
    /// Wrap a raw TCP read half as a WebSocket reader.
    ///
    /// The `writer` is the shared write half used for sending pong
    /// responses to pings. The WebSocket handshake must already be complete.
    pub fn new(inner: R, writer: Arc<Mutex<W>>) -> Self {
        Self {
            inner,
            writer,
            decoder: WsFrameDecoder::new(),
        }
    }

    /// Wrap a raw TCP read half as a client-side WebSocket reader.
    ///
    /// Identical to [`SyncWsRead::new`] except pong responses are masked,
    /// as RFC 6455 requires of every client-to-server frame.
    pub fn new_client(inner: R, writer: Arc<Mutex<W>>) -> Self {
        Self {
            inner,
            writer,
            decoder: WsFrameDecoder::new_client(),
        }
    }

    /// Read the next whole text or binary message into `buf`.
    ///
    /// Clears `buf`, then appends the payloads of the message's first frame
    /// and its continuation frames in order, up to the final fragment. Pings
    /// that arrive between fragments are answered. Returns the message's
    /// opcode, which compares against `OPCODE_TEXT` and `OPCODE_BINARY` in
    /// [`crate::models::frames::websocket`].
    ///
    /// Returns `Ok(None)` at a close frame or at end of input. Text payloads
    /// are returned as received, without UTF-8 validation.
    ///
    /// ## Errors
    ///
    /// - `io::ErrorKind::InvalidData` once the message would exceed
    ///   `max_len` bytes. `buf` grows by the bytes that have arrived, never
    ///   by a length the peer declared.
    /// - The message and frame errors of
    ///   [`WsFrameDecoder::decode_next_message`], and transport errors.
    pub fn read_message(&mut self, buf: &mut Vec<u8>, max_len: usize) -> io::Result<Option<u8>> {
        buf.clear();
        loop {
            match self.decoder.decode_next_message()? {
                WsDecodeResult::NeedMore => {
                    let remaining = self.decoder.payload_remaining();
                    if buf.len().saturating_add(remaining) > max_len {
                        self.decoder.close();
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "WebSocket message exceeds the maximum length",
                        ));
                    }
                    let start = buf.len();
                    buf.resize(start + remaining.min(MESSAGE_READ_CHUNK), 0);
                    match fill(&mut self.inner, &mut self.decoder, &mut buf[start..]) {
                        Ok(delivered) => buf.truncate(start + delivered),
                        Err(e) => {
                            buf.truncate(start);
                            return Err(e);
                        }
                    }
                }
                WsDecodeResult::Pong => send_pong(&self.writer, &mut self.decoder)?,
                WsDecodeResult::Message(opcode) => return Ok(Some(opcode)),
                WsDecodeResult::Closed => return Ok(None),
            }
        }
    }
}

/// Read transport bytes into the region the decoder returns for `target`,
/// and return the number of payload bytes delivered into `target`.
///
/// Retries interrupted reads, and marks the decoder closed at end of input.
fn fill<R: Read>(inner: &mut R, decoder: &mut WsFrameDecoder, target: &mut [u8]) -> io::Result<usize> {
    // SAFETY: `MaybeUninit<u8>` has the layout of `u8`. Only initialised
    // bytes are written through this view.
    let target = unsafe { &mut *(target as *mut [u8] as *mut [MaybeUninit<u8>]) };
    loop {
        let region = decoder.spare(target);
        // SAFETY: `target` is initialised, and the decoder's own regions are
        // always initialised, which makes every region `spare` returns here
        // initialised.
        let region = unsafe { &mut *(region as *mut [MaybeUninit<u8>] as *mut [u8]) };
        match inner.read(region) {
            Ok(0) => {
                decoder.close();
                return Ok(0);
            }
            // SAFETY: `read` initialised the first `n` bytes of the region,
            // and `target` is the buffer passed to `spare`.
            Ok(n) => return Ok(unsafe { decoder.advance(n, target) }),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Write the decoder's pending pong frame through the shared write half.
fn send_pong<W: Write>(writer: &Mutex<W>, decoder: &mut WsFrameDecoder) -> io::Result<()> {
    let mut guard = writer.lock().unwrap();
    let pong = decoder.pong();
    let len = pong.len();
    guard.write_all(pong)?;
    guard.flush()?;
    decoder.advance_pong(len);
    Ok(())
}

impl<R: Read, W: Write> Read for SyncWsRead<R, W> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.decoder.decode_next()? {
                WsDecodeResult::NeedMore => {
                    // Payload reads into an empty buffer return without reading.
                    // An empty region would otherwise read as end of input.
                    if self.decoder.payload_remaining() > 0 && buf.is_empty() {
                        return Ok(0);
                    }
                    let n = fill(&mut self.inner, &mut self.decoder, buf)?;
                    if n > 0 {
                        return Ok(n);
                    }
                }
                WsDecodeResult::Pong => send_pong(&self.writer, &mut self.decoder)?,
                WsDecodeResult::Closed | WsDecodeResult::Message(_) => return Ok(0),
            }
        }
    }
}

/// Blocking WebSocket write adapter over a `std::io::Write` stream.
///
/// Wraps the bytes of each `write` call in one WebSocket binary frame. The
/// header is written from a stack buffer, then the caller's payload follows,
/// or its masked copy for the connecting peer.
///
/// Designed for use with `write_all`, where each call represents a complete
/// message.
pub struct SyncWsWrite<W> {
    inner: Arc<Mutex<W>>,
    /// Role-dependent framing, including client frame masking, and the
    /// progress of the frame being written.
    encoder: WsFrameEncoder,
    /// Set once the close frame has been written.
    closed: bool,
}

impl<W: Write> SyncWsWrite<W> {
    /// Create a server-side WebSocket write adapter from a raw TCP
    /// write half. Frames are written unmasked, as RFC 6455 requires of
    /// server-to-client traffic.
    ///
    /// Returns the shared writer reference needed by [`SyncWsRead`] for pong
    /// responses, and the adapter for application data. The WebSocket
    /// handshake must already be complete.
    pub fn new(inner: W) -> (Arc<Mutex<W>>, Self) {
        Self::with_encoder(inner, WsFrameEncoder::new())
    }

    /// Create a client-side WebSocket write adapter from a raw TCP
    /// write half. Every frame is masked with a fresh key, as RFC 6455
    /// requires of client-to-server traffic.
    pub fn new_client(inner: W) -> (Arc<Mutex<W>>, Self) {
        Self::with_encoder(inner, WsFrameEncoder::new_client())
    }

    fn with_encoder(inner: W, encoder: WsFrameEncoder) -> (Arc<Mutex<W>>, Self) {
        let shared = Arc::new(Mutex::new(inner));
        let writer = Self {
            inner: Arc::clone(&shared),
            encoder,
            closed: false,
        };
        (shared, writer)
    }

    /// Write a normal-closure close frame for the connection's role, then
    /// flush the stream. Later calls flush only.
    pub fn close(&mut self) -> io::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        if !self.closed {
            self.encoder.begin_close();
            write_frame(&mut *guard, &mut self.encoder, &[])?;
            self.closed = true;
        }
        guard.flush()
    }
}

/// Write the encoder's frame in progress until it is complete.
fn write_frame<W: Write>(
    inner: &mut W,
    encoder: &mut WsFrameEncoder,
    payload: &[u8],
) -> io::Result<()> {
    while !encoder.is_complete() {
        match inner.write(encoder.pending(payload)) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => encoder.advance(n),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl<W: Write> Write for SyncWsWrite<W> {
    /// Returns an error once [`SyncWsWrite::close`] has been called.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.closed {
            return Err(io::Error::other("WebSocket writer already closed"));
        }
        let mut guard = self.inner.lock().unwrap();
        self.encoder.begin_binary(buf);
        write_frame(&mut *guard, &mut self.encoder, buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.lock().unwrap().flush()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// Zero-capacity reads answer pings ahead of a data frame and return
    /// without closing, and the next read with capacity returns the payload.
    #[test]
    fn zero_capacity_read_answers_ping_without_closing() {
        let wire = vec![0x89, 1, b'p', 0x82, 2, b'o', b'k'];
        let (shared, _writer) = SyncWsWrite::new(Vec::new());
        let mut reader = SyncWsRead::new(Cursor::new(wire), Arc::clone(&shared));

        assert_eq!(reader.read(&mut []).unwrap(), 0);
        assert_eq!(*shared.lock().unwrap(), [0x8A, 1, b'p']);

        let mut payload = [0u8; 2];
        reader.read_exact(&mut payload).unwrap();
        assert_eq!(&payload, b"ok");
    }
}
