// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebSocket frame decoder
//!
//! WebSocket protocol decoding shared by asynchronous and blocking readers.
//!
//! [`WsFrameDecoder`] processes bytes supplied by transport readers. It
//! parses frame headers, validates payload lengths and frame types, unmasks
//! payloads, generates pong responses, recognises close frames and assembles
//! fragmented messages. Pong responses follow the connection's client or
//! server role.
//!
//! ## Transport integration
//!
//! Decoding is synchronous. Readers handle transport I/O using buffer regions
//! supplied by the decoder. Header and control bytes go into internal buffers.
//! Payload bytes go directly into the caller's buffer to avoid intermediate
//! copies.
//!
//! Readers commit the number of bytes read and write any pong frames produced
//! by the decoder.
//! [`WsRead`](crate::models::streams::websocket::WsRead) uses an `AsyncRead`
//! source.
//! [`SyncWsRead`](crate::models::streams::sync_websocket::SyncWsRead) uses a
//! blocking `std::io::Read` source.
//!
//! ## Decoding modes
//!
//! - [`WsFrameDecoder::decode_next`] exposes binary payloads as a byte stream
//!   for table readers. It accepts binary and continuation frames in any
//!   order and rejects text frames.
//! - [`WsFrameDecoder::decode_next_message`] assembles complete text and
//!   binary messages according to the continuation rules in RFC 6455.
//!   It reports message boundaries and the corresponding opcodes.

use std::io;
use std::mem::MaybeUninit;

use crate::models::frames::websocket::{
    self, MAX_HEADER_LEN, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_CONTINUATION, OPCODE_PING,
    OPCODE_PONG, OPCODE_TEXT, WsHeader, fresh_mask_key,
};

/// Largest control frame payload permitted by RFC 6455.
const MAX_CONTROL_PAYLOAD: usize = 125;

/// Largest pong frame: a 2-byte header, a 4-byte masking key for the
/// connecting peer and the echoed control payload.
const MAX_PONG_LEN: usize = 6 + MAX_CONTROL_PAYLOAD;

/// Read state machine for WebSocket frame parsing.
enum WsReadState {
    /// Accumulating the WS frame header (2-14 bytes).
    Header { filled: usize },
    /// Reading payload bytes into the caller's buffer.
    Payload {
        remaining: usize,
        masked: bool,
        mask_key: [u8; 4],
        mask_offset: usize,
    },
    /// Reading a ping payload before sending the pong.
    ReadingPing {
        len: usize,
        filled: usize,
        masked: bool,
        mask_key: [u8; 4],
    },
    /// Consuming a valid control payload that needs no application action.
    IgnoringControl { remaining: usize },
    /// Sending a pong response. Transitions to Header when done.
    SendingPong { len: usize, written: usize },
    /// Connection closed.
    Closed,
}

/// Outcome of one step of the state machine, before the decoding form
/// applies its frame rules.
enum Step {
    NeedMore,
    Pong,
    Closed,
    /// A data frame header was parsed. Its payload follows on the transport.
    Frame(WsHeader),
}

/// Outcome of one [`WsFrameDecoder::decode_next`] or
/// [`WsFrameDecoder::decode_next_message`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsDecodeResult {
    /// More transport bytes are required. Read them into the region returned
    /// by [`WsFrameDecoder::spare`] and commit them with
    /// [`WsFrameDecoder::advance`].
    NeedMore,
    /// The decoder has a pong frame ready. Write [`WsFrameDecoder::pong`] to
    /// the transport and commit the written bytes with
    /// [`WsFrameDecoder::advance_pong`].
    Pong,
    /// The current message is complete, with the given opcode, `OPCODE_TEXT`
    /// or `OPCODE_BINARY`. Reported by
    /// [`WsFrameDecoder::decode_next_message`] only.
    Message(u8),
    /// The peer sent a close frame, the input ended, or the decoder rejected
    /// a frame.
    Closed,
}

/// Synchronous WebSocket frame decoder for bytes supplied by transport readers.
///
/// Parses frame headers in a fixed buffer and writes data frame payloads
/// directly into the caller's buffer. Handles ping, pong and close frames,
/// producing pong responses for readers to write.
///
/// ## Behaviour
///
/// - Rejects fragmented control frames and control payloads exceeding
///   125 bytes with `io::ErrorKind::InvalidData` before reading payload data.
/// - Rejects reserved opcodes with `io::ErrorKind::InvalidData`.
/// - Returns `io::ErrorKind::InvalidData` for malformed frame headers and
///   leaves the decoder open. All other rejections close the decoder.
/// - Masks pong responses in the client role, as required by RFC 6455 for
///   client-to-server frames.
/// - Returns [`WsDecodeResult::Closed`] on all decode calls after closure.
pub struct WsFrameDecoder {
    state: WsReadState,
    header_buf: [MaybeUninit<u8>; MAX_HEADER_LEN],
    /// Ping payload, or the discard region for ignored control payloads.
    control_buf: [MaybeUninit<u8>; MAX_CONTROL_PAYLOAD],
    /// Pending pong frame, written in place for each ping.
    pong_buf: [u8; MAX_PONG_LEN],
    /// Client pong responses are masked per RFC 6455.
    client: bool,
    /// Opcode of the message in progress, for message decoding.
    message: Option<u8>,
    /// Whether the latest data frame of the message in progress is final.
    message_fin: bool,
}

impl WsFrameDecoder {
    /// Construct a decoder for the accepting peer of a connection, which
    /// sends unmasked pong responses.
    pub fn new() -> Self {
        Self {
            state: WsReadState::Header { filled: 0 },
            // The decoder's own regions start zeroed and only ever receive
            // initialised bytes, which keeps them initialised throughout.
            header_buf: [MaybeUninit::new(0); MAX_HEADER_LEN],
            control_buf: [MaybeUninit::new(0); MAX_CONTROL_PAYLOAD],
            pong_buf: [0u8; MAX_PONG_LEN],
            client: false,
            message: None,
            message_fin: false,
        }
    }

    /// Construct a decoder for the connecting peer of a connection, which
    /// sends masked pong responses.
    pub fn new_client() -> Self {
        Self {
            client: true,
            ..Self::new()
        }
    }

    /// Get the region for the next transport read.
    ///
    /// `target` is the caller's payload buffer. While a data frame payload is
    /// being read, the region is the front of `target`, limited to the
    /// frame's remaining payload. Otherwise it is the unfilled part of the
    /// decoder's frame header, ping payload or control discard region.
    ///
    /// The decoder's own regions are always initialised. The returned region
    /// is therefore initialised whenever `target` is. The caller reads into
    /// this region, then calls [`Self::advance`] with the count and the same
    /// `target`. The region is empty only when `target` is empty during a
    /// payload read.
    #[inline]
    pub fn spare<'a>(
        &'a mut self,
        target: &'a mut [MaybeUninit<u8>],
    ) -> &'a mut [MaybeUninit<u8>] {
        match self.state {
            WsReadState::Header { filled } => {
                let need = header_len_needed(&self.header_buf, filled);
                &mut self.header_buf[filled..need]
            }
            WsReadState::Payload { remaining, .. } => {
                let len = remaining.min(target.len());
                &mut target[..len]
            }
            WsReadState::ReadingPing { len, filled, .. } => &mut self.control_buf[filled..len],
            WsReadState::IgnoringControl { remaining } => {
                &mut self.control_buf[..remaining.min(MAX_CONTROL_PAYLOAD)]
            }
            _ => &mut [],
        }
    }

    /// Commit `n` bytes read into the region returned by the preceding
    /// [`Self::spare`] call, and return the number of payload bytes delivered
    /// into the front of `target`.
    ///
    /// Payload bytes are unmasked in place. Returns 0 when the bytes belong
    /// to a frame header or a control frame.
    ///
    /// # Safety
    ///
    /// `target` must be the buffer passed to that [`Self::spare`] call, every
    /// byte in the first `n` elements of the returned region must have been
    /// initialised, and `n` must not exceed its length.
    #[inline]
    pub unsafe fn advance(&mut self, n: usize, target: &mut [MaybeUninit<u8>]) -> usize {
        match &mut self.state {
            WsReadState::Header { filled } => {
                *filled += n;
                0
            }
            WsReadState::ReadingPing {
                filled,
                masked,
                mask_key,
                ..
            } => {
                if *masked {
                    let offset = *filled;
                    // SAFETY: the decoder's own regions are always initialised.
                    let bytes = unsafe { assume_init_mut(&mut self.control_buf[offset..offset + n]) };
                    for (i, byte) in bytes.iter_mut().enumerate() {
                        *byte ^= mask_key[(offset + i) % 4];
                    }
                }
                *filled += n;
                0
            }
            WsReadState::IgnoringControl { remaining } => {
                *remaining -= n;
                0
            }
            WsReadState::Payload {
                remaining,
                masked,
                mask_key,
                mask_offset,
            } => {
                // SAFETY: the caller initialised the first `n` bytes of the
                // region, which is the front of `target`.
                let payload = unsafe { assume_init_mut(&mut target[..n]) };
                if *masked {
                    // Rotate the key to the frame's running mask offset.
                    let key = [
                        mask_key[*mask_offset % 4],
                        mask_key[(*mask_offset + 1) % 4],
                        mask_key[(*mask_offset + 2) % 4],
                        mask_key[(*mask_offset + 3) % 4],
                    ];
                    websocket::unmask(payload, key);
                    *mask_offset = (*mask_offset + n) % 4;
                }
                *remaining -= n;
                if *remaining == 0 {
                    self.state = WsReadState::Header { filled: 0 };
                }
                n
            }
            _ => {
                debug_assert_eq!(n, 0, "advance called without a spare region");
                0
            }
        }
    }

    /// Return the payload bytes of the current data frame that remain on the
    /// transport, or 0 outside a data frame payload.
    #[inline]
    pub fn payload_remaining(&self) -> usize {
        match self.state {
            WsReadState::Payload { remaining, .. } => remaining,
            _ => 0,
        }
    }

    /// Return the unwritten bytes of the pending pong frame.
    ///
    /// Empty unless the decoder reports [`WsDecodeResult::Pong`].
    #[inline]
    pub fn pong(&self) -> &[u8] {
        match &self.state {
            WsReadState::SendingPong { len, written } => &self.pong_buf[*written..*len],
            _ => &[],
        }
    }

    /// Commit `n` bytes of the pending pong frame as written to the transport.
    #[inline]
    pub fn advance_pong(&mut self, n: usize) {
        if let WsReadState::SendingPong { written, .. } = &mut self.state {
            *written += n;
        }
    }

    /// Mark the connection closed.
    ///
    /// Readers call this when the transport reaches end of input.
    #[inline]
    pub fn close(&mut self) {
        self.state = WsReadState::Closed;
    }

    /// Advance the decoder over the bytes committed so far, reading the
    /// connection as a byte stream of binary payloads.
    ///
    /// Binary and continuation frames are accepted in any order, and their
    /// payloads are delivered through [`Self::spare`] and [`Self::advance`].
    /// Text frames close the decoder and return
    /// `io::ErrorKind::InvalidData`.
    ///
    /// Malformed frame headers, oversized or fragmented control frames and
    /// reserved opcodes also return `io::ErrorKind::InvalidData`.
    pub fn decode_next(&mut self) -> io::Result<WsDecodeResult> {
        loop {
            match self.step()? {
                Step::NeedMore => return Ok(WsDecodeResult::NeedMore),
                Step::Pong => return Ok(WsDecodeResult::Pong),
                Step::Closed => return Ok(WsDecodeResult::Closed),
                Step::Frame(header) => {
                    if header.opcode != OPCODE_BINARY && header.opcode != OPCODE_CONTINUATION {
                        self.state = WsReadState::Closed;
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unsupported WebSocket opcode",
                        ));
                    }
                }
            }
        }
    }

    /// Advance the decoder over the bytes committed so far, reading the
    /// connection as whole text and binary messages.
    ///
    /// Message payloads are delivered through [`Self::spare`] and
    /// [`Self::advance`], with the first frame followed by its continuation
    /// frames up to the final fragment. Returns [`WsDecodeResult::Message`]
    /// with the message's opcode once the final fragment has been delivered.
    ///
    /// A continuation frame outside a message, or a new message frame before
    /// the previous message ended, closes the decoder and returns
    /// `io::ErrorKind::InvalidData`. Frame errors are those of
    /// [`Self::decode_next`].
    pub fn decode_next_message(&mut self) -> io::Result<WsDecodeResult> {
        loop {
            if let (Some(opcode), true, WsReadState::Header { filled: 0 }) =
                (self.message, self.message_fin, &self.state)
            {
                self.message = None;
                self.message_fin = false;
                return Ok(WsDecodeResult::Message(opcode));
            }
            match self.step()? {
                Step::NeedMore => return Ok(WsDecodeResult::NeedMore),
                Step::Pong => return Ok(WsDecodeResult::Pong),
                Step::Closed => return Ok(WsDecodeResult::Closed),
                Step::Frame(header) => {
                    let opcode = match (self.message, header.opcode) {
                        (None, OPCODE_TEXT | OPCODE_BINARY) => header.opcode,
                        (Some(opcode), OPCODE_CONTINUATION) => opcode,
                        _ => {
                            self.state = WsReadState::Closed;
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "unexpected WebSocket frame within a message",
                            ));
                        }
                    };
                    self.message = Some(opcode);
                    self.message_fin = header.fin;
                }
            }
        }
    }

    /// Advance the state machine to the next point that requires transport
    /// IO, a pong write or a decision on a data frame.
    fn step(&mut self) -> io::Result<Step> {
        loop {
            match &mut self.state {
                WsReadState::Closed => return Ok(Step::Closed),

                WsReadState::Header { filled } => {
                    let filled = *filled;
                    if filled < header_len_needed(&self.header_buf, filled) {
                        return Ok(Step::NeedMore);
                    }

                    // SAFETY: the decoder's own regions are always initialised.
                    let header_bytes = unsafe { assume_init(&self.header_buf[..filled]) };
                    let ws = match websocket::parse_header(header_bytes) {
                        Some(h) => h,
                        None => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "malformed WebSocket frame header",
                            ));
                        }
                    };

                    let payload_len = match checked_payload_len(ws) {
                        Ok(len) => len,
                        Err(error) => {
                            self.state = WsReadState::Closed;
                            return Err(error);
                        }
                    };

                    match ws.opcode {
                        OPCODE_BINARY | OPCODE_TEXT | OPCODE_CONTINUATION => {
                            self.state = if payload_len == 0 {
                                WsReadState::Header { filled: 0 }
                            } else {
                                WsReadState::Payload {
                                    remaining: payload_len,
                                    masked: ws.masked,
                                    mask_key: ws.mask_key,
                                    mask_offset: 0,
                                }
                            };
                            return Ok(Step::Frame(ws));
                        }
                        OPCODE_CLOSE => {
                            self.state = WsReadState::Closed;
                            return Ok(Step::Closed);
                        }
                        OPCODE_PING => {
                            if payload_len > 0 {
                                self.state = WsReadState::ReadingPing {
                                    len: payload_len,
                                    filled: 0,
                                    masked: ws.masked,
                                    mask_key: ws.mask_key,
                                };
                            } else {
                                // Empty ping - send empty pong immediately
                                self.state = WsReadState::SendingPong {
                                    len: build_pong(self.client, &[], &mut self.pong_buf),
                                    written: 0,
                                };
                            }
                        }
                        OPCODE_PONG => {
                            self.state = if payload_len == 0 {
                                WsReadState::Header { filled: 0 }
                            } else {
                                WsReadState::IgnoringControl {
                                    remaining: payload_len,
                                }
                            };
                        }
                        _ => {
                            self.state = WsReadState::Closed;
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "unsupported WebSocket opcode",
                            ));
                        }
                    }
                }

                WsReadState::Payload { .. } => return Ok(Step::NeedMore),

                WsReadState::ReadingPing { len, filled, .. } => {
                    if *filled < *len {
                        return Ok(Step::NeedMore);
                    }
                    // Build pong frame and transition to sending it
                    let len = *len;
                    // SAFETY: the decoder's own regions are always initialised.
                    let payload = unsafe { assume_init(&self.control_buf[..len]) };
                    self.state = WsReadState::SendingPong {
                        len: build_pong(self.client, payload, &mut self.pong_buf),
                        written: 0,
                    };
                }

                WsReadState::IgnoringControl { remaining } => {
                    if *remaining > 0 {
                        return Ok(Step::NeedMore);
                    }
                    self.state = WsReadState::Header { filled: 0 };
                }

                WsReadState::SendingPong { len, written } => {
                    if *written < *len {
                        return Ok(Step::Pong);
                    }
                    self.state = WsReadState::Header { filled: 0 };
                }
            }
        }
    }
}

impl Default for WsFrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Reinterpret initialised bytes as `u8`.
///
/// # Safety
///
/// Every element of `bytes` must be initialised.
unsafe fn assume_init(bytes: &[MaybeUninit<u8>]) -> &[u8] {
    // SAFETY: `MaybeUninit<u8>` has the layout of `u8`, and the caller
    // guarantees every element is initialised.
    unsafe { &*(bytes as *const [MaybeUninit<u8>] as *const [u8]) }
}

/// Reinterpret initialised bytes as mutable `u8`.
///
/// # Safety
///
/// Every element of `bytes` must be initialised.
unsafe fn assume_init_mut(bytes: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    // SAFETY: `MaybeUninit<u8>` has the layout of `u8`, and the caller
    // guarantees every element is initialised.
    unsafe { &mut *(bytes as *mut [MaybeUninit<u8>] as *mut [u8]) }
}

/// Header bytes required to parse the frame, given the `filled` bytes
/// accumulated so far.
fn header_len_needed(header_buf: &[MaybeUninit<u8>; MAX_HEADER_LEN], filled: usize) -> usize {
    if filled < 2 {
        return 2;
    }
    // SAFETY: the decoder's own regions are always initialised.
    let second = unsafe { header_buf[1].assume_init() };
    let masked = second & 0x80 != 0;
    let len7 = second & 0x7F;
    let extra = match len7 {
        0..=125 => 0usize,
        126 => 2,
        _ => 8,
    } + if masked { 4 } else { 0 };
    2 + extra
}

/// Validate the frame's payload length, rejecting fragmented or oversized
/// control frames.
fn checked_payload_len(header: WsHeader) -> io::Result<usize> {
    let payload_len = usize::try_from(header.payload_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "WebSocket payload length exceeds usize",
        )
    })?;
    if header.opcode & 0x08 != 0 && (!header.fin || payload_len > MAX_CONTROL_PAYLOAD) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fragmented or oversized WebSocket control frame",
        ));
    }
    Ok(payload_len)
}

/// Build the pong frame for `payload` according to the connection role into
/// `pong`, and return its length.
fn build_pong(client: bool, payload: &[u8], pong: &mut [u8; MAX_PONG_LEN]) -> usize {
    if client {
        websocket::write_masked_pong_frame(pong, payload, fresh_mask_key())
    } else {
        websocket::write_pong_frame(pong, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decoding form under test.
    #[derive(Clone, Copy)]
    enum Form {
        Bytes,
        Messages,
    }

    /// Outcome of feeding a wire image through a decoder.
    #[derive(Debug, Default)]
    struct Fed {
        payload: Vec<u8>,
        pongs: Vec<u8>,
        messages: Vec<(u8, Vec<u8>)>,
        error: Option<io::Error>,
        closed: bool,
    }

    /// Feed `wire` to `decoder` at most `step` bytes per read, with payload
    /// targets of `target_len` bytes, until it closes, errors or runs out of
    /// input.
    fn feed(
        decoder: &mut WsFrameDecoder,
        form: Form,
        wire: &[u8],
        step: usize,
        target_len: usize,
    ) -> Fed {
        let mut fed = Fed::default();
        let mut pos = 0;
        let mut message = Vec::new();
        loop {
            let result = match form {
                Form::Bytes => decoder.decode_next(),
                Form::Messages => decoder.decode_next_message(),
            };
            match result {
                Err(e) => {
                    fed.error = Some(e);
                    return fed;
                }
                Ok(WsDecodeResult::Closed) => {
                    fed.closed = true;
                    return fed;
                }
                Ok(WsDecodeResult::Pong) => {
                    let pong = decoder.pong().to_vec();
                    fed.pongs.extend_from_slice(&pong);
                    decoder.advance_pong(pong.len());
                }
                Ok(WsDecodeResult::Message(opcode)) => {
                    fed.messages.push((opcode, std::mem::take(&mut message)));
                }
                Ok(WsDecodeResult::NeedMore) => {
                    if pos == wire.len() {
                        return fed;
                    }
                    let mut target = vec![MaybeUninit::new(0u8); target_len];
                    let region = decoder.spare(&mut target);
                    let n = region.len().min(step).min(wire.len() - pos);
                    for (slot, byte) in region[..n].iter_mut().zip(&wire[pos..pos + n]) {
                        slot.write(*byte);
                    }
                    pos += n;
                    let delivered = unsafe { decoder.advance(n, &mut target) };
                    let bytes = unsafe { assume_init(&target[..delivered]) };
                    match form {
                        Form::Bytes => fed.payload.extend_from_slice(bytes),
                        Form::Messages => message.extend_from_slice(bytes),
                    }
                }
            }
        }
    }

    /// Masked frame with the given first header byte.
    fn masked(first_byte: u8, payload: &[u8]) -> Vec<u8> {
        let key = [0x37u8, 0xFA, 0x21, 0x3D];
        let mut header = [0u8; MAX_HEADER_LEN];
        let n = websocket::write_masked_binary_header(&mut header, payload.len(), key);
        header[0] = first_byte;
        let mut frame = header[..n].to_vec();
        frame.extend(payload.iter().enumerate().map(|(i, b)| b ^ key[i % 4]));
        frame
    }

    #[test]
    fn byte_stream_accepts_binary_and_continuation_frames_in_any_order() {
        let mut wire = masked(0x00, b"ab");
        wire.extend(masked(0x82, b"cd"));
        wire.extend(masked(0x80, b"ef"));
        wire.extend([0x88, 0]);
        for step in [1, 3, 64] {
            let fed = feed(&mut WsFrameDecoder::new(), Form::Bytes, &wire, step, 1);
            assert!(fed.closed && fed.error.is_none());
            assert_eq!(fed.payload, b"abcdef");
        }
    }

    #[test]
    fn byte_stream_rejects_text_frames_and_closes() {
        let mut decoder = WsFrameDecoder::new();
        let fed = feed(&mut decoder, Form::Bytes, &[0x81, 1, b'x'], 64, 64);
        let error = fed.error.unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "unsupported WebSocket opcode");
        assert_eq!(decoder.decode_next().unwrap(), WsDecodeResult::Closed);
    }

    #[test]
    fn messages_join_fragments_across_pings() {
        let mut wire = masked(0x01, b"bid ");
        wire.extend(masked(0x89, b"p"));
        wire.extend(masked(0x00, b"101."));
        wire.extend(masked(0x80, b"25"));
        wire.extend(masked(0x82, &[]));
        wire.extend(masked(0x82, &[9, 8]));
        for step in [1, 5, 64] {
            let fed = feed(&mut WsFrameDecoder::new(), Form::Messages, &wire, step, 3);
            assert!(fed.error.is_none());
            assert_eq!(
                fed.messages,
                vec![
                    (OPCODE_TEXT, b"bid 101.25".to_vec()),
                    (OPCODE_BINARY, Vec::new()),
                    (OPCODE_BINARY, vec![9, 8]),
                ]
            );
            assert_eq!(fed.pongs, [0x8A, 1, b'p']);
        }
    }

    #[test]
    fn messages_reject_continuation_outside_a_message() {
        let mut decoder = WsFrameDecoder::new();
        let fed = feed(&mut decoder, Form::Messages, &masked(0x80, b"x"), 64, 64);
        let error = fed.error.unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "unexpected WebSocket frame within a message");
        assert_eq!(decoder.decode_next_message().unwrap(), WsDecodeResult::Closed);
    }

    #[test]
    fn messages_reject_a_new_message_before_the_previous_one_ends() {
        let mut wire = masked(0x01, b"a");
        wire.extend(masked(0x82, b"b"));
        let fed = feed(&mut WsFrameDecoder::new(), Form::Messages, &wire, 64, 64);
        assert_eq!(fed.error.unwrap().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn oversized_control_frames_are_rejected_and_close_the_decoder() {
        let mut decoder = WsFrameDecoder::new();
        let fed = feed(&mut decoder, Form::Bytes, &[0x89, 126, 0, 126], 64, 64);
        assert_eq!(fed.error.unwrap().kind(), io::ErrorKind::InvalidData);
        assert_eq!(decoder.decode_next().unwrap(), WsDecodeResult::Closed);
    }

    #[test]
    fn client_pongs_are_masked() {
        let fed = feed(&mut WsFrameDecoder::new_client(), Form::Bytes, &[0x89, 2, b'h', b'i'], 64, 64);
        let pong = fed.pongs;
        assert_eq!(&pong[..2], &[0x8A, 0x82]);
        let key = [pong[2], pong[3], pong[4], pong[5]];
        assert_eq!([pong[6] ^ key[0], pong[7] ^ key[1]], *b"hi");
    }

    /// Pings with the largest control payload are answered in full in both
    /// roles, and consecutive pings each get their own pong.
    #[test]
    fn largest_pings_are_answered_in_both_roles() {
        let payload: Vec<u8> = (0..MAX_CONTROL_PAYLOAD as u8).collect();
        let mut wire = vec![0x89, MAX_CONTROL_PAYLOAD as u8];
        wire.extend_from_slice(&payload);
        wire.extend_from_slice(&[0x89, 2, b'h', b'i']);

        let pongs = feed(&mut WsFrameDecoder::new(), Form::Bytes, &wire, 64, 64).pongs;
        assert_eq!(&pongs[..2], &[0x8A, MAX_CONTROL_PAYLOAD as u8]);
        assert_eq!(&pongs[2..2 + MAX_CONTROL_PAYLOAD], &payload[..]);
        assert_eq!(&pongs[2 + MAX_CONTROL_PAYLOAD..], &[0x8A, 2, b'h', b'i']);

        let pongs = feed(&mut WsFrameDecoder::new_client(), Form::Bytes, &wire, 64, 64).pongs;
        assert_eq!(pongs.len(), MAX_PONG_LEN + 8);
        assert_eq!(&pongs[..2], &[0x8A, 0x80 | MAX_CONTROL_PAYLOAD as u8]);
        let key = [pongs[2], pongs[3], pongs[4], pongs[5]];
        let echoed: Vec<u8> =
            pongs[6..MAX_PONG_LEN].iter().enumerate().map(|(i, b)| b ^ key[i % 4]).collect();
        assert_eq!(echoed, payload);
    }

    #[test]
    fn spare_selects_header_then_payload_regions() {
        let mut decoder = WsFrameDecoder::new();
        let mut target = [MaybeUninit::new(0u8); 8];
        assert_eq!(decoder.spare(&mut target).len(), 2);

        // A 16-bit extended length requires two more header bytes.
        let region = decoder.spare(&mut target);
        region[0].write(0x82);
        region[1].write(126);
        assert_eq!(unsafe { decoder.advance(2, &mut target) }, 0);
        assert_eq!(decoder.decode_next().unwrap(), WsDecodeResult::NeedMore);
        let region = decoder.spare(&mut target);
        assert_eq!(region.len(), 2);
        region[0].write(0);
        region[1].write(10);
        assert_eq!(unsafe { decoder.advance(2, &mut target) }, 0);

        // The payload region is the front of the target, limited to the
        // target and then to the frame's remaining payload.
        assert_eq!(decoder.decode_next().unwrap(), WsDecodeResult::NeedMore);
        assert_eq!(decoder.payload_remaining(), 10);
        let target_ptr = target.as_ptr();
        let region = decoder.spare(&mut target);
        assert_eq!(region.as_ptr(), target_ptr);
        assert_eq!(region.len(), 8);
        for slot in region.iter_mut() {
            slot.write(7);
        }
        assert_eq!(unsafe { decoder.advance(8, &mut target) }, 8);
        assert_eq!(decoder.payload_remaining(), 2);
        assert_eq!(decoder.spare(&mut target).len(), 2);

        // An empty target gives an empty region during a payload read.
        assert!(decoder.spare(&mut []).is_empty());
    }

    #[test]
    fn end_of_input_closes_the_decoder() {
        let mut decoder = WsFrameDecoder::new();
        decoder.close();
        assert_eq!(decoder.decode_next().unwrap(), WsDecodeResult::Closed);
        assert_eq!(decoder.decode_next_message().unwrap(), WsDecodeResult::Closed);
    }
}
