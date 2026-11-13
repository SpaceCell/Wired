// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebSocket frame encoder
//!
//! Write-side WebSocket protocol handling, shared by the async and blocking
//! WebSocket writers.
//!
//! ## Context
//!
//! RFC 6455 requires every client-to-server frame to be masked, and every
//! server-to-client frame to be unmasked. [`WsFrameEncoder`] holds the
//! connection's role and the frame being written: its header, its payload
//! source and the progress through both. Payloads from the connecting peer
//! are masked into a buffer reused across frames. Payloads from the
//! accepting peer are written from the caller's slice without a copy.
//!
//! Encoding is synchronous and doesn't perform any IO. Writers own the
//! transport. They begin a frame, write the bytes [`WsFrameEncoder::pending`]
//! returns, and commit each written count until the frame is complete.
//! [`WsWrite`](crate::models::streams::websocket::WsWrite) does this over an
//! `AsyncWrite` destination, and
//! [`SyncWsWrite`](crate::models::streams::sync_websocket::SyncWsWrite) over
//! a blocking `std::io::Write` destination.

use crate::models::frames::websocket::{self, MAX_HEADER_LEN, fresh_mask_key};

/// Synchronous WebSocket frame encoder that doesn't own a transport.
///
/// Frames each payload as one binary frame, and writes the normal-closure
/// close frame. The connecting peer masks every frame with a fresh key.
///
/// ## Behaviour
///
/// - [`Self::pending`] returns the unwritten header bytes first, then the
///   unwritten payload bytes.
/// - The frame is complete once every header and payload byte has been
///   committed through [`Self::advance`]. New encoders report complete.
/// - Beginning a frame replaces the frame in progress.
pub struct WsFrameEncoder {
    /// Client role: frames are masked per RFC 6455.
    client: bool,
    /// Masked copy of the current frame's payload for client writes.
    masked_payload: Vec<u8>,
    header: [u8; MAX_HEADER_LEN],
    header_len: usize,
    header_written: usize,
    payload_len: usize,
    payload_written: usize,
}

impl WsFrameEncoder {
    /// Construct an encoder for the accepting peer of a connection, which
    /// writes unmasked frames.
    pub fn new() -> Self {
        Self {
            client: false,
            masked_payload: Vec::new(),
            header: [0u8; MAX_HEADER_LEN],
            header_len: 0,
            header_written: 0,
            payload_len: 0,
            payload_written: 0,
        }
    }

    /// Construct an encoder for the connecting peer of a connection, which
    /// masks every frame.
    pub fn new_client() -> Self {
        Self {
            client: true,
            ..Self::new()
        }
    }

    /// Begin a binary frame for `payload`.
    ///
    /// For the connecting peer, the payload is masked into the encoder's
    /// buffer with the key written into the header. Pass the same `payload`
    /// to every [`Self::pending`] call for this frame.
    pub fn begin_binary(&mut self, payload: &[u8]) {
        self.header_len = if self.client {
            let key = fresh_mask_key();
            self.masked_payload.clear();
            self.masked_payload
                .extend(payload.iter().enumerate().map(|(i, b)| b ^ key[i % 4]));
            websocket::write_masked_binary_header(&mut self.header, payload.len(), key)
        } else {
            websocket::write_binary_header(&mut self.header, payload.len())
        };
        self.header_written = 0;
        self.payload_len = payload.len();
        self.payload_written = 0;
    }

    /// Begin a normal-closure close frame for the connection's role.
    ///
    /// The close frame doesn't have a payload argument. Pass an empty slice
    /// to [`Self::pending`] for this frame.
    pub fn begin_close(&mut self) {
        self.header_len = if self.client {
            websocket::write_masked_close_frame(&mut self.header, fresh_mask_key())
        } else {
            websocket::write_close_frame(&mut self.header)
        };
        self.header_written = 0;
        self.payload_len = 0;
        self.payload_written = 0;
    }

    /// Return the next bytes to write for the frame in progress.
    ///
    /// Returns the unwritten header bytes, then the unwritten payload bytes:
    /// a suffix of `payload` for the accepting peer, and of the masked copy
    /// for the connecting peer. Empty once the frame is complete.
    #[inline]
    pub fn pending<'a>(&'a self, payload: &'a [u8]) -> &'a [u8] {
        if self.header_written < self.header_len {
            return &self.header[self.header_written..self.header_len];
        }
        let source = if self.client {
            &self.masked_payload[..]
        } else {
            payload
        };
        &source[self.payload_written..self.payload_len.min(source.len())]
    }

    /// Commit `n` bytes of the slice returned by the preceding
    /// [`Self::pending`] call as written to the transport.
    #[inline]
    pub fn advance(&mut self, n: usize) {
        let header_left = self.header_len - self.header_written;
        if header_left > 0 {
            self.header_written += n.min(header_left);
        } else {
            self.payload_written += n;
        }
    }

    /// Report whether every byte of the frame in progress has been written.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.header_written >= self.header_len && self.payload_written >= self.payload_len
    }
}

impl Default for WsFrameEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write the frame in progress `step` bytes at a time and return the
    /// bytes written.
    fn drain(encoder: &mut WsFrameEncoder, payload: &[u8], step: usize) -> Vec<u8> {
        let mut wire = Vec::new();
        while !encoder.is_complete() {
            let pending = encoder.pending(payload);
            let n = pending.len().min(step);
            wire.extend_from_slice(&pending[..n]);
            encoder.advance(n);
        }
        assert!(encoder.pending(payload).is_empty());
        wire
    }

    #[test]
    fn new_encoders_report_complete() {
        assert!(WsFrameEncoder::new().is_complete());
        assert!(WsFrameEncoder::new_client().is_complete());
    }

    #[test]
    fn server_frames_survive_partial_header_and_payload_writes() {
        let payload = vec![0x5Au8; 300];
        for step in [1, 3, 1000] {
            let mut encoder = WsFrameEncoder::new();
            encoder.begin_binary(&payload);
            assert!(!encoder.is_complete());
            let wire = drain(&mut encoder, &payload, step);
            let mut expected = vec![0x82, 126];
            expected.extend_from_slice(&300u16.to_be_bytes());
            expected.extend_from_slice(&payload);
            assert_eq!(wire, expected);
        }
    }

    #[test]
    fn server_payload_is_written_from_the_callers_slice() {
        let payload = b"quote";
        let mut encoder = WsFrameEncoder::new();
        encoder.begin_binary(payload);
        let header = encoder.pending(payload).len();
        encoder.advance(header);
        assert_eq!(encoder.pending(payload).as_ptr(), payload.as_ptr());
    }

    #[test]
    fn client_frames_are_masked_across_partial_writes() {
        let payload: Vec<u8> = (0..200u8).collect();
        for step in [1, 7, 1000] {
            let mut encoder = WsFrameEncoder::new_client();
            encoder.begin_binary(&payload);
            let wire = drain(&mut encoder, &payload, step);
            assert_eq!(&wire[..2], &[0x82, 0x80 | 126]);
            assert_eq!(&wire[2..4], &200u16.to_be_bytes());
            let key = [wire[4], wire[5], wire[6], wire[7]];
            let unmasked: Vec<u8> =
                wire[8..].iter().enumerate().map(|(i, b)| b ^ key[i % 4]).collect();
            assert_eq!(unmasked, payload);
        }
    }

    #[test]
    fn close_frames_follow_the_role() {
        let mut server = WsFrameEncoder::new();
        server.begin_close();
        assert_eq!(drain(&mut server, &[], 1), [0x88, 2, 0x03, 0xE8]);

        let mut client = WsFrameEncoder::new_client();
        client.begin_close();
        let wire = drain(&mut client, &[], 3);
        assert_eq!(&wire[..2], &[0x88, 0x82]);
        let key = [wire[2], wire[3], wire[4], wire[5]];
        assert_eq!([wire[6] ^ key[0], wire[7] ^ key[1]], [0x03, 0xE8]);
    }

    #[test]
    fn beginning_a_frame_replaces_the_frame_in_progress() {
        let mut encoder = WsFrameEncoder::new();
        encoder.begin_binary(b"first");
        encoder.advance(1);
        encoder.begin_binary(b"ok");
        assert_eq!(drain(&mut encoder, b"ok", 64), [0x82, 2, b'o', b'k']);
    }
}
