// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! WebSocket binary frame header parsing and payload unmasking.
//!
//! Here we operate directly on byte slices and write into the one pre-allocated
//! byte buffer. The caller manages all buffers - these functions interpret
//! header bytes and unmask payloads in place.

/// Maximum WebSocket frame header size: 2 (base) + 8 (extended length) + 4 (mask key).
pub const MAX_HEADER_LEN: usize = 14;

/// WebSocket opcodes we handle.
pub const OPCODE_CONTINUATION: u8 = 0x0;
pub const OPCODE_TEXT: u8 = 0x1;
pub const OPCODE_BINARY: u8 = 0x2;
pub const OPCODE_CLOSE: u8 = 0x8;
pub const OPCODE_PING: u8 = 0x9;
pub const OPCODE_PONG: u8 = 0xA;

/// Parsed WebSocket frame header. References the header bytes, owns nothing.
#[derive(Debug, Clone, Copy)]
pub struct WsHeader {
    /// True if this is the final fragment.
    pub fin: bool,
    /// Frame opcode (0x0=continuation, 0x2=binary, 0x8=close, etc.).
    pub opcode: u8,
    /// True if the payload is masked (client to server).
    pub masked: bool,
    /// Payload length in bytes.
    pub payload_len: u64,
    /// Masking key, only valid if `masked` is true.
    pub mask_key: [u8; 4],
    /// Total header bytes consumed from the input.
    pub header_len: usize,
}

/// Try to parse a WebSocket frame header from the given bytes.
///
/// Returns `None` if not enough bytes are available yet. The caller
/// should accumulate more bytes and retry.
pub fn parse_header(buf: &[u8]) -> Option<WsHeader> {
    if buf.len() < 2 {
        return None;
    }

    let fin = buf[0] & 0x80 != 0;
    let opcode = buf[0] & 0x0F;
    let masked = buf[1] & 0x80 != 0;
    let len7 = (buf[1] & 0x7F) as u64;

    let (payload_len, mut pos) = if len7 <= 125 {
        (len7, 2)
    } else if len7 == 126 {
        if buf.len() < 4 {
            return None;
        }
        let len = u16::from_be_bytes([buf[2], buf[3]]) as u64;
        (len, 4)
    } else {
        // len7 == 127
        if buf.len() < 10 {
            return None;
        }
        let len = u64::from_be_bytes(buf[2..10].try_into().unwrap());
        (len, 10)
    };

    let mut mask_key = [0u8; 4];
    if masked {
        if buf.len() < pos + 4 {
            return None;
        }
        mask_key.copy_from_slice(&buf[pos..pos + 4]);
        pos += 4;
    }

    Some(WsHeader {
        fin,
        opcode,
        masked,
        payload_len,
        mask_key,
        header_len: pos,
    })
}

/// Write a WebSocket binary frame header into `buf`.
///
/// Returns the number of bytes written. The buffer must be at least
/// [`MAX_HEADER_LEN`] bytes. No masking (server to client).
pub fn write_binary_header(buf: &mut [u8], payload_len: usize) -> usize {
    debug_assert!(buf.len() >= MAX_HEADER_LEN);

    // FIN=1, opcode=binary(0x2)
    buf[0] = 0x82;

    if payload_len <= 125 {
        buf[1] = payload_len as u8;
        2
    } else if payload_len <= 65535 {
        buf[1] = 126;
        buf[2..4].copy_from_slice(&(payload_len as u16).to_be_bytes());
        4
    } else {
        buf[1] = 127;
        buf[2..10].copy_from_slice(&(payload_len as u64).to_be_bytes());
        10
    }
}

/// Write a masked WebSocket binary frame header into `buf`.
///
/// Returns the number of bytes written. The buffer must be at least
/// [`MAX_HEADER_LEN`] bytes. RFC 6455 requires every client-to-server
/// frame to be masked, with the payload XORed against `mask_key`.
pub fn write_masked_binary_header(buf: &mut [u8], payload_len: usize, mask_key: [u8; 4]) -> usize {
    let n = write_binary_header(buf, payload_len);
    buf[1] |= 0x80;
    buf[n..n + 4].copy_from_slice(&mask_key);
    n + 4
}

/// Write a WebSocket close frame into `buf`.
///
/// Returns the number of bytes written. Status code 1000 (normal closure).
pub fn write_close_frame(buf: &mut [u8]) -> usize {
    // FIN=1, opcode=close(0x8), payload=2 bytes (status code)
    buf[0] = 0x88;
    buf[1] = 0x02; // payload length = 2
    buf[2..4].copy_from_slice(&1000u16.to_be_bytes()); // normal closure
    4
}

/// Write a masked WebSocket close frame into `buf`.
///
/// Returns the number of bytes written. RFC 6455 requires client-to-server
/// control frames to be masked. Status code 1000 (normal closure).
pub fn write_masked_close_frame(buf: &mut [u8], mask_key: [u8; 4]) -> usize {
    buf[0] = 0x88;
    buf[1] = 0x80 | 0x02;
    buf[2..6].copy_from_slice(&mask_key);
    let status = 1000u16.to_be_bytes();
    buf[6] = status[0] ^ mask_key[0];
    buf[7] = status[1] ^ mask_key[1];
    8
}

/// Write a masked WebSocket pong frame into `buf` with the given payload.
///
/// Returns the number of bytes written. RFC 6455 requires client-to-server
/// control frames to be masked, and the pong payload must echo the ping.
pub fn write_masked_pong_frame(buf: &mut [u8], ping_payload: &[u8], mask_key: [u8; 4]) -> usize {
    let len = ping_payload.len();
    debug_assert!(len <= 125); // control frames max 125 bytes payload
    buf[0] = 0x8A; // FIN=1, opcode=pong(0xA)
    buf[1] = 0x80 | len as u8;
    buf[2..6].copy_from_slice(&mask_key);
    for i in 0..len {
        buf[6 + i] = ping_payload[i] ^ mask_key[i % 4];
    }
    6 + len
}

/// Write a WebSocket pong frame into `buf` with the given payload.
///
/// Returns the number of bytes written. Pong payload must echo the ping.
pub fn write_pong_frame(buf: &mut [u8], ping_payload: &[u8]) -> usize {
    let len = ping_payload.len();
    debug_assert!(len <= 125); // control frames max 125 bytes payload
    buf[0] = 0x8A; // FIN=1, opcode=pong(0xA)
    buf[1] = len as u8;
    if len > 0 {
        buf[2..2 + len].copy_from_slice(ping_payload);
    }
    2 + len
}

/// Generates fresh 32-bit masking keys for client-to-server frames.
///
/// Masking prevents clients from constructing wire bytes that resemble HTTP
/// traffic and could poison intermediary proxy caches. It provides no
/// encryption.
pub(crate) fn fresh_mask_key() -> [u8; 4] {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let h = RandomState::new().build_hasher().finish();
    ((h as u32) ^ ((h >> 32) as u32)).to_le_bytes()
}

/// Unmask a WebSocket payload in place.
///
/// Applies the 4-byte XOR masking key cyclically. For unmasked frames
/// (server to client), this is a no-op - don't call it.
#[inline]
pub fn unmask(payload: &mut [u8], mask_key: [u8; 4]) {
    // Process 8 bytes at a time using a doubled mask key
    let mask_u32 = u32::from_ne_bytes(mask_key);
    let mask_u64 = (mask_u32 as u64) | ((mask_u32 as u64) << 32);

    let (prefix, aligned, suffix) = unsafe { payload.align_to_mut::<u64>() };

    // Handle unaligned prefix
    for (i, byte) in prefix.iter_mut().enumerate() {
        *byte ^= mask_key[i % 4];
    }

    // XOR 8 bytes at a time on the aligned region
    let offset = prefix.len() % 4;
    let aligned_mask = if offset == 0 {
        mask_u64
    } else {
        // Rotate the mask to account for prefix alignment offset
        let mut rotated = [0u8; 8];
        for (i, byte) in rotated.iter_mut().enumerate() {
            *byte = mask_key[(offset + i) % 4];
        }
        u64::from_ne_bytes(rotated)
    };
    for word in aligned.iter_mut() {
        *word ^= aligned_mask;
    }

    // Handle unaligned suffix
    let total_prefix_aligned = prefix.len() + aligned.len() * 8;
    for (i, byte) in suffix.iter_mut().enumerate() {
        *byte ^= mask_key[(total_prefix_aligned + i) % 4];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_short_payload() {
        // FIN=1, opcode=binary, no mask, payload=5
        let buf = [0x82, 0x05];
        let h = parse_header(&buf).unwrap();
        assert!(h.fin);
        assert_eq!(h.opcode, OPCODE_BINARY);
        assert!(!h.masked);
        assert_eq!(h.payload_len, 5);
        assert_eq!(h.header_len, 2);
    }

    #[test]
    fn test_parse_medium_payload() {
        // FIN=1, opcode=binary, no mask, payload=300
        let mut buf = [0u8; 4];
        buf[0] = 0x82;
        buf[1] = 126;
        buf[2..4].copy_from_slice(&300u16.to_be_bytes());
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.payload_len, 300);
        assert_eq!(h.header_len, 4);
    }

    #[test]
    fn test_parse_large_payload() {
        // FIN=1, opcode=binary, no mask, payload=70000
        let mut buf = [0u8; 10];
        buf[0] = 0x82;
        buf[1] = 127;
        buf[2..10].copy_from_slice(&70000u64.to_be_bytes());
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.payload_len, 70000);
        assert_eq!(h.header_len, 10);
    }

    #[test]
    fn test_parse_masked() {
        // FIN=1, opcode=binary, masked, payload=5, mask=[1,2,3,4]
        let buf = [0x82, 0x85, 1, 2, 3, 4];
        let h = parse_header(&buf).unwrap();
        assert!(h.masked);
        assert_eq!(h.payload_len, 5);
        assert_eq!(h.mask_key, [1, 2, 3, 4]);
        assert_eq!(h.header_len, 6);
    }

    #[test]
    fn test_parse_incomplete() {
        assert!(parse_header(&[0x82]).is_none());
        assert!(parse_header(&[0x82, 126, 0]).is_none()); // need 4 bytes
        assert!(parse_header(&[0x82, 0x85, 1, 2]).is_none()); // need mask key
    }

    #[test]
    fn test_write_binary_header_short() {
        let mut buf = [0u8; MAX_HEADER_LEN];
        let n = write_binary_header(&mut buf, 50);
        assert_eq!(n, 2);
        assert_eq!(buf[0], 0x82);
        assert_eq!(buf[1], 50);
    }

    #[test]
    fn test_write_binary_header_medium() {
        let mut buf = [0u8; MAX_HEADER_LEN];
        let n = write_binary_header(&mut buf, 300);
        assert_eq!(n, 4);
        assert_eq!(buf[1], 126);
        assert_eq!(u16::from_be_bytes([buf[2], buf[3]]), 300);
    }

    #[test]
    fn test_write_binary_header_large() {
        let mut buf = [0u8; MAX_HEADER_LEN];
        let n = write_binary_header(&mut buf, 70000);
        assert_eq!(n, 10);
        assert_eq!(buf[1], 127);
        assert_eq!(u64::from_be_bytes(buf[2..10].try_into().unwrap()), 70000);
    }

    #[test]
    fn test_unmask_roundtrip() {
        let mask_key = [0xAB, 0xCD, 0xEF, 0x01];
        let original = b"Hello, WebSocket!".to_vec();
        let mut data = original.clone();

        unmask(&mut data, mask_key);
        assert_ne!(data, original); // should be different after masking

        unmask(&mut data, mask_key); // XOR is its own inverse
        assert_eq!(data, original);
    }

    /// Unmasking matches a byte-by-byte XOR for every start alignment and
    /// length across the prefix, aligned and suffix regions.
    #[test]
    fn test_unmask_matches_bytewise_xor_at_every_alignment() {
        let mask_key = [0xAB, 0xCD, 0xEF, 0x01];
        let backing: Vec<u8> = (0..64u8).collect();
        for start in 0..8 {
            for len in 0..40 {
                let mut data = backing[start..start + len].to_vec();
                // Shift the slice within a larger buffer to vary its
                // alignment independently of the allocation.
                let mut shifted = vec![0u8; start + len];
                shifted[start..].copy_from_slice(&data);
                unmask(&mut shifted[start..], mask_key);
                for (i, byte) in data.iter_mut().enumerate() {
                    *byte ^= mask_key[i % 4];
                }
                assert_eq!(&shifted[start..], &data[..], "start {start}, len {len}");
            }
        }
    }

    #[test]
    fn test_unmask_empty() {
        let mask_key = [1, 2, 3, 4];
        let mut data = vec![];
        unmask(&mut data, mask_key); // should not panic
    }

    #[test]
    fn test_close_frame() {
        let mut buf = [0u8; 16];
        let n = write_close_frame(&mut buf);
        assert_eq!(n, 4);
        assert_eq!(buf[0], 0x88); // FIN + close
        assert_eq!(buf[1], 2); // payload len
        assert_eq!(u16::from_be_bytes([buf[2], buf[3]]), 1000);
    }
}
