// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow IPC Encoders
//!
//! Frame-level encoding, schema serialisation, and record batch writing
//! for the Arrow IPC wire format (file and stream protocols).

/// IPC schema serialisation.
pub mod schema;

/// Record batch encoding with zero intermediate allocations.
pub mod record_batch;

/// Encoder for Arrow tables.
pub mod table_encoder;

// ---------------------------------------------------------------------------
// IPC Frame Encoder
//
// Encodes Arrow IPC messages and bodies into file or stream frames,
// handling continuation markers, padding, alignment, and file footers.
// ---------------------------------------------------------------------------

use std::io;

use crate::constants::{
    ARROW_MAGIC_NUMBER, ARROW_MAGIC_NUMBER_PADDED, CONTINUATION_MARKER_LEN,
    DEFAULT_FRAME_ALLOCATION_SIZE, EOS_MARKER_LEN, METADATA_SIZE_PREFIX,
};
use crate::enums::IPCMessageProtocol;
use crate::models::frames::ipc_message::IPCFrameMetadata;
use crate::traits::frame_encoder::FrameEncoder;
use crate::traits::stream_buffer::StreamBuffer;
use crate::utils::align_to;

pub use crate::models::frames::ipc_message::IPCFrame;

/// Encodes a message + body as a valid Arrow IPC frame (file or stream).
///
/// See the [Arrow Columnar IPC Specification](https://arrow.apache.org/docs/format/Columnar.html#serialisation-and-interprocess-communication-ipc)
/// for the canonical layout.
pub struct IPCFrameEncoder;

impl FrameEncoder for IPCFrameEncoder {
    type Frame<'a> = IPCFrame<'a>;
    type Metadata = IPCFrameMetadata;

    /// Encode a single frame into the chosen [`StreamBuffer`], returning buffer and metadata.
    ///
    /// Behaviour:
    /// - **Stream**: always emits the 8-byte continuation marker (`0xFFFF_FFFF`, then metadata size).
    /// - **File**: emits opening magic on the first frame, and on the last frame emits EOS,
    ///   the file footer bytes + footer length (u32 LE), then the closing magic.
    ///
    /// Arrow framing layout:
    /// ```text
    /// <continuation: 0xFFFF_FFFF>
    /// <metadata_size: int32>
    /// <metadata_flatbuffer: bytes>
    /// <padding>
    /// <message body>
    /// ```
    fn encode<'a, B: StreamBuffer>(
        global_offset: &mut usize,
        frame: &Self::Frame<'a>,
    ) -> io::Result<(B, Self::Metadata)> {
        let mut out = B::with_capacity(DEFAULT_FRAME_ALLOCATION_SIZE);
        let mut ipc_frame_metadata = IPCFrameMetadata::default();

        if frame.protocol == IPCMessageProtocol::File && frame.is_first {
            out.extend_from_slice(ARROW_MAGIC_NUMBER_PADDED);
            ipc_frame_metadata.magic_len = ARROW_MAGIC_NUMBER_PADDED.len();
            *global_offset += ipc_frame_metadata.magic_len;
        }

        // Skip emitting an empty message+body unless it is the terminal EOS case handled below.
        let write_msg_frame = !frame.meta.is_empty() || !frame.body.is_empty();
        if write_msg_frame {
            Self::append_message_frame(
                global_offset,
                &mut out,
                frame.meta,
                frame.body,
                &mut ipc_frame_metadata,
            );
        };

        if frame.protocol == IPCMessageProtocol::File && frame.is_last {
            // The spec allows using the EOS marker in file mode prior to writing footer + magic.
            Self::append_eos_marker(global_offset, &mut out, &mut ipc_frame_metadata);
            Self::append_file_footer(
                global_offset,
                &mut out,
                frame
                    .footer_bytes
                    .expect("`is_last` must include footer bytes for IPCMessageProtocol::File."),
                &mut ipc_frame_metadata,
            );
            ipc_frame_metadata.footer_len = frame
                .footer_bytes
                .expect("Expected footer bytes for last message")
                .len();
        }
        if frame.protocol == IPCMessageProtocol::Stream && frame.is_last {
            Self::append_eos_marker(global_offset, &mut out, &mut ipc_frame_metadata);
        }
        Ok((out, ipc_frame_metadata))
    }
}

impl IPCFrameEncoder {
    /// Append the Arrow file footer + footer length (LE u32) + closing magic.
    ///
    /// Updates `global_offset` and records lengths in `ipc_frame_meta`.
    pub fn append_file_footer<B: StreamBuffer>(
        global_offset: &mut usize,
        out: &mut B,
        footer_bytes: &[u8],
        ipc_frame_meta: &mut IPCFrameMetadata,
    ) {
        out.extend_from_slice(footer_bytes);
        *global_offset += footer_bytes.len();

        out.extend_from_slice(&(footer_bytes.len() as u32).to_le_bytes());
        *global_offset += 4;

        ipc_frame_meta.footer_len = out.len();

        // Closing magic (unpadded).
        out.extend_from_slice(ARROW_MAGIC_NUMBER);
        ipc_frame_meta.magic_len = ARROW_MAGIC_NUMBER.len();
        *global_offset += ipc_frame_meta.magic_len;
    }

    /// Append the end-of-stream marker: `0xFFFF_FFFF` followed by `0x00000000`.
    fn append_eos_marker<B: StreamBuffer>(
        global_offset: &mut usize,
        out: &mut B,
        ipc_frame_meta: &mut IPCFrameMetadata,
    ) {
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // Continuation marker
        out.extend_from_slice(&0u32.to_le_bytes()); // Zero metadata length
        ipc_frame_meta.eos_len = EOS_MARKER_LEN;
        *global_offset += EOS_MARKER_LEN;
    }

    /// Append a single Arrow IPC message frame to `out`, including padding for alignment.
    fn append_message_frame<B: StreamBuffer>(
        global_offset: &mut usize,
        out: &mut B,
        meta: &[u8],
        body: &[u8],
        ipc_frame_meta: &mut IPCFrameMetadata,
    ) {
        // Header length - continuation (u32) + metadata_size (u32).
        ipc_frame_meta.header_len = CONTINUATION_MARKER_LEN + METADATA_SIZE_PREFIX;
        ipc_frame_meta.meta_len = meta.len();
        ipc_frame_meta.body_len = body.len();

        // Ensure capacity upfront to avoid repeated reallocations.
        out.reserve(ipc_frame_meta.frame_len());

        // Sentinel continuation marker
        let cont_marker = &0xFFFF_FFFFu32.to_le_bytes();
        out.extend_from_slice(cont_marker);
        *global_offset += 4;

        // Pad metadata as per the supplied buffer `ALIGN`.
        // Padding is computed against the running `global_offset`.
        // This is at minimum for compliance with the Arrow spec (`8-bytes`) or `64-bytes` for *SIMD*.
        ipc_frame_meta.meta_pad = align_to::<B>(*global_offset + 4 + meta.len());

        // Metadata size, including any padding
        let metadata_size =
            &(ipc_frame_meta.meta_len as u32 + ipc_frame_meta.meta_pad as u32).to_le_bytes();
        out.extend_from_slice(metadata_size);
        *global_offset += 4;

        // Message metadata.
        out.extend_from_slice(meta);
        *global_offset += meta.len();

        // Pad metadata, if required.
        if ipc_frame_meta.meta_pad != 0 {
            out.extend_from_slice(&vec![0u8; ipc_frame_meta.meta_pad]);
            *global_offset += ipc_frame_meta.meta_pad;
        }

        // Message body.
        out.extend_from_slice(body);
        *global_offset += body.len();

        // Body pad - usually zero due to per-buffer padding upstream, but here for completeness
        ipc_frame_meta.body_pad = align_to::<B>(*global_offset);
        if ipc_frame_meta.body_pad != 0 {
            out.extend_from_slice(&vec![0u8; ipc_frame_meta.body_pad]);
            *global_offset += ipc_frame_meta.body_pad;
        }
    }
}

#[cfg(test)]
mod tests {

    use minarrow::{Vec64, vec64};

    use super::*;
    use crate::enums::IPCMessageProtocol;

    #[test]
    fn test_ipc_frame_metadata_calculations() {
        let mut metadata = IPCFrameMetadata::default();
        metadata.header_len = 8;
        metadata.meta_len = 120;
        metadata.meta_pad = 8;
        metadata.body_len = 16;
        metadata.body_pad = 0;

        assert_eq!(metadata.metadata_total_len(), 128);
        assert_eq!(metadata.body_total_len(), 16);
        assert_eq!(metadata.frame_len(), 152);
    }

    #[test]
    fn test_empty_stream_frame() {
        let frame = IPCFrame {
            meta: &[],
            body: &[],
            protocol: IPCMessageProtocol::Stream,
            is_first: false,
            is_last: false,
            footer_bytes: None,
        };
        let (out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        assert_eq!(out.len(), 0);
        assert_eq!(metadata.frame_len(), 0);
    }

    #[test]
    fn test_stream_message_frame() {
        let meta_buf = vec![0u8; 120];
        let body_buf = vec![1u8; 16];
        let frame = IPCFrame {
            meta: &meta_buf,
            body: &body_buf,
            protocol: IPCMessageProtocol::Stream,
            is_first: false,
            is_last: false,
            footer_bytes: None,
        };
        let (out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();
        // Verify frame structure
        assert_eq!(&out.0[0..4], &0xFFFF_FFFFu32.to_le_bytes()); // Continuation marker

        // Metadata size field should include padding
        let meta_size = u32::from_le_bytes([out.0[4], out.0[5], out.0[6], out.0[7]]);
        assert_eq!(meta_size, (metadata.meta_len + metadata.meta_pad) as u32);

        // Verify metadata
        assert_eq!(metadata.header_len, 8);
        assert_eq!(metadata.meta_len, 120);
        // After header (8) + meta (120) = 128 bytes total
        // 128 % 64 = 0, so no padding needed for 64-byte alignment
        assert_eq!(metadata.meta_pad, 0); // No padding needed when already aligned
        assert_eq!(metadata.body_len, 16);
        // After header (8) + meta (120) + body (16) = 144 bytes
        // 144 % 64 = 16, so need 48 bytes padding to reach 192
        assert_eq!(metadata.body_pad, 48); // pads to next 64-byte boundary

        // Total frame size: header (8) + meta (120) + meta_pad (0) + body (16) + body_pad (48)
        assert_eq!(out.0.len(), 8 + 120 + 0 + 16 + 48); // = 192
        assert_eq!(out.0.len(), metadata.frame_len());
    }

    #[test]
    fn test_file_first_frame_with_magic() {
        let meta_buf = vec![0u8; 120];
        let body_buf = vec![1u8; 16];
        let frame = IPCFrame {
            meta: &meta_buf,
            body: &body_buf,
            protocol: IPCMessageProtocol::File,
            is_first: true,
            is_last: false,
            footer_bytes: None,
        };
        let (out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        // Should start with magic number
        assert_eq!(&out.0[0..8], ARROW_MAGIC_NUMBER_PADDED);
        assert_eq!(metadata.magic_len, 8);

        // Then continuation marker
        assert_eq!(&out.0[8..12], &0xFFFF_FFFFu32.to_le_bytes());

        // Verify total size includes first magic
        assert_eq!(metadata.magic_len, 8);
    }

    #[test]
    fn test_file_regular_frame() {
        let meta_buf = vec![0u8; 120];
        let body_buf = vec![1u8; 16];
        let frame = IPCFrame {
            meta: &meta_buf,
            body: &body_buf,
            protocol: IPCMessageProtocol::File,
            is_first: false,
            is_last: false,
            footer_bytes: None,
        };
        let (file_out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        // Should NOT have magic number
        assert_eq!(metadata.magic_len, 0);

        // Should start with continuation marker
        assert_eq!(&file_out.0[0..4], &0xFFFF_FFFFu32.to_le_bytes());

        // Same size as a regular stream frame
        let meta_buf = vec![0u8; 120];
        let body_buf = vec![1u8; 16];
        let frame = IPCFrame {
            meta: &meta_buf,
            body: &body_buf,
            protocol: IPCMessageProtocol::Stream,
            is_first: false,
            is_last: false,
            footer_bytes: None,
        };
        let (stream_out, _) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        assert_eq!(file_out.0.len(), stream_out.0.len());
    }

    #[test]
    fn test_stream_eos_marker() {
        let frame = IPCFrame {
            meta: &[],
            body: &[],
            protocol: IPCMessageProtocol::Stream,
            is_first: false,
            is_last: true,
            footer_bytes: None,
        };
        let (out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        // EOS marker is 8 bytes: 0xFFFFFFFF followed by 0x00000000
        assert_eq!(out.0.len(), 8);
        assert_eq!(&out.0[0..4], &0xFFFF_FFFFu32.to_le_bytes());
        assert_eq!(&out.0[4..8], &0u32.to_le_bytes());
        assert_eq!(metadata.eos_len, 8);
    }

    #[test]
    fn test_file_footer() {
        let footer = vec![2u8; 100];
        let frame = IPCFrame {
            meta: &[],
            body: &[],
            protocol: IPCMessageProtocol::File,
            is_first: false,
            is_last: true,
            footer_bytes: Some(&footer),
        };
        let (out, _) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        // Should have: EOS (8) + footer (100) + footer_len (4) + magic (6)
        let expected_len = 8 + 100 + 4 + 6;
        assert_eq!(out.0.len(), expected_len);

        // Verify EOS marker
        assert_eq!(&out.0[0..4], &0xFFFF_FFFFu32.to_le_bytes());
        assert_eq!(&out.0[4..8], &0u32.to_le_bytes());

        // Verify footer data
        assert_eq!(&out.0[8..108], &footer[..]);

        // Verify footer length
        let footer_len = u32::from_le_bytes([out.0[108], out.0[109], out.0[110], out.0[111]]);
        assert_eq!(footer_len, 100);

        // Verify final magic
        assert_eq!(&out.0[112..118], ARROW_MAGIC_NUMBER);
    }
    #[test]
    fn test_padding_calculations() {
        // Test various message sizes to ensure padding is correct
        let test_cases = vec![
            (1, 7),   // 1 byte -> 7 padding
            (7, 1),   // 7 bytes -> 1 padding
            (8, 0),   // 8 bytes -> 0 padding
            (9, 7),   // 9 bytes -> 7 padding
            (120, 0), // 120 bytes -> 0 padding (8 + 120 = 128)
            (121, 7), // 121 bytes -> 7 padding
        ];

        for (meta_size, expected_pad) in test_cases {
            let meta = vec![0u8; meta_size];
            let body = vec![1u8; 8]; // Use aligned body to isolate meta padding

            let frame = IPCFrame {
                meta: &meta,
                body: &body,
                protocol: IPCMessageProtocol::Stream,
                is_first: false,
                is_last: false,
                footer_bytes: None,
            };
            let (out, metadata) = IPCFrameEncoder::encode::<Vec<u8>>(&mut 0, &frame).unwrap();

            assert_eq!(
                metadata.meta_pad, expected_pad,
                "Failed for meta_size={}",
                meta_size
            );

            // Verify the actual padding bytes are zeros
            let pad_start = 8 + meta_size; // header + metadata
            let pad_end = pad_start + expected_pad;
            if expected_pad > 0 {
                assert!(out[pad_start..pad_end].iter().all(|&b| b == 0));
            }
        }
    }

    #[test]
    fn test_body_padding_vec() {
        let meta = vec![0u8; 8]; // Small aligned metadata

        let test_cases = vec![
            (1, 7),  // 1 byte -> 7 padding
            (8, 0),  // 8 bytes -> 0 padding
            (15, 1), // 15 bytes -> 1 padding
            (16, 0), // 16 bytes -> 0 padding
        ];

        for (body_size, expected_pad) in test_cases {
            let body = vec![1u8; body_size];

            let frame = IPCFrame {
                meta: &meta,
                body: &body,
                protocol: IPCMessageProtocol::Stream,
                is_first: false,
                is_last: false,
                footer_bytes: None,
            };
            let (_out, metadata) = IPCFrameEncoder::encode::<Vec<u8>>(&mut 0, &frame).unwrap();

            assert_eq!(
                metadata.body_pad, expected_pad,
                "Failed for body_size={}",
                body_size
            );
        }
    }

    #[test]
    fn test_body_padding_vec64() {
        let meta = vec64![0u8; 8];

        let test_cases = vec![
            (1, 63),  // 1 byte -> 63 padding
            (8, 56),  // 8 bytes -> 56 padding
            (15, 49), // 15 bytes -> 49 padding
            (16, 48), // 16 bytes -> 48 padding
        ];

        for (body_size, expected_pad) in test_cases {
            let body = vec![1u8; body_size];

            let frame = IPCFrame {
                meta: &meta,
                body: &body,
                protocol: IPCMessageProtocol::Stream,
                is_first: false,
                is_last: false,
                footer_bytes: None,
            };
            let (_out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

            assert_eq!(
                metadata.body_pad, expected_pad,
                "Failed for body_size={}",
                body_size
            );
        }
    }

    #[test]
    fn test_metadata_size_field() {
        // The metadata size field should contain
        // current_out_size + meta_entry_size + ipc_frame_meta.header_len + meta.len()
        // + padding to the 64 byte boundary
        let meta = vec![0u8; 120];
        let body = vec![1u8; 16];

        let frame = IPCFrame {
            meta: &meta,
            body: &body,
            protocol: IPCMessageProtocol::Stream,
            is_first: false,
            is_last: false,
            footer_bytes: None,
        };
        let (out, metadata) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();

        // Read the metadata size field
        let meta_size = u32::from_le_bytes([out.0[4], out.0[5], out.0[6], out.0[7]]);

        let expected_meta_pad = out.len()
            - metadata.body_total_len()
            - metadata.footer_eos_len()
            - metadata.magic_len
            - metadata.header_len;

        // Should equal metadata + padding (not including header)
        assert_eq!(meta_size, (metadata.meta_len + metadata.meta_pad) as u32);
        assert_eq!(meta_size, expected_meta_pad as u32);
    }

    #[test]
    #[should_panic(expected = "`is_last` must include footer bytes for IPCMessageProtocol::File.")]
    fn test_file_last_without_footer_panics() {
        let frame = IPCFrame {
            meta: &[],
            body: &[],
            protocol: IPCMessageProtocol::File,
            is_first: false,
            is_last: true,
            footer_bytes: None,
        };
        let _ = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame).unwrap();
    }

    #[test]
    fn test_full_file_sequence() {
        // Test a complete file write sequence

        // 1. First frame with schema
        let schema_meta = vec![0u8; 120];
        let frame1 = IPCFrame {
            meta: &schema_meta,
            body: &[],
            protocol: IPCMessageProtocol::File,
            is_first: true,
            is_last: false,
            footer_bytes: None,
        };
        let (_out1, meta1) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame1).unwrap();
        assert_eq!(meta1.magic_len, 8);

        // 2. Record batch frame
        let batch_meta = vec![0u8; 136];
        let batch_body = vec![1u8; 16];
        let frame2 = IPCFrame {
            meta: &batch_meta,
            body: &batch_body,
            protocol: IPCMessageProtocol::File,
            is_first: false,
            is_last: false,
            footer_bytes: None,
        };
        let (_out2, meta2) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame2).unwrap();
        assert_eq!(meta2.magic_len, 0);
        assert_eq!(meta2.body_len, 16);

        // 3. Footer frame
        let footer_data = vec![2u8; 88];
        let frame3 = IPCFrame {
            meta: &[],
            body: &[],
            protocol: IPCMessageProtocol::File,
            is_first: false,
            is_last: true,
            footer_bytes: Some(&footer_data),
        };
        let (_out3, meta3) = IPCFrameEncoder::encode::<Vec64<u8>>(&mut 0, &frame3).unwrap();
        assert_eq!(meta3.eos_len, 8);
        assert_eq!(meta3.footer_len, 88);
        assert!(_out3.0.ends_with(ARROW_MAGIC_NUMBER));
    }
}
