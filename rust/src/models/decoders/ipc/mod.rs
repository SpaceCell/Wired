// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow IPC Decoders
//!
//! Frame-level decoding, record batch parsing, and table stream decoding
//! for the Arrow IPC wire format (file and stream protocols).

/// FlatBuffer parser for record batches.
pub mod parser;

/// Table decoder with zero-copy record batch decoding.
pub mod table_decoder;

// ---------------------------------------------------------------------------
// IPC Frame Decoder
//
// Decodes Arrow IPC frames for both file and stream protocols.
// File protocol: is bounded, random-access, with header magic and footer.
// Stream protocol: is unbounded, with continuation markers before messages.
//
// Two zero-copy methods cover the two streaming models:
// - decode (FrameDecoder): for contiguous payloads where all bytes are in one buffer
// - decode_header: for streaming reads where the body may not yet be buffered
// ---------------------------------------------------------------------------

use std::io;
use std::marker::PhantomData;

use self::parser::{decode_record_batch, handle_dictionary_batch, handle_schema_header};
use crate::arrow::message::org::apache::arrow::flatbuf as fb;
use crate::constants::{
    ARROW_MAGIC_NUMBER, ARROW_MAGIC_NUMBER_PADDED, CONTINUATION_SENTINEL, FILE_CLOSING_MAGIC_LEN,
    FILE_OPENING_MAGIC_LEN, METADATA_SIZE_PREFIX,
};
use crate::enums::{DecodeResult, IPCMessageProtocol};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::frames::ipc_message::{ArrowIPCFrameRanges, IPCFrameHeader, IPCFrameResult};
use crate::traits::frame_decoder::FrameDecoder;
use crate::traits::stream_buffer::StreamBuffer;
use crate::utils::{align_8, align_to};
use minarrow::Field;
use minarrow::structs::shared_buffer::SharedBuffer;
use std::collections::HashMap;

/// Zero-copy decoder for Arrow IPC frame boundaries.
///
/// Implements [`FrameDecoder`] with `Frame = ArrowIPCFrameRanges`, returning
/// byte ranges into the caller's buffer rather than copying data.
///
/// Also provides [`decode_header`](Self::decode_header) for the streaming
/// case where the body may not yet be buffered.
pub struct ArrowIPCFrameDecoder<B: StreamBuffer> {
    format: IPCMessageProtocol,
    /// True until we have accounted for the initial 8-byte file magic
    /// by including it in the `consumed` count of the first FILE frame.
    file_magic_unconsumed: bool,
    /// Resource caps applied during decode of untrusted input. Threaded into
    /// the underlying record-batch / dictionary / schema parsers when this
    /// decoder is paired with a payload-level driver.
    limits: DecodeLimits,
    _phantom: PhantomData<B>,
}

// ---------------------------------------------------------------------------
// FrameDecoder impl - zero-copy, returns byte ranges
// ---------------------------------------------------------------------------

impl<B: StreamBuffer> FrameDecoder for ArrowIPCFrameDecoder<B> {
    type Frame = ArrowIPCFrameRanges;

    /// Decode the next frame as byte ranges into the caller's buffer.
    ///
    /// Re-parses protocol headers from the start of the buffer on each call.
    /// This is cheap since it involves only pointer arithmetic and a single
    /// flatbuffer root read.
    ///
    /// After a successful frame, the caller must drain `consumed` bytes from
    /// the front of the buffer before calling again.
    fn decode(&mut self, buf: &[u8]) -> io::Result<DecodeResult<ArrowIPCFrameRanges>> {
        // Auto-detect File magic when configured for Stream mode.
        if self.format == IPCMessageProtocol::Stream && Self::has_opening_file_magic(buf) {
            self.format = IPCMessageProtocol::File;
            self.file_magic_unconsumed = true;
        }

        // Base offset accounts for the 8-byte file magic on the first frame.
        let base_off = if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
            if buf.len() < FILE_OPENING_MAGIC_LEN {
                return Ok(DecodeResult::NeedMore);
            }
            if !Self::has_opening_file_magic(buf) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Arrow file magic header",
                ));
            }
            FILE_OPENING_MAGIC_LEN
        } else {
            0
        };

        // Determine the prefix length and where the length field starts.
        let (prefix_len, len_off) = if self.format == IPCMessageProtocol::Stream {
            // Check for end-of-stream marker.
            if buf.len() >= base_off + 8 && Self::has_eos_marker(&buf[base_off..]) {
                return Ok(DecodeResult::Frame {
                    frame: ArrowIPCFrameRanges {
                        message_range: 0..0,
                        body_range: 0..0,
                    },
                    consumed: base_off + 8,
                });
            }
            let has_marker =
                buf.len() >= base_off + 4 && Self::has_continuation_sentinel(&buf[base_off..]);
            if has_marker {
                (8, base_off + 4)
            } else {
                (4, base_off)
            }
        } else {
            (METADATA_SIZE_PREFIX, base_off)
        };

        // Read the message length prefix.
        if buf.len() < len_off + METADATA_SIZE_PREFIX {
            return Ok(DecodeResult::NeedMore);
        }
        let msg_len = Self::read_u32_le(&buf[len_off..len_off + METADATA_SIZE_PREFIX]) as usize;

        // Zero-length message handling.
        if msg_len == 0 {
            if self.format == IPCMessageProtocol::Stream {
                return Ok(DecodeResult::Frame {
                    frame: ArrowIPCFrameRanges {
                        message_range: 0..0,
                        body_range: 0..0,
                    },
                    consumed: base_off + prefix_len,
                });
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Zero-length message",
            ));
        }

        // Footer detection for FILE protocol.
        if self.format == IPCMessageProtocol::File
            && Self::has_file_footer_markers(buf, len_off, msg_len)
        {
            let magic_start = len_off + METADATA_SIZE_PREFIX + msg_len;
            let magic_end = magic_start + FILE_OPENING_MAGIC_LEN;
            if &buf[magic_start..magic_end] == ARROW_MAGIC_NUMBER {
                // Footer reached, signal end with an empty frame.
                return Ok(DecodeResult::Frame {
                    frame: ArrowIPCFrameRanges {
                        message_range: 0..0,
                        body_range: 0..0,
                    },
                    consumed: magic_end + 4 + FILE_CLOSING_MAGIC_LEN,
                });
            }
        }

        // Ensure the full message is available.
        let meta_start = base_off + prefix_len;
        let meta_end = meta_start + msg_len;
        if buf.len() < meta_end {
            return Ok(DecodeResult::NeedMore);
        }

        // Parse the flatbuffer to extract the body length.
        use crate::AFMessage;
        let root = flatbuffers::root::<AFMessage>(&buf[meta_start..meta_end]).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Failed to parse message: {e}"),
            )
        })?;
        let body_len = root.bodyLength() as usize;

        // metadata_size on the wire already includes the encoder's metadata
        // padding, so no additional meta alignment is needed. The 8-byte
        // check matches the Arrow spec minimum.
        let meta_pad = align_8(msg_len);
        let body_start = meta_end + meta_pad;

        if body_len > 0 {
            let body_end = body_start + body_len;
            if buf.len() < body_end {
                return Ok(DecodeResult::NeedMore);
            }

            // Align frame boundary to B::ALIGN, matching the encoder's padding.
            let consumed_before_body_pad = base_off + prefix_len + msg_len + meta_pad + body_len;
            let body_pad = align_to::<B>(consumed_before_body_pad);
            let consumed = consumed_before_body_pad + body_pad;

            if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
                self.file_magic_unconsumed = false;
            }

            Ok(DecodeResult::Frame {
                frame: ArrowIPCFrameRanges {
                    message_range: meta_start..meta_end,
                    body_range: body_start..body_end,
                },
                consumed,
            })
        } else {
            let consumed = base_off + prefix_len + msg_len + meta_pad;

            if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
                self.file_magic_unconsumed = false;
            }

            Ok(DecodeResult::Frame {
                frame: ArrowIPCFrameRanges {
                    message_range: meta_start..meta_end,
                    body_range: 0..0,
                },
                consumed,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Construction and protocol helpers
// ---------------------------------------------------------------------------

impl<B: StreamBuffer> ArrowIPCFrameDecoder<B> {
    /// Construct a frame decoder. Pass `None` for the default resource caps
    /// applied to the record-batch / dictionary / schema decoders this walker
    /// drives, or `Some(...)` to override them.
    pub fn new(format: IPCMessageProtocol, limits: Option<DecodeLimits>) -> Self {
        Self {
            format,
            file_magic_unconsumed: matches!(format, IPCMessageProtocol::File),
            limits: limits.unwrap_or_default(),
            _phantom: PhantomData,
        }
    }

    /// Return the resource limits in effect for this decoder.
    pub fn limits(&self) -> DecodeLimits {
        self.limits
    }

    #[inline]
    fn read_u32_le(buf: &[u8]) -> u32 {
        u32::from_le_bytes(buf[..4].try_into().unwrap())
    }

    #[inline]
    fn has_opening_file_magic(buf: &[u8]) -> bool {
        buf.len() >= FILE_OPENING_MAGIC_LEN
            && &buf[..FILE_OPENING_MAGIC_LEN] == ARROW_MAGIC_NUMBER_PADDED
    }

    #[inline]
    fn has_continuation_sentinel(buf: &[u8]) -> bool {
        buf.len() >= METADATA_SIZE_PREFIX && Self::read_u32_le(buf) == CONTINUATION_SENTINEL
    }

    #[inline]
    fn has_eos_marker(buf: &[u8]) -> bool {
        buf.len() >= 8
            && Self::read_u32_le(&buf[0..4]) == 0xFFFF_FFFF
            && Self::read_u32_le(&buf[4..8]) == 0x0000_0000
    }

    #[inline]
    fn has_file_footer_markers(buf: &[u8], len_off: usize, msg_len: usize) -> bool {
        // After the (u32) length and `msg_len` bytes, the trailing magic must fit.
        msg_len > 0
            && len_off + METADATA_SIZE_PREFIX + msg_len + FILE_OPENING_MAGIC_LEN <= buf.len()
    }
}

// ---------------------------------------------------------------------------
// Streaming header decode for the two-step body read pattern
// ---------------------------------------------------------------------------

impl<B: StreamBuffer> ArrowIPCFrameDecoder<B> {
    /// Parse a frame header and metadata, returning body information
    /// without requiring the body to be in the buffer.
    ///
    /// For frames with no body, or small frames where the body IS in the
    /// buffer, returns `Complete`. For frames where the body is not yet
    /// buffered, returns `BodyPending` with the body length so the caller
    /// can read it directly into a dedicated buffer.
    pub fn decode_header(&mut self, buf: &[u8]) -> io::Result<IPCFrameHeader> {
        // Auto-detect File magic
        if self.format == IPCMessageProtocol::Stream && Self::has_opening_file_magic(buf) {
            self.format = IPCMessageProtocol::File;
            self.file_magic_unconsumed = true;
        }

        let base_off = if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
            if buf.len() < FILE_OPENING_MAGIC_LEN {
                return Ok(IPCFrameHeader::NeedMore);
            }
            if !Self::has_opening_file_magic(buf) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Arrow file magic header",
                ));
            }
            FILE_OPENING_MAGIC_LEN
        } else {
            0
        };

        let (prefix_len, len_off) = if self.format == IPCMessageProtocol::Stream {
            if buf.len() >= base_off + 8 && Self::has_eos_marker(&buf[base_off..]) {
                return Ok(IPCFrameHeader::EndOfStream {
                    consumed: base_off + 8,
                });
            }
            let has_marker =
                buf.len() >= base_off + 4 && Self::has_continuation_sentinel(&buf[base_off..]);
            if has_marker {
                (8, base_off + 4)
            } else {
                (4, base_off)
            }
        } else {
            (METADATA_SIZE_PREFIX, base_off)
        };

        if buf.len() < len_off + METADATA_SIZE_PREFIX {
            return Ok(IPCFrameHeader::NeedMore);
        }
        let msg_len = Self::read_u32_le(&buf[len_off..len_off + METADATA_SIZE_PREFIX]) as usize;

        if msg_len == 0 {
            if self.format == IPCMessageProtocol::Stream {
                return Ok(IPCFrameHeader::EndOfStream {
                    consumed: base_off + prefix_len,
                });
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Zero-length message",
            ));
        }

        // Footer detection for FILE protocol
        if self.format == IPCMessageProtocol::File
            && Self::has_file_footer_markers(buf, len_off, msg_len)
        {
            let magic_start = len_off + METADATA_SIZE_PREFIX + msg_len;
            let magic_end = magic_start + FILE_OPENING_MAGIC_LEN;
            if buf.len() >= magic_end && &buf[magic_start..magic_end] == ARROW_MAGIC_NUMBER {
                return Ok(IPCFrameHeader::EndOfStream {
                    consumed: magic_end + 4 + FILE_CLOSING_MAGIC_LEN,
                });
            }
        }

        let meta_start = base_off + prefix_len;
        let meta_end = meta_start + msg_len;
        if buf.len() < meta_end {
            return Ok(IPCFrameHeader::NeedMore);
        }

        // Parse metadata to get the body length
        use crate::AFMessage;
        let root = flatbuffers::root::<AFMessage>(&buf[meta_start..meta_end]).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Failed to parse message: {e}"),
            )
        })?;
        let body_len = root.bodyLength() as usize;
        let meta_pad = align_8(msg_len);

        if body_len == 0 {
            // No body - return as complete
            let consumed = base_off + prefix_len + msg_len + meta_pad;
            if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
                self.file_magic_unconsumed = false;
            }
            return Ok(IPCFrameHeader::Complete {
                frame: ArrowIPCFrameRanges {
                    message_range: meta_start..meta_end,
                    body_range: 0..0,
                },
                consumed,
            });
        }

        let body_start = meta_end + meta_pad;
        let body_end = body_start + body_len;

        if buf.len() >= body_end {
            // Body is already in the buffer - return as complete
            let consumed_before_body_pad = base_off + prefix_len + msg_len + meta_pad + body_len;
            let body_pad = align_to::<B>(consumed_before_body_pad);
            let consumed = consumed_before_body_pad + body_pad;

            if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
                self.file_magic_unconsumed = false;
            }

            return Ok(IPCFrameHeader::Complete {
                frame: ArrowIPCFrameRanges {
                    message_range: meta_start..meta_end,
                    body_range: body_start..body_end,
                },
                consumed,
            });
        }

        // Body not in buffer yet - return metadata with body info
        let header_consumed = body_start; // consume everything up to where the body starts
        let consumed_before_body_pad = base_off + prefix_len + msg_len + meta_pad + body_len;
        let body_pad = align_to::<B>(consumed_before_body_pad);

        if self.file_magic_unconsumed && self.format == IPCMessageProtocol::File {
            self.file_magic_unconsumed = false;
        }

        Ok(IPCFrameHeader::BodyPending {
            message_range: meta_start..meta_end,
            header_consumed,
            body_len,
            body_pad,
        })
    }
}

impl<B: StreamBuffer> Default for ArrowIPCFrameDecoder<B> {
    fn default() -> Self {
        ArrowIPCFrameDecoder::new(IPCMessageProtocol::Stream, None)
    }
}

// ---------------------------------------------------------------------------
// IPC Frame Decode Functions
//
// These are the public decode entry points used by ArrowIpcCodec.
// - decode_ipc_frame: single frame dispatch (schema/dict/batch/EOS)
// - decode_ipc_payload: contiguous payload containing all IPC frames
// ---------------------------------------------------------------------------

/// Decode a single IPC frame from message bytes and body.
///
/// Handles all message types: schema, dictionary, record batch, and EOS.
/// Column data is mapped as zero-copy SharedBuffer views for record batches.
///
/// The `fields` and `dicts` parameters accumulate state across frames.
/// `shared_cache` should be passed and returned for SharedBuffer recycling.
pub(crate) fn decode_ipc_frame(
    message: &[u8],
    body: SharedBuffer,
    body_len: usize,
    fields: &mut Vec<Field>,
    dicts: &mut HashMap<i64, Vec<String>>,
    shared_cache: &mut Option<SharedBuffer>,
    limits: DecodeLimits,
) -> io::Result<IPCFrameResult> {
    if message.is_empty() && body_len == 0 {
        return Ok(IPCFrameResult::EndOfStream);
    }

    let af_msg = flatbuffers::root::<fb::Message>(message)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    match af_msg.header_type() {
        fb::MessageHeader::Schema => {
            *fields = handle_schema_header(&af_msg, limits)?;
            Ok(IPCFrameResult::Schema)
        }
        fb::MessageHeader::DictionaryBatch => {
            let db = af_msg.header_as_dictionary_batch().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing DictionaryBatch header")
            })?;
            handle_dictionary_batch(&db, body.as_slice(), dicts, limits)?;
            Ok(IPCFrameResult::Dictionary)
        }
        fb::MessageHeader::RecordBatch => {
            if fields.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "received record batch before schema",
                ));
            }
            let rec = af_msg.header_as_record_batch().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing RecordBatch header")
            })?;
            *shared_cache = None;
            let (table, shared) =
                decode_record_batch(&rec, fields, dicts, body, 0, body_len, None, limits)?;
            *shared_cache = Some(shared);
            Ok(IPCFrameResult::Batch(table))
        }
        fb::MessageHeader::NONE => Ok(IPCFrameResult::EndOfStream),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected IPC message type: {:?}", other),
        )),
    }
}

/// Decode a contiguous IPC payload containing schema + dicts + record batch.
///
/// Used by the Lightstream protocol where the TLV frame contains the
/// entire IPC payload in one buffer. The alignment type `B` must match
/// what was used during encoding for correct frame boundary parsing.
///
/// Drops `cached` before decode to release the previous batch's memory.
pub(crate) fn decode_ipc_payload<B: StreamBuffer>(
    payload: SharedBuffer,
    fields: &mut Vec<Field>,
    dicts: &mut HashMap<i64, Vec<String>>,
    cached: Option<SharedBuffer>,
    limits: DecodeLimits,
) -> io::Result<(minarrow::Table, SharedBuffer)> {
    if let Some(prev) = cached {
        drop(prev);
    }

    let payload_ref = payload.as_slice();
    let mut decoder = ArrowIPCFrameDecoder::<B>::new(IPCMessageProtocol::Stream, Some(limits));
    let mut offset = 0;
    let mut record_batch_body: Option<(usize, usize)> = None;
    let mut record_batch_msg_range: Option<std::ops::Range<usize>> = None;

    while offset < payload_ref.len() {
        match decoder.decode(&payload_ref[offset..])? {
            DecodeResult::Frame { frame, consumed } => {
                if frame.message_range.is_empty() && frame.body_range.is_empty() {
                    break;
                }

                let msg_bytes = &payload_ref
                    [offset + frame.message_range.start..offset + frame.message_range.end];
                let af_msg = flatbuffers::root::<fb::Message>(msg_bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                match af_msg.header_type() {
                    fb::MessageHeader::Schema => {
                        *fields = handle_schema_header(&af_msg, limits)?;
                    }
                    fb::MessageHeader::DictionaryBatch => {
                        let db = af_msg.header_as_dictionary_batch().ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "missing DictionaryBatch header",
                            )
                        })?;
                        let body_bytes = &payload_ref
                            [offset + frame.body_range.start..offset + frame.body_range.end];
                        handle_dictionary_batch(&db, body_bytes, dicts, limits)?;
                    }
                    fb::MessageHeader::RecordBatch => {
                        if fields.is_empty() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "received record batch before schema",
                            ));
                        }
                        record_batch_body = Some((
                            offset + frame.body_range.start,
                            frame.body_range.end - frame.body_range.start,
                        ));
                        record_batch_msg_range = Some(
                            offset + frame.message_range.start..offset + frame.message_range.end,
                        );
                    }
                    fb::MessageHeader::NONE => break,
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unexpected IPC message type: {:?}", other),
                        ));
                    }
                }
                offset += consumed;
            }
            DecodeResult::NeedMore => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated IPC frame within table payload",
                ));
            }
        }
    }

    let (body_start, body_len) = record_batch_body.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "table payload did not contain a record batch",
        )
    })?;
    let msg_range = record_batch_msg_range.unwrap();

    let msg_ref = payload.clone();
    let msg_bytes = &msg_ref.as_slice()[msg_range.start..msg_range.end];
    let af_msg = flatbuffers::root::<fb::Message>(msg_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let rec = af_msg
        .header_as_record_batch()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing RecordBatch header"))?;

    decode_record_batch(
        &rec, fields, dicts, payload, body_start, body_len, None, limits,
    )
}
