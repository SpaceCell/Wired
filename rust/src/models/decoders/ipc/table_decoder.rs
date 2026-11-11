// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow IPC Table Decoder
//!
//! Decodes Arrow IPC frames into tables with zero-copy record batch decoding.
//!
//! - Uses a two-step read:
//!     1. A small accumulation buffer for frame headers and metadata
//!     2. A StreamArena for record batch bodies read directly from the transport
//! - Column data is decoded via SharedBuffer mapping without userspace copies.
//! - Each batch is yielded individually as a SharedBuffer-backed Table.
//! - The arena is recycled when all previous batch SharedBuffer views are dropped.
//!
//! Decoding is synchronous and doesn't perform any IO. Readers own the
//! transport, read its bytes into the region the decoder exposes, and call
//! the decoder for the next table.
//! [`TableReader`](crate::models::readers::ipc::table::TableReader) does this
//! over an `AsyncRead` source, and
//! [`SyncTableReader`](crate::models::readers::ipc::sync_table::SyncTableReader)
//! over a blocking `std::io::Read` source.

use std::io;
use std::mem::MaybeUninit;

use minarrow::structs::shared_buffer::SharedBuffer;
use minarrow::*;

use crate::arrow::message::org::apache::arrow::flatbuf as fb;
use crate::enums::{BatchState, IPCMessageProtocol};
use crate::models::codecs::ipc::ArrowIpcCodec;
use crate::models::decoders::ipc::{ArrowIPCFrameDecoder, IPCFrameHeader};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::frames::ipc_message::IPCFrameResult;
use crate::models::streams::stream_arena::StreamArena;
use crate::traits::stream_buffer::StreamBuffer;

const DEFAULT_CHUNK: usize = 64 * 1024;

/// Largest read requested while discarding body padding.
const SKIP_PAD_CHUNK: usize = 1024;

/// An owned Arrow `custom_metadata` key/value pair decoded from a record
/// batch message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValue {
    /// Metadata key.
    pub key: String,
    /// Metadata value.
    pub value: String,
}

/// Two-step read state machine.
enum Phase {
    /// Accumulating frame headers and metadata.
    Metadata,
    /// Reading a record batch body into the arena from the transport.
    Body {
        meta_bytes: Vec<u8>,
        body_start: usize,
        body_filled: usize,
        body_len: usize,
        body_pad: usize,
        custom_metadata: Option<Vec<KeyValue>>,
    },
    /// Discarding body padding bytes after a body read.
    SkipPad { remaining: usize },
}

/// Outcome of one [`TableDecoder::decode_next`] call.
pub enum TableDecodeResult {
    /// A decoded record batch with the custom_metadata pairs of its message.
    Batch(Table, Option<Vec<KeyValue>>),
    /// More bytes are required before the next frame can be decoded.
    NeedMore,
    /// The stream has ended.
    EndOfStream,
}

/// Direct Arrow IPC decoder with zero-copy record batch decoding.
///
/// Yields each batch as an individual SharedBuffer-backed Table. The body
/// Vec64 is recycled across batches - after the caller drops the previous
/// Table, the SharedBuffer releases the Vec64 for reuse.
///
/// The decoder is synchronous and doesn't perform any IO. A reader calls
/// [`Self::decode_next`], and on [`TableDecodeResult::NeedMore`] it reads
/// from its transport into [`Self::spare_uninit`], commits the bytes through
/// [`Self::advance`], and calls [`Self::decode_next`] again. Record batch
/// bodies are read from the transport into the arena this way.
///
/// The `B` parameter controls frame alignment and must match what the
/// encoder used. Use `Vec64<u8>` for 64-byte SIMD alignment or `Vec<u8>`
/// for standard 8-byte Arrow alignment.
pub struct TableDecoder<B: StreamBuffer = Vec64<u8>> {
    decoder: ArrowIPCFrameDecoder<B>,
    /// Small accumulation buffer for frame prefixes and metadata.
    buf: Vec64<u8>,
    chunk_size: usize,
    eof: bool,
    state: BatchState,
    phase: Phase,
    codec: ArrowIpcCodec<B>,
    /// Arena for zero-allocation body reads.
    arena: StreamArena,
    /// Discard region for body padding reads.
    discard: [MaybeUninit<u8>; SKIP_PAD_CHUNK],
}

impl<B: StreamBuffer + Unpin> TableDecoder<B> {
    /// Construct a table decoder. Pass `None` for the default per-decode
    /// resource caps applied to every IPC frame.
    pub fn new(
        initial_capacity: usize,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self {
        Self {
            decoder: ArrowIPCFrameDecoder::new(protocol, limits),
            buf: Vec64::with_capacity(initial_capacity),
            chunk_size: DEFAULT_CHUNK,
            eof: false,
            state: BatchState::NeedSchema,
            phase: Phase::Metadata,
            codec: ArrowIpcCodec::new(Vec::new(), protocol, None, limits),
            arena: StreamArena::new(),
            discard: [MaybeUninit::uninit(); SKIP_PAD_CHUNK],
        }
    }

    /// Return the decoded schema, if available.
    pub fn schema(&self) -> Option<&[Field]> {
        let s = self.codec.schema();
        if s.is_empty() { None } else { Some(s) }
    }

    /// Drains the consumed parts of the buffer
    fn drain_consumed(&mut self, n: usize) {
        if n >= self.buf.len() {
            self.buf.clear();
        } else {
            let remaining = self.buf.len() - n;
            self.buf.copy_within(n.., 0);
            self.buf.truncate(remaining);
        }
    }

    /// Get the region for the next transport read.
    ///
    /// Returns the spare capacity of the metadata accumulation buffer, the
    /// unfilled part of the current record batch body in the arena, or the
    /// padding discard region, depending on what the decoder requires next.
    /// The caller writes into this region, then calls `advance(n)` to commit
    /// the `n` bytes it initialised. Asynchronous read paths wrap the slice
    /// with `ReadBuf::uninit`.
    pub fn spare_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        match &self.phase {
            Phase::Metadata => {
                // Ensures at least `chunk_size` spare capacity before reading.
                let current_len = self.buf.len();
                if self.buf.capacity() - current_len < self.chunk_size {
                    self.buf.reserve(self.chunk_size);
                }
                self.buf.spare_capacity_mut()
            }
            Phase::Body {
                body_filled,
                body_len,
                ..
            } => {
                let remaining = body_len - body_filled;
                let spare = self.arena.spare_uninit();
                let read_len = spare.len().min(remaining);
                &mut spare[..read_len]
            }
            Phase::SkipPad { remaining } => {
                let want = (*remaining).min(SKIP_PAD_CHUNK);
                &mut self.discard[..want]
            }
        }
    }

    /// Commit the first `n` bytes of the region returned by the preceding
    /// [`Self::spare_uninit`] call.
    ///
    /// # Safety
    ///
    /// Every byte in the first `n` elements of that region must have been
    /// initialised, and `n` must not exceed its length.
    pub unsafe fn advance(&mut self, n: usize) {
        match &mut self.phase {
            Phase::Metadata => {
                let current_len = self.buf.len();
                // SAFETY: the caller initialised `n` bytes within the spare
                // capacity returned by `spare_uninit`, and `current_len + n`
                // cannot exceed `self.buf.capacity()` because `n` is at most
                // the length of that region.
                unsafe { self.buf.set_len(current_len + n) };
            }
            Phase::Body { body_filled, .. } => {
                // SAFETY: the caller initialised the first `n` bytes of the
                // arena's spare region.
                unsafe { self.arena.advance(n) };
                *body_filled += n;
            }
            Phase::SkipPad { remaining } => *remaining = remaining.saturating_sub(n),
        }
    }

    /// Mark the input as ended.
    ///
    /// Readers call this when the transport reports the end of its data or
    /// a read error. [`Self::decode_next`] then returns the end of the
    /// stream, or an error when the stream stops inside a frame.
    pub fn end_input(&mut self) {
        self.eof = true;
    }

    /// Start a body read phase: copy overread bytes from the metadata
    /// accumulation buffer into the arena, then transition to
    /// Phase::Body for the remaining bytes.
    fn begin_body_read(
        &mut self,
        meta_bytes: Vec<u8>,
        body_len: usize,
        body_pad: usize,
        custom_metadata: Option<Vec<KeyValue>>,
    ) {
        // Rewind over previous bodies once their windows have dropped,
        // so steady-state streaming reuses one committed region.
        self.arena.recycle_if_free();
        // Ensure arena has room for this body, growing a generation when
        // the body exceeds the arena capacity.
        self.arena.ensure_capacity(body_len);

        let body_start = self.arena.write_pos();

        // Copy any overread bytes from the metadata buffer into the arena
        let overread = self.buf.len().min(body_len);
        if overread > 0 {
            self.arena
                .extend_from_slice(&self.buf[..overread])
                .expect("arena capacity checked above");
            self.drain_consumed(overread);
        }

        // Discard any body padding that was also overread
        let pad_overread = self.buf.len().min(body_pad);
        if pad_overread > 0 {
            self.drain_consumed(pad_overread);
        }

        self.phase = Phase::Body {
            meta_bytes,
            body_start,
            body_filled: overread,
            body_len,
            body_pad: body_pad - pad_overread,
            custom_metadata,
        };
    }

    /// Decode the next table from the bytes received so far.
    ///
    /// Returns the next record batch paired with the custom_metadata pairs
    /// from its message, the end of the stream, or
    /// [`TableDecodeResult::NeedMore`] when the next frame is incomplete.
    pub fn decode_next(&mut self) -> io::Result<TableDecodeResult> {
        let this = self;

        loop {
            match &mut this.phase {
                Phase::SkipPad { remaining } => {
                    if *remaining == 0 {
                        this.phase = Phase::Metadata;
                        continue;
                    }
                    if this.eof {
                        this.phase = Phase::Metadata;
                        continue;
                    }
                    return Ok(TableDecodeResult::NeedMore);
                }

                Phase::Body {
                    body_filled,
                    body_len,
                    ..
                } => {
                    if *body_filled < *body_len {
                        if this.eof {
                            this.phase = Phase::Metadata;
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "stream ended during record batch body",
                            ));
                        }
                        return Ok(TableDecodeResult::NeedMore);
                    }

                    let Phase::Body {
                        meta_bytes,
                        body_start,
                        body_len,
                        body_pad,
                        custom_metadata,
                        ..
                    } = std::mem::replace(&mut this.phase, Phase::Metadata)
                    else {
                        unreachable!()
                    };

                    // Body complete - create SharedBuffer window from the arena
                    let shared = this.arena.window(body_start, body_len);
                    this.arena.align();

                    let batch = match this.codec.decode_frame(&meta_bytes, shared, body_len)? {
                        IPCFrameResult::Batch(t) => t,
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "expected RecordBatch frame for body",
                            ));
                        }
                    };

                    if body_pad > 0 {
                        this.phase = Phase::SkipPad {
                            remaining: body_pad,
                        };
                    }

                    return Ok(TableDecodeResult::Batch(batch, custom_metadata));
                }

                Phase::Metadata => {
                    match this.decoder.decode_header(&this.buf)? {
                        IPCFrameHeader::EndOfStream { consumed } => {
                            this.state = BatchState::Done;
                            this.drain_consumed(consumed);
                            return Ok(TableDecodeResult::EndOfStream);
                        }

                        IPCFrameHeader::Complete { frame, consumed } => {
                            let msg_bytes = &this.buf[frame.message_range.clone()];
                            let body_len = frame.body_range.end - frame.body_range.start;

                            // Peek at the message type to decide how to handle it
                            let af_msg = flatbuffers::root::<fb::Message>(msg_bytes)
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                            match af_msg.header_type() {
                                fb::MessageHeader::Schema | fb::MessageHeader::DictionaryBatch => {
                                    // Schema and dict frames are small - decode
                                    // directly from the accumulation buffer
                                    let body_shared = SharedBuffer::from_vec(
                                        this.buf[frame.body_range.clone()].to_vec(),
                                    );
                                    let result = this.codec.decode_frame(
                                        msg_bytes,
                                        body_shared,
                                        body_len,
                                    )?;
                                    this.drain_consumed(consumed);
                                    if let IPCFrameResult::Schema = result {
                                        this.state = BatchState::Ready;
                                    }
                                    continue;
                                }
                                fb::MessageHeader::RecordBatch => {
                                    // Record batch - read body into a dedicated
                                    // Vec64 for zero-copy SharedBuffer decode.
                                    // Carry the message's first custom_metadata
                                    // pair coupled with the table. No key is
                                    // interpreted here.
                                    let custom_metadata = af_msg.custom_metadata().map(|kvs| {
                                        kvs.iter()
                                            .filter_map(|kv| match (kv.key(), kv.value()) {
                                                (Some(k), Some(v)) => Some(KeyValue {
                                                    key: k.to_string(),
                                                    value: v.to_string(),
                                                }),
                                                _ => None,
                                            })
                                            .collect()
                                    });
                                    let meta_saved = msg_bytes.to_vec();
                                    let body_pad = consumed - frame.body_range.end;
                                    let body_start = frame.body_range.start;
                                    this.drain_consumed(body_start);
                                    this.begin_body_read(
                                        meta_saved,
                                        body_len,
                                        body_pad,
                                        custom_metadata,
                                    );
                                    continue;
                                }
                                fb::MessageHeader::NONE => {
                                    this.state = BatchState::Done;
                                    this.drain_consumed(consumed);
                                    return Ok(TableDecodeResult::EndOfStream);
                                }
                                _ => {
                                    this.drain_consumed(consumed);
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "unexpected message order",
                                    ));
                                }
                            }
                        }

                        IPCFrameHeader::BodyPending {
                            message_range,
                            header_consumed,
                            body_len,
                            body_pad,
                        } => {
                            if !matches!(this.state, BatchState::Ready) {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "unexpected message order",
                                ));
                            }

                            let meta_bytes = this.buf[message_range].to_vec();
                            this.drain_consumed(header_consumed);

                            // Carry the message's first custom_metadata pair
                            // coupled with the table. No key is interpreted here.
                            let kvs = flatbuffers::root::<fb::Message>(&meta_bytes)
                                .ok()
                                .and_then(|m| m.custom_metadata());
                            let custom_metadata = kvs.map(|kvs| {
                                kvs.iter()
                                    .filter_map(|kv| match (kv.key(), kv.value()) {
                                        (Some(k), Some(v)) => Some(KeyValue {
                                            key: k.to_string(),
                                            value: v.to_string(),
                                        }),
                                        _ => None,
                                    })
                                    .collect()
                            });
                            this.begin_body_read(meta_bytes, body_len, body_pad, custom_metadata);
                            continue;
                        }

                        IPCFrameHeader::NeedMore => {
                            if this.eof {
                                if matches!(this.state, BatchState::Done) {
                                    return Ok(TableDecodeResult::EndOfStream);
                                }
                                return Err(io::Error::new(
                                    io::ErrorKind::UnexpectedEof,
                                    "stream ended before Arrow EOS marker",
                                ));
                            }
                            return Ok(TableDecodeResult::NeedMore);
                        }
                    }
                }
            }
        }
    }
}
