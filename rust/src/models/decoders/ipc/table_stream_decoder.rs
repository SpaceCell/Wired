// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow IPC Table Stream Decoder
//!
//! Reads Arrow IPC frames from an `AsyncRead` source with zero-copy record
//! batch decoding.
//!
//! - Uses a two-step read:
//!     1. A small accumulation buffer for frame headers and metadata
//!     2. A StreamArena for record batch bodies read directly from the transport
//! - Column data is decoded via SharedBuffer mapping without userspace copies.
//! - Each batch is yielded individually as a SharedBuffer-backed Table.
//! - The arena is recycled when all previous batch SharedBuffer views are dropped.
//!
//! The decode loop itself is independent of the transport. [`TableStreamDecoder`]
//! drives it over an `AsyncRead` source, and
//! [`SyncTableReader`](crate::models::readers::ipc::sync_table::SyncTableReader)
//! drives the same loop over a blocking `std::io::Read` source.

use std::io;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use minarrow::structs::shared_buffer::SharedBuffer;
use minarrow::*;
use tokio::io::{AsyncRead, ReadBuf};

use crate::arrow::message::org::apache::arrow::flatbuf as fb;
use crate::enums::{BatchState, IPCMessageProtocol};
use crate::models::codecs::ipc::ArrowIpcCodec;
use crate::models::decoders::ipc::{ArrowIPCFrameDecoder, IPCFrameHeader};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::frames::ipc_message::IPCFrameResult;
use crate::models::streams::stream_arena::StreamArena;
use crate::traits::stream_buffer::StreamBuffer;

const DEFAULT_CHUNK: usize = 64 * 1024;

/// Largest read issued while discarding body padding.
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

/// Byte source read by the IPC table decode loop.
///
/// The async [`TableStreamDecoder`] and the blocking `SyncTableReader` share
/// one decode loop, [`TableStreamCore`], and differ only in how bytes arrive.
/// Async sources return `Poll::Pending` when no bytes are ready. Blocking
/// sources wait for bytes and always return `Poll::Ready`.
///
/// # Safety
///
/// `Poll::Ready(Ok(n))` must satisfy `n <= buf.len()`, with the first `n`
/// bytes of `buf` initialised. The decode loop commits those bytes to its
/// buffers on that basis. `Ok(0)` signals end of stream.
pub(crate) unsafe trait IpcByteSource {
    /// Read into `buf` and return the number of bytes initialised.
    fn poll_read_uninit(&mut self, buf: &mut [MaybeUninit<u8>]) -> Poll<io::Result<usize>>;
}

/// [`IpcByteSource`] over an `AsyncRead` transport, polled from a task context.
struct AsyncByteSource<'a, 'b> {
    source: &'a mut (dyn AsyncRead + Unpin + Send),
    cx: &'a mut Context<'b>,
}

// SAFETY: `ReadBuf::uninit` tracks the bytes `poll_read` initialises, and
// `filled().len()` never exceeds the slice it wraps.
unsafe impl IpcByteSource for AsyncByteSource<'_, '_> {
    fn poll_read_uninit(&mut self, buf: &mut [MaybeUninit<u8>]) -> Poll<io::Result<usize>> {
        let mut read_buf = ReadBuf::uninit(buf);
        match Pin::new(&mut *self.source).poll_read(self.cx, &mut read_buf) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(read_buf.filled().len())),
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
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

/// Transport-independent state of the Arrow IPC table decode loop.
///
/// Holds the frame decoder, the metadata accumulation buffer, the body arena
/// and the codec. Each poll reads from the [`IpcByteSource`] passed to it,
/// which allows async and blocking readers to decode through the same code.
///
/// The `B` parameter controls frame alignment and must match what the
/// encoder used.
pub(crate) struct TableStreamCore<B: StreamBuffer> {
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
}

impl<B: StreamBuffer + Unpin> TableStreamCore<B> {
    /// Construct the decode state. Pass `None` for the default per-decode
    /// resource caps applied to every IPC frame.
    pub(crate) fn new(
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
        }
    }

    /// Return the decoded schema, if available.
    pub(crate) fn schema(&self) -> Option<&[Field]> {
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

    /// Poll the transport for more bytes into the metadata accumulation buffer.
    ///
    /// Ensures at least `chunk_size` spare capacity before reading, then
    /// appends whatever the source provides. Returns the number of bytes
    /// read, or sets `eof` if the source is exhausted.
    fn poll_fill_buf<S: IpcByteSource>(&mut self, source: &mut S) -> Poll<io::Result<usize>> {
        let current_len = self.buf.len();
        if self.buf.capacity() - current_len < self.chunk_size {
            self.buf.reserve(self.chunk_size);
        }

        let spare = self.buf.spare_capacity_mut();

        match source.poll_read_uninit(spare) {
            Poll::Ready(Ok(n)) => {
                if n == 0 {
                    self.eof = true;
                }
                // SAFETY: the `IpcByteSource` contract guarantees `n` bytes
                // were initialised within the spare slice above, and
                // `current_len + n` cannot exceed `self.buf.capacity()`
                // because `n` is at most `spare.len()`.
                unsafe { self.buf.set_len(current_len + n) };
                Poll::Ready(Ok(n))
            }
            Poll::Ready(Err(e)) => {
                self.eof = true;
                Poll::Ready(Err(e))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    /// Read bytes from the transport into the arena at the current write position.
    fn poll_read_body<S: IpcByteSource>(
        &mut self,
        source: &mut S,
        filled: &mut usize,
        target: usize,
    ) -> Poll<io::Result<bool>> {
        let remaining = target - *filled;
        if remaining == 0 {
            return Poll::Ready(Ok(true));
        }

        let n = {
            let spare = self.arena.spare_uninit();
            let read_len = spare.len().min(remaining);
            match source.poll_read_uninit(&mut spare[..read_len]) {
                Poll::Ready(Ok(n)) => n,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Pending => return Poll::Pending,
            }
        };

        if n == 0 {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "stream ended during record batch body",
            )));
        }

        // SAFETY: the `IpcByteSource` contract guarantees the first `n` bytes
        // of the spare region were initialised.
        unsafe { self.arena.advance(n) };
        *filled += n;
        Poll::Ready(Ok(*filled >= target))
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

    /// Poll for the next decoded table paired with the single custom_metadata
    /// pair from its record batch message.
    #[allow(clippy::type_complexity)]
    pub(crate) fn poll_next_keyed<S: IpcByteSource>(
        &mut self,
        source: &mut S,
    ) -> Poll<Option<io::Result<(Table, Option<Vec<KeyValue>>)>>> {
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
                    let want = (*remaining).min(SKIP_PAD_CHUNK);
                    let mut discard = [MaybeUninit::<u8>::uninit(); SKIP_PAD_CHUNK];
                    match source.poll_read_uninit(&mut discard[..want]) {
                        Poll::Ready(Ok(n)) => {
                            if n == 0 {
                                this.eof = true;
                            }
                            *remaining = remaining.saturating_sub(n);
                            continue;
                        }
                        Poll::Ready(Err(e)) => {
                            this.eof = true;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }

                Phase::Body { .. } => {
                    // Take the phase to avoid borrow conflicts with poll_read_body
                    let phase = std::mem::replace(&mut this.phase, Phase::Metadata);
                    let (
                        meta_bytes,
                        body_start,
                        mut body_filled,
                        body_len,
                        body_pad,
                        custom_metadata,
                    ) = match phase {
                        Phase::Body {
                            meta_bytes,
                            body_start,
                            body_filled,
                            body_len,
                            body_pad,
                            custom_metadata,
                        } => (
                            meta_bytes,
                            body_start,
                            body_filled,
                            body_len,
                            body_pad,
                            custom_metadata,
                        ),
                        _ => unreachable!(),
                    };

                    match this.poll_read_body(source, &mut body_filled, body_len) {
                        Poll::Ready(Ok(true)) => {
                            // Body complete - create SharedBuffer window from the arena
                            let shared = this.arena.window(body_start, body_len);
                            this.arena.align();

                            let batch =
                                match this.codec.decode_frame(&meta_bytes, shared, body_len)? {
                                    IPCFrameResult::Batch(t) => t,
                                    _ => {
                                        return Poll::Ready(Some(Err(io::Error::new(
                                            io::ErrorKind::InvalidData,
                                            "expected RecordBatch frame for body",
                                        ))));
                                    }
                                };

                            if body_pad > 0 {
                                this.phase = Phase::SkipPad {
                                    remaining: body_pad,
                                };
                            }

                            return Poll::Ready(Some(Ok((batch, custom_metadata))));
                        }
                        Poll::Ready(Ok(false)) => {
                            this.phase = Phase::Body {
                                meta_bytes,
                                body_start,
                                body_filled,
                                body_len,
                                body_pad,
                                custom_metadata,
                            };
                            continue;
                        }
                        Poll::Ready(Err(e)) => {
                            this.eof = true;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Pending => {
                            this.phase = Phase::Body {
                                meta_bytes,
                                body_start,
                                body_filled,
                                body_len,
                                body_pad,
                                custom_metadata,
                            };
                            return Poll::Pending;
                        }
                    }
                }

                Phase::Metadata => {
                    match this.decoder.decode_header(&this.buf)? {
                        IPCFrameHeader::EndOfStream { consumed } => {
                            this.state = BatchState::Done;
                            this.drain_consumed(consumed);
                            return Poll::Ready(None);
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
                                    return Poll::Ready(None);
                                }
                                _ => {
                                    this.drain_consumed(consumed);
                                    return Poll::Ready(Some(Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "unexpected message order",
                                    ))));
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
                                return Poll::Ready(Some(Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "unexpected message order",
                                ))));
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
                                    return Poll::Ready(None);
                                }
                                return Poll::Ready(Some(Err(io::Error::new(
                                    io::ErrorKind::UnexpectedEof,
                                    "stream ended before Arrow EOS marker",
                                ))));
                            }

                            match this.poll_fill_buf(source) {
                                Poll::Ready(Ok(_)) => continue,
                                Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                                Poll::Pending => return Poll::Pending,
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Direct Arrow IPC decoder with zero-copy record batch decoding.
///
/// Yields each batch as an individual SharedBuffer-backed Table. The body
/// Vec64 is recycled across batches - after the caller drops the previous
/// Table, the SharedBuffer releases the Vec64 for reuse.
///
/// The `B` parameter controls frame alignment and must match what the
/// encoder used. Use `Vec64<u8>` for 64-byte SIMD alignment or `Vec<u8>`
/// for standard 8-byte Arrow alignment.
pub struct TableStreamDecoder<B: StreamBuffer = Vec64<u8>> {
    source: Box<dyn AsyncRead + Unpin + Send>,
    core: TableStreamCore<B>,
}

impl<B: StreamBuffer + Unpin> TableStreamDecoder<B> {
    /// Construct a table-stream decoder. Pass `None` for the default per-decode
    /// resource caps applied to every IPC frame consumed from `source`.
    pub fn new<R: AsyncRead + Unpin + Send + 'static>(
        source: R,
        initial_capacity: usize,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self {
        Self {
            source: Box::new(source),
            core: TableStreamCore::new(initial_capacity, protocol, limits),
        }
    }

    /// Return the decoded schema, if available.
    pub fn schema(&self) -> Option<&[Field]> {
        self.core.schema()
    }

    /// Poll for the next decoded table paired with the single custom_metadata
    /// pair from its record batch message.
    #[allow(clippy::type_complexity)]
    fn poll_next_keyed(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<io::Result<(Table, Option<Vec<KeyValue>>)>>> {
        let mut source = AsyncByteSource {
            source: &mut *self.source,
            cx,
        };
        self.core.poll_next_keyed(&mut source)
    }

    /// Read the next decoded table paired with the single custom_metadata pair
    /// from its record batch message.
    pub async fn read_keyed(&mut self) -> Option<io::Result<(Table, Option<Vec<KeyValue>>)>> {
        std::future::poll_fn(|cx| self.poll_next_keyed(cx)).await
    }
}

impl<B: StreamBuffer + Unpin> Stream for TableStreamDecoder<B> {
    type Item = io::Result<Table>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut()
            .poll_next_keyed(cx)
            .map(|opt| opt.map(|res| res.map(|(table, _)| table)))
    }
}
