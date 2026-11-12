// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Arrow IPC streaming codec with zero-copy encode and decode.
//!
//! Owns the persistent state needed for Arrow IPC streaming: the encoder
//! (which tracks schema/dictionary emission), decoded schema fields,
//! accumulated dictionaries, and the SharedBuffer cache for buffer
//! recycling across batches.
//!
//! The codec delegates all IPC frame dispatch to the decoder layer
//! (`decoders::ipc`). Callers feed it raw IPC frames or payloads and
//! receive decoded Tables back.

use std::collections::HashMap;
use std::io;

use minarrow::structs::shared_buffer::SharedBuffer;
use minarrow::{Field, Table, Vec64};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::decoders::ipc::{decode_ipc_frame, decode_ipc_payload};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::encoders::ipc::record_batch::encode_record_batch;
use crate::models::encoders::ipc::table_encoder::TableEncoder;
use crate::traits::decoder::Decoder;
use crate::traits::encoder::Encoder;
use crate::traits::stream_buffer::StreamBuffer;

// Re-export for callers that match on decode results
pub use crate::models::frames::ipc_message::IPCFrameResult;

/// Arrow IPC streaming codec with persistent encoder/decoder state.
///
/// Each instance handles one table type's schema. The encoder emits
/// schema and dictionary frames on the first batch, then only record
/// batches. The decoder accumulates schema and dictionary state from
/// the first payload and reuses it for subsequent batches.
///
/// SharedBuffer caching enables zero-allocation steady-state: when
/// the caller drops the previous Table before decoding the next batch,
/// the Vec64 backing is reclaimed.
pub struct ArrowIpcCodec<B: StreamBuffer> {
    /// Persistent IPC streaming encoder.
    pub(crate) encoder: TableEncoder<B>,
    /// Decoded schema fields learned from the first payload.
    fields: Vec<Field>,
    /// Accumulated dictionaries across payloads.
    dicts: HashMap<i64, Vec<String>>,
    /// Cached SharedBuffer from the previous decode.
    shared_cache: Option<SharedBuffer>,
    /// Resource caps applied during decode of untrusted input. Threaded into
    /// every decoder entry point this codec drives.
    limits: DecodeLimits,
}

impl<B: StreamBuffer + Unpin> ArrowIpcCodec<B> {
    /// Create a new codec for the given schema, protocol, and compression.
    /// Pass `None` for `compression` to write uncompressed batches.
    /// Pass `None` for `limits` to apply the default per-decode resource caps,
    /// or `Some(...)` to override them.
    pub fn new(
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
        limits: Option<DecodeLimits>,
    ) -> Self {
        Self {
            encoder: TableEncoder::new(schema, protocol, compression),
            fields: Vec::new(),
            dicts: HashMap::new(),
            shared_cache: None,
            limits: limits.unwrap_or_default(),
        }
    }

    /// Return the resource limits in effect for this codec.
    pub fn limits(&self) -> DecodeLimits {
        self.limits
    }

    /// Encode one record batch's IPC frames into the streaming output
    /// buffer, appending after any bytes already in `out`.
    ///
    /// Writes IPC frames (schema + dicts on first call, then record
    /// batch only) directly into `out`. The `base_offset` parameter
    /// controls alignment relative to the buffer start - pass 0 for
    /// raw IPC, or the TLV header length for Lightstream framing.
    ///
    /// Returns the number of bytes appended. Streaming callers pair
    /// this with `finish` once they have written every batch they
    /// intend to send through this codec session.
    pub fn encode_stream_batch(
        &mut self,
        view: &minarrow::TableV,
        out: &mut B,
        base_offset: usize,
        custom_metadata: Option<&[(String, String)]>,
    ) -> io::Result<usize> {
        encode_record_batch(&mut self.encoder, view, out, base_offset, custom_metadata)
    }

    /// Decode a contiguous IPC payload containing schema + dicts + record batch.
    ///
    /// Used by the Lightstream protocol where the TLV frame contains
    /// the entire IPC payload in one buffer.
    pub fn decode_payload(&mut self, payload: SharedBuffer) -> io::Result<minarrow::Table> {
        let (table, shared) = decode_ipc_payload::<B>(
            payload,
            &mut self.fields,
            &mut self.dicts,
            self.shared_cache.take(),
            self.limits,
        )?;
        self.shared_cache = Some(shared);
        Ok(table)
    }

    /// Decode a multi-batch IPC Stream payload into the sequence of
    /// record-batch Tables it contains.
    ///
    /// Walks every IPC frame in `bytes` via
    /// [`ArrowIPCFrameDecoder`](crate::models::decoders::ipc::ArrowIPCFrameDecoder),
    /// feeding each frame's metadata and body slices to
    /// [`Self::decode_frame`]. Schema and dictionary frames advance
    /// the codec's internal state and yield no table; record-batch
    /// frames produce one `Table` each. The walk stops at the EOS
    /// marker.
    ///
    /// The input buffer is wrapped as a `SharedBuffer` once and body
    /// slices are zero-copy `SharedBuffer::slice` views into it, so
    /// the column buffers in each returned `Table` reference the
    /// original allocation.
    pub fn decode_stream(&mut self, bytes: Vec64<u8>) -> io::Result<Vec<minarrow::Table>>
    where
        B: 'static,
    {
        use crate::enums::DecodeResult;
        use crate::models::decoders::ipc::ArrowIPCFrameDecoder;
        use crate::models::frames::ipc_message::IPCFrameResult;
        use crate::traits::frame_decoder::FrameDecoder;

        let shared = SharedBuffer::from_vec64(bytes);
        let mut frame_decoder: ArrowIPCFrameDecoder<B> =
            ArrowIPCFrameDecoder::new(self.encoder.protocol, Some(self.limits));
        let total = shared.len();
        let mut tables: Vec<minarrow::Table> = Vec::new();
        let mut pos = 0;

        while pos < total {
            let buf = &shared.as_slice()[pos..];
            match frame_decoder.decode(buf)? {
                DecodeResult::Frame { frame, consumed } => {
                    let meta_range = frame.message_range;
                    let body_range = frame.body_range;
                    if meta_range.is_empty() && body_range.is_empty() {
                        pos += consumed;
                        if pos < total {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "{} bytes of trailing data after Arrow IPC EOS marker",
                                    total - pos
                                ),
                            ));
                        }
                        break;
                    }
                    let meta_bytes =
                        &shared.as_slice()[pos + meta_range.start..pos + meta_range.end];
                    let body_len = body_range.end - body_range.start;
                    let body_shared = shared.slice(pos + body_range.start..pos + body_range.end);
                    match self.decode_frame(meta_bytes, body_shared, body_len)? {
                        IPCFrameResult::Batch(table) => tables.push(table),
                        IPCFrameResult::Schema
                        | IPCFrameResult::Dictionary
                        | IPCFrameResult::EndOfStream => {}
                    }
                    pos += consumed;
                }
                DecodeResult::NeedMore => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Incomplete IPC frame in multi-batch buffer",
                    ));
                }
            }
        }

        Ok(tables)
    }

    /// Decode a single IPC frame from a framed stream.
    ///
    /// Handles all message types: schema, dictionary, record batch, EOS.
    /// Column data is mapped as zero-copy SharedBuffer views for record
    /// batch frames.
    pub fn decode_frame(
        &mut self,
        message: &[u8],
        body: SharedBuffer,
        body_len: usize,
    ) -> io::Result<IPCFrameResult> {
        decode_ipc_frame(
            message,
            body,
            body_len,
            &mut self.fields,
            &mut self.dicts,
            &mut self.shared_cache,
            self.limits,
        )
    }

    /// Access the decoded schema, if available.
    pub fn schema(&self) -> &[Field] {
        &self.fields
    }

    /// Access the accumulated dictionaries.
    pub fn dicts(&self) -> &HashMap<i64, Vec<String>> {
        &self.dicts
    }

    /// Access the protocol in use.
    pub fn protocol(&self) -> IPCMessageProtocol {
        self.encoder.protocol
    }

    /// Check whether schema has been received.
    pub fn has_schema(&self) -> bool {
        !self.fields.is_empty()
    }

    /// Register a dictionary for a categorical column.
    ///
    /// Must be called before the first encode for any column that uses
    /// dictionary encoding. The `id` is the column index.
    pub fn register_dictionary(&mut self, id: i64, values: Vec<String>) {
        self.encoder.register_dictionary(id, values);
    }

    /// Write the EOS marker into the output buffer, finalising the stream.
    ///
    /// For Stream protocol: writes the 8-byte EOS marker.
    /// For File protocol: delegates to the encoder's finish which
    /// handles footer + EOS + magic.
    ///
    /// Must be called after the last record batch is written into the
    /// streaming buffer via `encode_stream_batch`.
    pub fn finish(&mut self, out: &mut B) -> io::Result<()> {
        // EOS marker: continuation (0xFFFFFFFF) + zero metadata length
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        Ok(())
    }
}

/// Codec contract for Arrow IPC Stream: take a `Table`, return a
/// self-contained 64-byte aligned byte buffer holding the schema,
/// dictionary, record-batch, and EOS frames in order. Each call
/// produces a complete decoder-ready payload against the codec's
/// current schema and dictionary state. The trait method is the
/// codec's one-shot encode entry point; streaming and protocol
/// callers use `encode_stream_batch` and `finish` to append frames
/// into a caller-owned buffer across a multi-batch session.
impl Encoder for ArrowIpcCodec<Vec64<u8>> {
    type Input = Table;
    type Error = io::Error;

    fn encode(&mut self, table: &Table) -> io::Result<Vec64<u8>> {
        let mut out: Vec64<u8> = Vec64::new();
        let view = minarrow::TableV::from_table(table.clone(), 0, table.n_rows);
        self.encode_stream_batch(&view, &mut out, 0, None)?;
        self.finish(&mut out)?;
        Ok(out)
    }
}

/// Codec contract for Arrow IPC Stream: take a self-contained byte
/// buffer, parse the schema, dictionary, and record-batch frames it
/// holds, and return the reconstructed `Table`. `decode_owned` is
/// the zero-copy entry: an aligned `Vec64<u8>` is wrapped as a
/// `SharedBuffer` and the IPC parser reads typed views in place
/// without an intermediate copy.
impl Decoder for ArrowIpcCodec<Vec64<u8>> {
    type Output = Table;
    type Error = io::Error;

    fn decode(&mut self, bytes: &[u8]) -> io::Result<Table> {
        self.decode_owned(Vec64::from_slice(bytes))
    }

    fn decode_owned(&mut self, bytes: Vec64<u8>) -> io::Result<Table> {
        let payload = SharedBuffer::from_vec64(bytes);
        self.decode_payload(payload)
    }
}
