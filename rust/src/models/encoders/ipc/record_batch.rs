// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Arrow IPC record batch encoding.
//!
//! Writes column data from a table view's arrays into an output buffer in
//! one pass, with no intermediate IPC body or frame allocations. The
//! output is a standard Arrow IPC record batch frame: continuation
//! marker, flatbuffer metadata, padding, and column data with alignment
//! gaps.
//!
//! Every writer encodes through a [`minarrow::TableV`], with a whole table as the
//! full-width view of itself. Every region borrows from the view's
//! arrays. Where the wire needs transformed values, i.e. string offsets
//! rebased to the window's values start or bit-packed windows at
//! non-byte offsets, the region carries the borrow plus a constant and
//! the transform fuses into the one copy that assembles the frame, so a
//! windowed send moves exactly the same bytes as a whole-table send.
//!
//! Used by both raw Arrow IPC writers and the Lightstream protocol codec.
//! The Lightstream codec wraps the output with a TLV header; raw Arrow
//! writers use it as-is.

use std::io;

use minarrow::ffi::arrow_dtype::ArrowType;
use minarrow::{Array, Bitmask, NumericArray, TableV, TextArray};

use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
use crate::compression::compress;
use crate::enums::{IPCMessageProtocol, WriterState};
use crate::models::encoders::ipc::schema::build_flatbuf_recordbatch;
use crate::models::encoders::ipc::table_stream::TableStreamEncoder;
use crate::models::encoders::ipc::{IPCFrame, IPCFrameEncoder};
use crate::traits::frame_encoder::FrameEncoder;
use crate::traits::stream_buffer::StreamBuffer;
use crate::utils::{align_to, as_bytes, dict_values};

// ---------------------------------------------------------------------------
// Body layout types
// ---------------------------------------------------------------------------

/// Borrowed column bytes for one wire buffer, with any constant the
/// frame write applies while copying.
pub(crate) enum RegionBytes<'a> {
    /// Bytes written verbatim.
    Plain(&'a [u8]),
    /// String offsets entries written minus the rebase constant, so the
    /// receiver's offsets begin at the window's values start.
    OffsetsU32(&'a [u32], u32),
    /// As `OffsetsU32` for 64-bit string offsets.
    #[cfg(feature = "large_string")]
    OffsetsU64(&'a [u64], u64),
    /// Bit-packed bytes written shifted down by the bit offset, for
    /// windows that start inside a byte. The final field is the output
    /// byte count.
    Bits(&'a [u8], u8, usize),
}

impl RegionBytes<'_> {
    /// Number of bytes this region writes into the frame body.
    pub(crate) fn len(&self) -> usize {
        match self {
            RegionBytes::Plain(d) => d.len(),
            RegionBytes::OffsetsU32(o, _) => o.len() * size_of::<u32>(),
            #[cfg(feature = "large_string")]
            RegionBytes::OffsetsU64(o, _) => o.len() * size_of::<u64>(),
            RegionBytes::Bits(_, _, out_len) => *out_len,
        }
    }

    /// Returns true when the region writes no bytes.
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the region's bytes into `out`, applying the region's
    /// constant during the copy. Verbatim regions append the borrowed
    /// slice as-is.
    pub(crate) fn write_into<B: StreamBuffer>(&self, out: &mut B) {
        match self {
            RegionBytes::Plain(d) => out.extend_from_slice(d),
            RegionBytes::OffsetsU32(offs, base) => {
                for v in *offs {
                    out.extend_from_slice(&(v - base).to_le_bytes());
                }
            }
            #[cfg(feature = "large_string")]
            RegionBytes::OffsetsU64(offs, base) => {
                for v in *offs {
                    out.extend_from_slice(&(v - base).to_le_bytes());
                }
            }
            RegionBytes::Bits(bytes, shift, out_len) => {
                for i in 0..*out_len {
                    let lo = bytes[i] >> shift;
                    let hi = if i + 1 < bytes.len() {
                        bytes[i + 1] << (8 - shift)
                    } else {
                        0
                    };
                    out.extend_from_slice(&[lo | hi]);
                }
            }
        }
    }

    /// Bytes for the compression path, borrowed when verbatim and
    /// assembled otherwise. Compression allocates per buffer by nature,
    /// so the uncompressed path never calls this.
    pub(crate) fn compression_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            RegionBytes::Plain(d) => std::borrow::Cow::Borrowed(d),
            _ => {
                let mut owned: Vec<u8> = Vec::with_capacity(self.len());
                match self {
                    RegionBytes::Plain(_) => unreachable!(),
                    RegionBytes::OffsetsU32(offs, base) => {
                        for v in *offs {
                            owned.extend_from_slice(&(v - base).to_le_bytes());
                        }
                    }
                    #[cfg(feature = "large_string")]
                    RegionBytes::OffsetsU64(offs, base) => {
                        for v in *offs {
                            owned.extend_from_slice(&(v - base).to_le_bytes());
                        }
                    }
                    RegionBytes::Bits(bytes, shift, out_len) => {
                        for i in 0..*out_len {
                            let lo = bytes[i] >> shift;
                            let hi = if i + 1 < bytes.len() {
                                bytes[i + 1] << (8 - shift)
                            } else {
                                0
                            };
                            owned.push(lo | hi);
                        }
                    }
                }
                std::borrow::Cow::Owned(owned)
            }
        }
    }
}

/// A wire buffer region plus the alignment padding that follows it.
pub(crate) struct WireRegion<'a> {
    pub(crate) data: RegionBytes<'a>,
    pub(crate) pad: usize,
}

/// Computed body layout for a record batch, collected without copying
/// column data.
pub(crate) struct BodyLayout<'a> {
    pub(crate) regions: Vec<WireRegion<'a>>,
    pub(crate) fb_field_nodes: Vec<fbm::FieldNode>,
    pub(crate) fb_buffers: Vec<fbm::Buffer>,
    pub(crate) body_size: usize,
}

// ---------------------------------------------------------------------------
// Body layout computation
// ---------------------------------------------------------------------------

/// Collect column data slice references for the view's row window and
/// compute the exact IPC body layout without copying any data.
///
/// Returns the regions to write, the flatbuffer metadata vectors, and
/// the exact body size. The caller builds the flatbuffer RecordBatch
/// message with the known body size and writes everything into a single
/// output buffer.
pub(crate) fn compute_body_layout<'a, B: StreamBuffer>(
    view: &'a TableV,
) -> io::Result<BodyLayout<'a>> {
    let n_cols = view.cols.len();
    let mut regions: Vec<WireRegion<'a>> = Vec::with_capacity(n_cols * 3);
    let mut fb_field_nodes: Vec<fbm::FieldNode> = Vec::with_capacity(n_cols);
    let mut fb_buffers: Vec<fbm::Buffer> = Vec::with_capacity(n_cols * 2);
    let mut body_offset = 0usize;

    for (field, col) in view.fields.iter().zip(view.cols.iter()) {
        let nullable = field.nullable;
        let (array, offset, len) = col.as_tuple_ref();
        let null_count = col.null_count();

        match array {
            Array::NumericArray(num) => {
                let (data_bytes, null_mask): (&[u8], Option<&Bitmask>) = match num {
                    NumericArray::Int32(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    NumericArray::Int64(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    NumericArray::UInt32(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    NumericArray::UInt64(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    NumericArray::Float32(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    NumericArray::Float64(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "decimal")]
                    NumericArray::Decimal32(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "decimal")]
                    NumericArray::Decimal64(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    #[cfg(feature = "decimal")]
                    NumericArray::Decimal128(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "unsupported numeric subtype",
                        ));
                    }
                };
                push_null_region::<B>(
                    nullable,
                    null_mask,
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(data_bytes),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            Array::BooleanArray(arr) => {
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    window_bits(arr.data.bits.as_slice(), offset, len),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            Array::TextArray(TextArray::String32(arr)) => {
                let offs = arr.offsets.as_slice();
                let base = offs[offset];
                let end = offs[offset + len] as usize;
                let offsets_region = if base == 0 {
                    RegionBytes::Plain(as_bytes(&offs[offset..offset + len + 1]))
                } else {
                    RegionBytes::OffsetsU32(&offs[offset..offset + len + 1], base)
                };
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    offsets_region,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(&arr.data.as_slice()[base as usize..end]),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => {
                let offs = arr.offsets.as_slice();
                let base = offs[offset];
                let end = offs[offset + len] as usize;
                let offsets_region = if base == 0 {
                    RegionBytes::Plain(as_bytes(&offs[offset..offset + len + 1]))
                } else {
                    RegionBytes::OffsetsU64(&offs[offset..offset + len + 1], base)
                };
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    offsets_region,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(&arr.data.as_slice()[base as usize..end]),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            Array::TextArray(TextArray::Categorical32(arr)) => {
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(as_bytes(&arr.data.as_slice()[offset..offset + len])),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(arr)) => {
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(as_bytes(&arr.data.as_slice()[offset..offset + len])),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => {
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(as_bytes(&arr.data.as_slice()[offset..offset + len])),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => {
                push_null_region::<B>(
                    nullable,
                    arr.null_mask.as_ref(),
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(as_bytes(&arr.data.as_slice()[offset..offset + len])),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp) => {
                let (data_bytes, null_mask): (&[u8], Option<&Bitmask>) = match temp {
                    minarrow::TemporalArray::Datetime32(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    minarrow::TemporalArray::Datetime64(arr) => {
                        (as_bytes(&arr.data.as_slice()[offset..offset + len]), arr.null_mask.as_ref())
                    }
                    minarrow::TemporalArray::Null => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "null temporal array not supported",
                        ));
                    }
                };
                push_null_region::<B>(
                    nullable,
                    null_mask,
                    offset,
                    len,
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                push_data_region::<B>(
                    RegionBytes::Plain(data_bytes),
                    &mut body_offset,
                    &mut regions,
                    &mut fb_buffers,
                );
                fb_field_nodes.push(fbm::FieldNode::new(len as i64, null_count as i64));
            }

            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported column type: {}", field.name),
                ));
            }
        }
    }

    Ok(BodyLayout {
        regions,
        fb_field_nodes,
        fb_buffers,
        body_size: body_offset,
    })
}

// ---------------------------------------------------------------------------
// Record batch encoding
// ---------------------------------------------------------------------------

/// Encode a table view as IPC frames, appending to the caller's buffer.
///
/// Handles schema and dictionary frames on the first batch via the encoder's
/// internal state. Appends the complete IPC frame sequence: prefix frames
/// then the record batch frame with column data.
///
/// `base_offset` is the byte position in `out` where IPC data starts,
/// used for alignment calculations so the body lands on a SIMD boundary
/// when the buffer's base address is aligned. Callers that prepend a wire
/// header pass `out.len()` before calling. Callers writing IPC-only pass 0.
///
/// Handles compression when enabled: each buffer is compressed individually
/// with a u64 LE uncompressed length prefix, per the Arrow IPC spec.
///
/// Returns the number of bytes appended to `out`.
pub(crate) fn encode_record_batch<B: StreamBuffer + Unpin>(
    encoder: &mut TableStreamEncoder<B>,
    view: &TableV,
    out: &mut B,
    base_offset: usize,
    custom_metadata: Option<&[(String, String)]>,
) -> io::Result<usize> {
    // Register dictionary values from categorical columns
    for (i, (field, col)) in view.fields.iter().zip(view.cols.iter()).enumerate() {
        if let ArrowType::Dictionary(_) = field.dtype {
            let uniques = dict_values(col.as_tuple_ref().0).unwrap_or_default();
            encoder.register_dictionary(i as i64, uniques);
        }
    }

    // Encode schema + dictionary prefix frames on the first batch
    let mut prefix_frames: Vec<Vec<u8>> = Vec::new();
    if encoder.state == WriterState::Fresh {
        let schema_meta = encoder.encode_schema()?;
        // Emit schema as an IPC frame
        let frame = IPCFrame {
            meta: &schema_meta,
            body: &[],
            protocol: encoder.protocol,
            is_first: encoder.protocol == IPCMessageProtocol::File,
            is_last: false,
            footer_bytes: None,
        };
        let mut frame_offset = 0usize;
        let (encoded, _) = IPCFrameEncoder::encode::<B>(&mut frame_offset, &frame)?;
        prefix_frames.push(encoded.as_ref().to_vec());
    }
    let dict_ids = encoder.pending_dict_ids();
    for dict_id in dict_ids {
        if let Some((meta, body_vec)) = encoder.encode_dictionary(dict_id)? {
            let frame = IPCFrame {
                meta: &meta,
                body: &body_vec,
                protocol: encoder.protocol,
                is_first: false,
                is_last: false,
                footer_bytes: None,
            };
            let mut frame_offset = 0usize;
            let (encoded, _) = IPCFrameEncoder::encode::<B>(&mut frame_offset, &frame)?;
            prefix_frames.push(encoded.as_ref().to_vec());
        }
    }

    let prefix_size: usize = prefix_frames.iter().map(|f| f.len()).sum();

    // Compute the record batch body layout without copying column data
    let layout = compute_body_layout::<B>(view)?;

    // If compression is active, compress each buffer and recompute sizes.
    // Each compressed buffer gets a u64 LE uncompressed length prefix.
    let compressed: Option<Vec<Vec<u8>>> = if let Some(codec) = encoder.compression {
        let mut bufs = Vec::with_capacity(layout.regions.len());
        for region in &layout.regions {
            if region.data.is_empty() {
                bufs.push(Vec::new());
            } else {
                let raw = region.data.compression_bytes();
                let c = compress(&raw, codec)
                    .map_err(|e| io::Error::other(format!("{}", e)))?;
                let mut wire = Vec::with_capacity(8 + c.len());
                wire.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                wire.extend_from_slice(&c);
                bufs.push(wire);
            }
        }
        Some(bufs)
    } else {
        None
    };

    // Recompute buffer metadata if compressed
    let (body_size, fb_buffers) = if let Some(ref comp) = compressed {
        let mut bufs = Vec::with_capacity(comp.len());
        let mut offset = 0usize;
        for c in comp {
            let len = c.len();
            bufs.push(fbm::Buffer::new(offset as i64, len as i64));
            let pad = align_to::<B>(len);
            offset += len + pad;
        }
        (offset, bufs)
    } else {
        (layout.body_size, layout.fb_buffers)
    };

    let compression_type = match encoder.compression {
        Some(c) => Some(c.to_arrow_ipc_type()?),
        None => None,
    };
    let meta = build_flatbuf_recordbatch(
        &mut encoder.fbb,
        view.len,
        &layout.fb_field_nodes,
        &fb_buffers,
        body_size,
        compression_type,
        custom_metadata,
    )?;

    // Compute IPC frame sizes with alignment
    let meta_end = base_offset + prefix_size + 4 + 4 + meta.len();
    let meta_pad = align_to::<B>(meta_end);
    let body_end = meta_end + meta_pad + body_size;
    let body_pad = align_to::<B>(body_end);

    let ipc_size = prefix_size + 4 + 4 + meta.len() + meta_pad + body_size + body_pad;
    out.reserve(ipc_size);

    // Write prefix frames (schema + dicts) into the output buffer
    for frame in &prefix_frames {
        out.extend_from_slice(frame);
    }

    // IPC continuation marker
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());

    // Metadata size field (includes padding)
    out.extend_from_slice(&((meta.len() + meta_pad) as u32).to_le_bytes());

    // Metadata bytes
    out.extend_from_slice(&meta);

    // Metadata padding
    if meta_pad > 0 {
        out.extend_from_slice(&[0u8; 64][..meta_pad]);
    }

    // Column data - compressed or raw
    if let Some(ref comp) = compressed {
        for c in comp {
            out.extend_from_slice(c);
            let pad = align_to::<B>(c.len());
            if pad > 0 {
                out.extend_from_slice(&[0u8; 64][..pad]);
            }
        }
    } else {
        for region in &layout.regions {
            region.data.write_into(out);
            if region.pad > 0 {
                out.extend_from_slice(&[0u8; 64][..region.pad]);
            }
        }
    }

    // Body padding
    if body_pad > 0 {
        out.extend_from_slice(&[0u8; 64][..body_pad]);
    }

    Ok(ipc_size)
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

/// Bit-packed bytes for the bit window `[offset, offset + len)`. Byte
/// aligned windows write verbatim and other offsets carry the shift the
/// frame write applies while copying.
fn window_bits(bits: &[u8], offset: usize, len: usize) -> RegionBytes<'_> {
    let out_len = len.div_ceil(8);
    if offset % 8 == 0 {
        return RegionBytes::Plain(&bits[offset / 8..offset / 8 + out_len]);
    }
    RegionBytes::Bits(
        &bits[offset / 8..(offset + len).div_ceil(8)],
        (offset % 8) as u8,
        out_len,
    )
}

/// Record a null mask region for the row window into the body layout.
///
/// When the column is nullable and has a mask, pushes the mask's
/// windowed bytes. Otherwise records a zero-length buffer in the
/// flatbuffer metadata and an empty region, keeping regions and buffer
/// metadata index-aligned. The compression path rebuilds the buffer
/// metadata one entry per region, so a metadata-only placeholder would
/// shift every buffer that follows it.
fn push_null_region<'a, B: StreamBuffer>(
    nullable: bool,
    mask: Option<&'a Bitmask>,
    offset: usize,
    len: usize,
    body_offset: &mut usize,
    regions: &mut Vec<WireRegion<'a>>,
    fb_buffers: &mut Vec<fbm::Buffer>,
) {
    if nullable
        && let Some(m) = mask {
            let data = window_bits(m.bits.as_slice(), offset, len);
            let pad = align_to::<B>(data.len());
            fb_buffers.push(fbm::Buffer::new(*body_offset as i64, data.len() as i64));
            *body_offset += data.len() + pad;
            regions.push(WireRegion { data, pad });
            return;
        }
    fb_buffers.push(fbm::Buffer::new(0, 0));
    regions.push(WireRegion {
        data: RegionBytes::Plain(&[]),
        pad: 0,
    });
}

/// Record a data region into the body layout.
fn push_data_region<'a, B: StreamBuffer>(
    data: RegionBytes<'a>,
    body_offset: &mut usize,
    regions: &mut Vec<WireRegion<'a>>,
    fb_buffers: &mut Vec<fbm::Buffer>,
) {
    let pad = align_to::<B>(data.len());
    fb_buffers.push(fbm::Buffer::new(*body_offset as i64, data.len() as i64));
    *body_offset += data.len() + pad;
    regions.push(WireRegion { data, pad });
}
