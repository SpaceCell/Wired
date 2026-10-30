// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # IPC Parser
//!
//! Parses Arrow IPC `RecordBatch` messages into a `minarrow::Table`.
//!
//! ## Supported columns
//! - Fixed-width numerics: i32/i64, u32/u64, f32/f64
//!   - Optional: i8/u8, i16/u16 via `extended_numeric_types`
//! - Boolean (bit-packed)
//! - UTF-8 / LargeUTF-8 (32/64-bit offsets; `LargeString` behind `large_string`)
//! - Dictionary-encoded text (UInt32 by default; UInt8/UInt16/UInt64 behind `extended_categorical`)
//! - Date32/Date64 behind `datetime`
//!
//! ## Zero-copy behaviour
//! If an Arc-backed body is supplied and buffers are correctly aligned, columns are
//! created as shared views without buffer copies. Otherwise the decoder falls back to copying.
//!
//! ## Errors and limits
//! - Compressed `RecordBatch` bodies are supported when the `zstd` feature is
//!   enabled. Per-buffer decompression allocates a new Vec64 for the body.
//! - Dictionary delta batches are rejected.
//! - All buffer regions are bounds-checked; malformed metadata yields `io::Error`.
//!
//! See the [Apache Arrow IPC specification](https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format).

//! RecordBatch -> Table decoder supporting Minarrow fixed‑width, Boolean,
//! UTF‑8, LargeUTF‑8, and Dictionary columns.

use std::io;
use std::sync::Arc;
use log::warn;
#[cfg(feature = "default_categorical_8")]
use tracing::debug;

use flatbuffers::Vector;
#[cfg(feature = "datetime")]
use minarrow::enums::time_units::TimeUnit as MnTimeUnit;
use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use minarrow::*;

use crate::arrow::message::org::apache::arrow::flatbuf as fb;
use crate::arrow::message::org::apache::arrow::flatbuf::Buffer;
#[cfg(feature = "zstd")]
use crate::compression::ipc::decompress_ipc_body;
use crate::debug_println;
use crate::models::decoders::limits::DecodeLimits;
use crate::{AFMessage, AFMessageHeader};
use std::collections::{HashMap, HashSet};

/// Used for parsing a RecordBatch into a Minarrow `Table`.
///
/// Can also be 'intercepted' and used in extreme performance-critical
/// situations to avoid async overhead.
pub struct RecordBatchParser;

impl RecordBatchParser {
    /// Parses a RecordBatch into a Minarrow `Table`.
    ///
    /// `arc_opt` may supply the backing buffer; if `None` we allocate a single
    /// `Arc<Vec64<u8>>` wrapping `arrow_buf` so that all aligned columns can
    /// reuse it without copies.
    pub fn parse_record_batch<'a>(
        message: &AFMessage<'a>,
        arrow_buf: &'a [u8],
        fields: &[Field],
        arc_opt: Option<Arc<[u8]>>,
    ) -> io::Result<Table> {
        if message.header_type() != AFMessageHeader::RecordBatch {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Expected RecordBatch header",
            ));
        }

        let header = message.header_as_record_batch().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Missing RecordBatch payload")
        })?;

        let n_rows = header.length() as usize;
        let nodes = header
            .nodes()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing nodes"))?;
        let fbuf_meta = header
            .buffers()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing fbuf_meta"))?;

        // Handle compressed body: decompress into a Vec64, build an Arc
        // from it for zero-copy column views within the decompressed buffer.
        #[cfg(feature = "zstd")]
        #[allow(clippy::type_complexity)]
        let decompressed: Option<(Vec64<u8>, Vec<(usize, usize)>)> =
            if let Some(ref compression) = header.compression() {
                Some(decompress_ipc_body::<Vec<u8>>(
                    arrow_buf,
                    &fbuf_meta,
                    compression,
                    DecodeLimits::default(),
                )?)
            } else {
                None
            };
        #[cfg(not(feature = "zstd"))]
        let decompressed: Option<(Vec64<u8>, Vec<(usize, usize)>)> = {
            if header.compression().is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Compressed IPC requires the 'zstd' feature",
                ));
            }
            None
        };

        let (arrow_buf, arc_opt, corrections) = if let Some((ref dec, ref corr)) = decompressed {
            let arc: Arc<[u8]> = Arc::from(dec.as_slice());
            (dec.as_slice(), Some(arc), Some(corr.as_slice()))
        } else {
            (arrow_buf, arc_opt, None)
        };

        if nodes.len() != fields.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Field count mismatch",
            ));
        }

        let mut cols = Vec::with_capacity(fields.len());
        let mut buffer_idx = 0;

        for (i, field) in fields.iter().enumerate() {
            let node = nodes.get(i);
            let field_len = node.length() as usize;
            let null_count = node.null_count() as usize;

            if field_len != n_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Row count mismatch for {}", field.name),
                ));
            }

            let null_mask = Self::extract_null_mask(
                field,
                field_len,
                null_count,
                &fbuf_meta,
                &mut buffer_idx,
                arrow_buf,
                corrections,
            )?;

            let arr = match &field.dtype {
                // numeric primitives
                ArrowType::Int32 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i32>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                ArrowType::Int64 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i64>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                ArrowType::UInt32 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<u32>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::UInt32(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                ArrowType::UInt64 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<u64>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::UInt64(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                ArrowType::Float32 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<f32>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Float32(Arc::new(FloatArray::new(
                        data, null_mask,
                    ))))
                }
                ArrowType::Float64 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<f64>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray::new(
                        data, null_mask,
                    ))))
                }
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int8 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data = unsafe { Self::buffer_from_slice::<i8>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Int8(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int16 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i16>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Int16(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::UInt8 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data = unsafe { Self::buffer_from_slice::<u8>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::UInt8(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::UInt16 => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<u16>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::UInt16(Arc::new(IntegerArray::new(
                        data, null_mask,
                    ))))
                }

                // ---- boolean ---------------------------------------------------------------
                ArrowType::Boolean => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    // Bitmask::from_bytes carries field_len, so new() sets
                    // the array length to match.
                    let bool_data = Bitmask::from_bytes(slice, field_len);
                    let bool_array = BooleanArray::new(bool_data, null_mask);
                    Array::BooleanArray(Arc::new(bool_array))
                }

                // ---- UTF-8 -----------------------------------------------------------------
                // The schema declares the offset width: Utf8 carries u32
                // offsets and LargeUtf8 carries u64 offsets.
                ArrowType::String => {
                    let (data, offsets) = Self::parse_utf8_array::<u32>(
                        arrow_buf,
                        &fbuf_meta,
                        &mut buffer_idx,
                        field_len,
                        &field.name,
                        &arc_opt,
                        corrections,
                    )?;
                    Array::TextArray(TextArray::String32(Arc::new(StringArray::new(
                        data, null_mask, offsets,
                    ))))
                }
                #[cfg(feature = "large_string")]
                ArrowType::LargeString => {
                    let (data, offsets) = Self::parse_utf8_array::<u64>(
                        arrow_buf,
                        &fbuf_meta,
                        &mut buffer_idx,
                        field_len,
                        &field.name,
                        &arc_opt,
                        corrections,
                    )?;
                    Array::TextArray(TextArray::String64(Arc::new(StringArray::new(
                        data, null_mask, offsets,
                    ))))
                }
                #[cfg(feature = "datetime")]
                ArrowType::DateTime(dt @ (Dt::Date32 | Dt::Time32(_))) => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i32>(slice, field_len, &arc_opt) };
                    Array::TemporalArray(TemporalArray::Datetime32(Arc::new(DatetimeArray::new(
                        data,
                        null_mask,
                        dt.clone(),
                    ))))
                }
                #[cfg(feature = "datetime")]
                ArrowType::DateTime(
                    dt @ (Dt::Date64 | Dt::Timestamp(_, _) | Dt::Time64(_) | Dt::Duration64(_)),
                ) => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i64>(slice, field_len, &arc_opt) };
                    Array::TemporalArray(TemporalArray::Datetime64(Arc::new(DatetimeArray::new(
                        data,
                        null_mask,
                        dt.clone(),
                    ))))
                }
                // dictionary
                ArrowType::Dictionary(idx_ty) => {
                    // indices
                    let (idx_slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;

                    // unique offsets + bytes
                    let (off_start, off_len) = if let Some(c) = corrections {
                        c[buffer_idx]
                    } else {
                        let m = fbuf_meta.get(buffer_idx);
                        (m.offset() as usize, m.length() as usize)
                    };
                    buffer_idx += 1;
                    let (val_start, val_len) = if let Some(c) = corrections {
                        c[buffer_idx]
                    } else {
                        let m = fbuf_meta.get(buffer_idx);
                        (m.offset() as usize, m.length() as usize)
                    };
                    buffer_idx += 1;

                    let off_slice = &arrow_buf[off_start..off_start + off_len];
                    let val_slice = &arrow_buf[val_start..val_start + val_len];

                    let unique_values =
                        parse_dictionary_strings(off_slice, val_slice, DecodeLimits::default())?;

                    // choose variant by index type
                    match idx_ty {
                        CategoricalIndexType::UInt32 => {
                            let data_buf: minarrow::Buffer<u32> = unsafe {
                                Self::buffer_from_slice::<u32>(idx_slice, field_len, &arc_opt)
                            };
                            Array::TextArray(TextArray::Categorical32(Arc::new(
                                CategoricalArray::<u32>::new(data_buf, unique_values, null_mask),
                            )))
                        }
                        #[cfg(feature = "default_categorical_8")]
                        CategoricalIndexType::UInt8 => {
                            debug!(
                                "DEBUG parse_record_batch: Creating Categorical8, field_len={}, idx_slice.len()={}, unique_values.len()={}, null_mask={:?}",
                                field_len,
                                idx_slice.len(),
                                unique_values.len(),
                                null_mask.as_ref().map(|m| m.len())
                            );
                            let data8 = unsafe {
                                Self::buffer_from_slice::<u8>(idx_slice, field_len, &arc_opt)
                            };
                            Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray::<
                                u8,
                            >::new(
                                data8,
                                unique_values,
                                null_mask,
                            ))))
                        }
                        #[cfg(feature = "extended_categorical")]
                        CategoricalIndexType::UInt16 => {
                            let data16 = unsafe {
                                Self::buffer_from_slice::<u16>(idx_slice, field_len, &arc_opt)
                            };
                            Array::TextArray(TextArray::Categorical16(Arc::new(
                                CategoricalArray::<u16>::new(data16, unique_values, null_mask),
                            )))
                        }
                        #[cfg(feature = "extended_categorical")]
                        CategoricalIndexType::UInt64 => {
                            let data64 = unsafe {
                                Self::buffer_from_slice::<u64>(idx_slice, field_len, &arc_opt)
                            };
                            Array::TextArray(TextArray::Categorical64(Arc::new(
                                CategoricalArray::<u64>::new(data64, unique_values, null_mask),
                            )))
                        }
                    }
                }

                #[cfg(feature = "decimal")]
                ArrowType::Decimal32(p, s) => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i32>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Decimal32(Arc::new(
                        DecimalArray { data, null_mask, precision: *p, scale: *s },
                    )))
                }
                #[cfg(feature = "decimal")]
                ArrowType::Decimal64(p, s) => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i64>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Decimal64(Arc::new(
                        DecimalArray { data, null_mask, precision: *p, scale: *s },
                    )))
                }
                #[cfg(feature = "decimal")]
                ArrowType::Decimal128(p, s) => {
                    let (slice, _) = Self::extract_buffer_slice(
                        &fbuf_meta,
                        &mut buffer_idx,
                        arrow_buf,
                        &field.name,
                        corrections,
                    )?;
                    let data =
                        unsafe { Self::buffer_from_slice::<i128>(slice, field_len, &arc_opt) };
                    Array::NumericArray(NumericArray::Decimal128(Arc::new(
                        DecimalArray { data, null_mask, precision: *p, scale: *s },
                    )))
                }

                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Unsupported type: {:?}", other),
                    ));
                }
            };

            cols.push(FieldArray::new(field.clone(), arr));
        }

        Ok(Table {
            cols,
            n_rows,
            name: "RecordBatch".to_owned(),
            ..Default::default()
        })
    }

    #[inline]
    pub fn extract_buffer_slice<'a>(
        fbuf_meta: &Vector<'a, Buffer>,
        buffer_idx: &mut usize,
        arrow_buf: &'a [u8],
        field_name: &str,
        corrections: Option<&[(usize, usize)]>,
    ) -> io::Result<(&'a [u8], usize)> {
        let idx = *buffer_idx;
        *buffer_idx += 1;

        let (offset, length) = if let Some(c) = corrections {
            c[idx]
        } else {
            let buf = fbuf_meta.get(idx);
            (buf.offset() as usize, buf.length() as usize)
        };

        // Reject offset + length that overflows usize before comparing
        // against the buffer length. A crafted i64-cast-to-usize pair (large
        // negative i64 cast to large usize) would otherwise wrap past the
        // bounds check and select an invalid slice.
        let end = offset.checked_add(length).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "buffer offset+length overflow")
        })?;
        if end > arrow_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Buffer out of bounds for {}", field_name),
            ));
        }

        Ok((&arrow_buf[offset..end], offset))
    }

    /// Turn a raw byte‑slice into a `Buffer<T>`.
    ///
    /// * If the bytes are properly aligned **and** we have an `Arc`,
    ///   we create a `Shared` view.
    /// * If aligned and no `Arc` is available, we copy into an
    ///   owned `Vec64<T>`.
    ///   in which case `decode_fixed_width_batch` supplies the `Arc`.
    /// * If unaligned we always copy.
    ///
    /// # Safety
    ///
    /// `slice` must contain at least `len * size_of::<T>()` bytes of
    /// initialised data whose bit patterns are valid values of `T`. When
    /// `arc_bytes` is `Some`, the `Arc` must own the allocation backing
    /// `slice` so the shared view keeps the bytes alive.
    #[inline]
    pub unsafe fn buffer_from_slice<T: Copy>(
        slice: &[u8],
        len: usize,
        arc_bytes: &Option<Arc<[u8]>>,
    ) -> minarrow::Buffer<T> {
        let ptr = slice.as_ptr() as *const T;

        // Alignment diagnostics. Computed inline so the bindings do not
        // sit unused in release where `debug_println!` expands to nothing.
        debug_println!(
            "Creating buffer with:\nAligned 8: {:?}\nAligned 64: {:?}\narc_bytes is some: {:?}\n",
            (ptr as usize).is_multiple_of(8),
            (ptr as usize).is_multiple_of(64),
            arc_bytes.is_some()
        );

        if ptr as usize & 63 == 0 {
            if let Some(arc) = arc_bytes {
                debug_println!("Aligned: Creating buffer with arc_bytes");
                // SAFETY: ptr must be within arc's allocation, correctly aligned and cover [ptr, ptr+len*T].
                // if it is not 64-byte aligned, the function will copy data to an owned buffer and flag it.
                unsafe { minarrow::Buffer::from_shared_raw(arc.clone(), ptr, len) }
            } else {
                Self::warn_no_arc_bytes_copy_once();
                // No reusable Arc -> copy.
                let mut v = Vec64::with_capacity(len);
                unsafe { std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), len) };
                unsafe { v.set_len(len) };
                minarrow::Buffer::from(v)
            }
        } else {
            Self::warn_unaligned_copy_once();
            let elem_size = std::mem::size_of::<T>();
            let mut v = Vec64::with_capacity(len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    v.as_mut_ptr() as *mut u8,
                    len * elem_size,
                )
            };
            unsafe { v.set_len(len) };
            minarrow::Buffer::from(v)
        }
    }

    /// One-shot warning: emitted once per process the first time the
    /// decoder copies a misaligned buffer into a fresh `Vec64`. The
    /// source IPC file was produced by a writer that did not align
    /// buffers to 64 bytes; the decoder still works, but a copy is
    /// added on every misaligned buffer.
    #[inline(never)]
    fn warn_unaligned_copy_once() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            warn!(
                "decoder copying misaligned input buffer; source file is not 64-byte aligned"
            );
        });
    }

    /// One-shot warning: emitted once per process the first time the
    /// decoder is handed an aligned source pointer with no owning
    /// allocation reference, forcing a copy of the buffer rather than
    /// a zero-copy reference.
    #[inline(never)]
    fn warn_no_arc_bytes_copy_once() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            warn!(
                "decoder copying aligned input buffer; no owning allocation reference supplied"
            );
        });
    }

    /// Consume a null-mask‑bitmap buffer when present and build the proper
    /// `Bitmask`.  
    /// * For nullable fields we always return `Some(Bitmask)`  
    ///   - all‑true when the bitmap length is 0.  
    /// * For non‑nullable fields we skip an unexpected null mask
    ///   (length 0 or `ceil(n_rows/8)`) but return `None`.
    #[inline]
    pub fn extract_null_mask<'a>(
        field: &Field,
        field_len: usize,
        null_count: usize,
        fbuf_meta: &Vector<'a, Buffer>,
        buffer_idx: &mut usize,
        arrow_buf: &'a [u8],
        corrections: Option<&[(usize, usize)]>,
    ) -> io::Result<Option<Bitmask>> {
        // handle non-nullable fields
        if !field.nullable {
            // Peek at the next buffer: it *may* be a redundant bitmap.
            let len_bytes = if let Some(c) = corrections {
                c[*buffer_idx].1
            } else {
                fbuf_meta.get(*buffer_idx).length() as usize
            };
            let expected_null_mask_len = field_len.div_ceil(8); // ceil(n/8)

            if len_bytes == 0 || len_bytes == expected_null_mask_len {
                // It is a null mask - consume it, but ignore contents.
                *buffer_idx += 1;
            }
            return Ok(None);
        }

        // nullable: we must consume exactly one buffer
        let idx = *buffer_idx;
        *buffer_idx += 1;

        let (offset, len_bytes) = if let Some(c) = corrections {
            c[idx]
        } else {
            let buf = fbuf_meta.get(idx);
            (buf.offset() as usize, buf.length() as usize)
        };

        // If writer says `null_count == 0` **or** the bitmap is empty,
        // fabricate an all-true mask.
        if null_count == 0 || len_bytes == 0 {
            return Ok(Some(Bitmask::new_set_all(field_len, true)));
        }

        // Handle real bitmap with bounds-check and build
        if offset + len_bytes > arrow_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Null buffer out of bounds for {}", field.name),
            ));
        }
        let bytes = &arrow_buf[offset..offset + len_bytes];
        Ok(Some(Bitmask::from_bytes(bytes, field_len)))
    }

    // Decodes an Arrow UTF8 or LargeUTF8 array from the IPC buffers.
    // This extracts both the concatenated string data (`Buffer<u8>`)
    // and the corresponding offset buffer (`Buffer<OffsetType>`).
    #[inline]
    pub fn parse_utf8_array<'a, OffsetType: Copy>(
        arrow_buf: &'a [u8],
        fbuf_meta: &Vector<'a, Buffer>,
        buffer_idx: &mut usize,
        field_len: usize,
        field_name: &str,
        arc_opt: &Option<Arc<[u8]>>,
        corrections: Option<&[(usize, usize)]>,
    ) -> io::Result<(minarrow::Buffer<u8>, minarrow::Buffer<OffsetType>)> {
        let (offsets_o, offsets_l) = if let Some(c) = corrections {
            c[*buffer_idx]
        } else {
            let buf = fbuf_meta.get(*buffer_idx);
            (buf.offset() as usize, buf.length() as usize)
        };
        let (values_o, values_l) = if let Some(c) = corrections {
            c[*buffer_idx + 1]
        } else {
            let buf = fbuf_meta.get(*buffer_idx + 1);
            (buf.offset() as usize, buf.length() as usize)
        };

        if offsets_o + offsets_l > arrow_buf.len() || values_o + values_l > arrow_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("String buffer out of bounds for {}", field_name),
            ));
        }

        let data = unsafe {
            Self::buffer_from_slice::<u8>(
                &arrow_buf[values_o..values_o + values_l],
                values_l,
                arc_opt,
            )
        };

        let offsets = unsafe {
            let off_slice = &arrow_buf[offsets_o..offsets_o + offsets_l];
            Self::buffer_from_slice::<OffsetType>(off_slice, field_len + 1, arc_opt)
        };

        *buffer_idx += 2;
        Ok((data, offsets))
    }
}

// ------------------------- Format Handlers ------------------------------------------------//

/// Parses and inserts a dictionary batch from Arrow IPC into the provided dictionary map.
///
/// Handles dictionary batches for categorical columns as per the Arrow IPC specification.
/// Validates offsets and buffer lengths. Returns error if out of bounds or malformed.
#[inline(always)]
pub fn handle_dictionary_batch(
    db: &fb::DictionaryBatch,
    body: &[u8],
    dicts: &mut HashMap<i64, Vec<String>>,
    limits: DecodeLimits,
) -> io::Result<()> {
    let is_delta = db.isDelta();
    let dict_id = db.id();

    // Delta dictionaries append to an existing base. Reject if no base exists.
    if is_delta && !dicts.contains_key(&dict_id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "delta dictionary for id {} received before base dictionary",
                dict_id
            ),
        ));
    }

    let rec = db
        .data()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad dict batch"))?;
    let buffers = rec
        .buffers()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no buffers"))?;
    if buffers.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "dictionary batch buffers < 3",
        ));
    }
    let off_meta = buffers.get(1);
    let data_meta = buffers.get(2);
    let off_off = off_meta.offset() as usize;
    let off_len = off_meta.length() as usize;
    let data_off = data_meta.offset() as usize;
    let data_len = data_meta.length() as usize;

    if off_off + off_len > body.len() || data_off + data_len > body.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "dictionary batch buffer out of bounds: off_off+off_len={}+{} or data_off+data_len={}+{} > body.len()={}",
                off_off,
                off_len,
                data_off,
                data_len,
                body.len()
            ),
        ));
    }

    let offs_slice = &body[off_off..off_off + off_len];
    let values =
        parse_dictionary_strings(offs_slice, &body[data_off..data_off + data_len], limits)?;

    if is_delta {
        // Append new values to the existing dictionary. The is_delta + contains
        // check above guarantees this entry exists, but we still surface a
        // proper error rather than panicking if the invariant is ever broken
        // by future edits.
        dicts
            .get_mut(&dict_id)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("delta dictionary base for id {} missing", dict_id),
                )
            })?
            .extend(values);
    } else {
        // Replace with new base dictionary
        dicts.insert(dict_id, values.into_iter().collect());
    }
    Ok(())
}

/// Parse dictionary string values from offset and data buffers.
///
/// Dictionary values are declared string (Utf8) in the schema, so the body
/// carries u32 offsets. Per the Arrow columnar format, N strings carry N+1
/// offsets, where the final offset is the total length of the data buffer.
fn parse_dictionary_strings(
    offs_slice: &[u8],
    data_slice: &[u8],
    limits: DecodeLimits,
) -> io::Result<Vec64<String>> {
    let offset_size = std::mem::size_of::<u32>();
    let count = offs_slice.len() / offset_size;
    // The Arrow columnar format stores N+1 offsets for N strings, so a valid
    // buffer holds at least 2 - one string plus the final total-length
    // offset. The guard also stops `count - 1` from underflowing to
    // usize::MAX and blowing up Vec64::with_capacity.
    if count < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "dictionary batch offset count < 2",
        ));
    }
    // Bound the per-batch entry count before allocating.
    limits.check(count - 1, limits.max_dictionary_entries, "dictionary entries")?;
    let read_offset = |k: usize| -> usize {
        let bytes = &offs_slice[k * offset_size..(k + 1) * offset_size];
        u32::from_le_bytes(bytes.try_into().unwrap()) as usize
    };
    let mut values = Vec64::with_capacity(count - 1);
    for i in 0..(count - 1) {
        let start = read_offset(i);
        let end = read_offset(i + 1);
        if end > data_slice.len() || start > end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dictionary batch string slice out of bounds",
            ));
        }
        let s = std::str::from_utf8(&data_slice[start..end])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        values.push(s.to_owned());
    }
    Ok(values)
}

/// Arena-based RecordBatch decoder for streaming zero-copy ingestion.
///
/// Uses a single `Arena` allocation for all column buffers in a batch,
/// reducing per-batch allocation count from O(columns) to O(1). Each column
/// buffer sits in a 64-byte aligned region within the arena. After freezing,
/// all columns share one `SharedBuffer` refcount.
///
/// The body bytes are copied into Arena regions as a structured write. With
/// `MAllocPg64` backing the Arena, large allocations use mmap and growth
/// uses `mremap` - a page table operation rather than a memcpy.
///
/// When `projection` is `None`, all columns are decoded. When `Some`, only
/// the columns whose indices appear in the set are materialised. Buffer
/// descriptors for skipped columns are still consumed to keep the sequential
/// buffer index in sync with the IPC metadata. Only array construction and
/// SharedBuffer slicing are avoided for non-projected columns, so mmap
/// pages for skipped columns are never faulted in.
#[allow(clippy::too_many_arguments)]
pub fn decode_record_batch(
    rec: &fb::RecordBatch,
    fields: &[Field],
    dicts: &HashMap<i64, Vec<String>>,
    shared: SharedBuffer,
    body_start: usize,
    body_len: usize,
    projection: Option<&HashSet<usize>>,
    limits: DecodeLimits,
) -> io::Result<(Table, SharedBuffer)> {
    let nodes = rec
        .nodes()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no nodes"))?;
    let buffers = rec
        .buffers()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no buffers"))?;

    // Cap declared counts before any per-element loop or capacity allocation.
    limits.check(nodes.len(), limits.max_fields, "record batch nodes")?;
    limits.check(buffers.len(), limits.max_buffers, "record batch buffers")?;

    if nodes.len() != fields.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Field count mismatch: {} nodes vs {} fields",
                nodes.len(),
                fields.len()
            ),
        ));
    }

    // Handle compressed body: decompress into a new Vec64, replace the
    // shared buffer, and use corrected offsets for all buffer access.
    // The uncompressed path is untouched - this is a single branch check.
    #[cfg(feature = "zstd")]
    let (shared, body_start, body_len, corrections_storage) = if let Some(ref compression) =
        rec.compression()
    {
        let body = &shared.as_slice()[body_start..body_start + body_len];
        let (dec, corr) = decompress_ipc_body::<Vec64<u8>>(body, &buffers, compression, limits)?;
        let len = dec.len();
        (SharedBuffer::from_vec64(dec), 0, len, Some(corr))
    } else {
        (shared, body_start, body_len, None)
    };
    #[cfg(not(feature = "zstd"))]
    let corrections_storage: Option<Vec<(usize, usize)>> = {
        if rec.compression().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Compressed IPC requires the 'zstd' feature",
            ));
        }
        None
    };
    let corrections = corrections_storage.as_deref();

    let n_rows = if !nodes.is_empty() {
        let raw = nodes.get(0).length();
        if raw < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "negative record batch row count",
            ));
        }
        raw as usize
    } else {
        0
    };
    limits.check(n_rows, limits.max_n_rows, "record batch rows")?;

    let projected_count = projection.map_or(fields.len(), |p| p.len());
    let mut buffer_idx = 0;
    let mut cols: Vec<FieldArray> = Vec::with_capacity(projected_count);

    for (col_idx, field) in fields.iter().enumerate() {
        let node = nodes.get(col_idx);
        let row_count = node.length() as usize;
        let null_count = node.null_count() as usize;

        if row_count != n_rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Row count mismatch for {}", field.name),
            ));
        }

        // Extract null mask from the shared buffer.
        // This always runs even for skipped columns to keep buffer_idx in sync.
        let null_mask = extract_null_mask(
            field,
            row_count,
            null_count,
            &buffers,
            &mut buffer_idx,
            &shared,
            body_start,
            body_len,
            corrections,
        )?;

        // Skip non-projected columns by advancing past their data buffer
        // descriptors without materialising any arrays or SharedBuffer slices.
        let projected = projection.is_none_or(|p| p.contains(&col_idx));
        if !projected {
            buffer_idx += data_buffer_count(&field.dtype);
            continue;
        }

        let array = match &field.dtype {
            ArrowType::Int32
            | ArrowType::Int64
            | ArrowType::UInt32
            | ArrowType::UInt64
            | ArrowType::Float32
            | ArrowType::Float64 => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let data = shared.slice(off..off + len);
                make_numeric_array(&field.dtype, data, null_mask)?
            }

            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 | ArrowType::UInt8 | ArrowType::Int16 | ArrowType::UInt16 => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let data = shared.slice(off..off + len);
                make_numeric_array(&field.dtype, data, null_mask)?
            }

            ArrowType::Boolean => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                // Bitmask::from_bytes carries n_rows, so new() sets the
                // array length to match.
                let bits = Bitmask::from_bytes(shared.slice(off..off + len).as_slice(), n_rows);
                Array::BooleanArray(Arc::new(minarrow::BooleanArray::new(bits, null_mask)))
            }

            ArrowType::String => {
                let (offs_off, offs_len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let (data_off, data_len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let offs_buf = shared.slice(offs_off..offs_off + offs_len);
                let data_buf = shared.slice(data_off..data_off + data_len);

                Array::TextArray(TextArray::String32(Arc::new(StringArray::new(
                    minarrow::Buffer::from_shared(data_buf),
                    null_mask,
                    minarrow::Buffer::from_shared(offs_buf),
                ))))
            }

            #[cfg(feature = "large_string")]
            ArrowType::LargeString => {
                let (offs_off, offs_len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let (data_off, data_len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                Array::TextArray(TextArray::String64(Arc::new(StringArray::new(
                    minarrow::Buffer::from_shared(shared.slice(data_off..data_off + data_len)),
                    null_mask,
                    minarrow::Buffer::from_shared(shared.slice(offs_off..offs_off + offs_len)),
                ))))
            }

            #[cfg(feature = "datetime")]
            ArrowType::DateTime(
                Dt::Date32
                | Dt::Date64
                | Dt::Timestamp(_, _)
                | Dt::Time32(_)
                | Dt::Time64(_)
                | Dt::Duration64(_),
            ) => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let data = shared.slice(off..off + len);
                make_numeric_array(&field.dtype, data, null_mask)?
            }

            ArrowType::Dictionary(_idx_ty) => {
                let dict_key = col_idx as i64;
                let dict_values = dicts.get(&dict_key).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Dictionary for column '{}' (col_idx={}) missing",
                            field.name, col_idx
                        ),
                    )
                })?;
                let (idx_off, idx_len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                make_categorical_array(
                    _idx_ty,
                    shared.slice(idx_off..idx_off + idx_len),
                    dict_values,
                    null_mask,
                )?
            }

            #[cfg(feature = "decimal")]
            ArrowType::Decimal32(p, s) => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let data = shared.slice(off..off + len);
                Array::NumericArray(NumericArray::Decimal32(Arc::new(DecimalArray {
                    data: minarrow::Buffer::from_shared(data),
                    null_mask,
                    precision: *p,
                    scale: *s,
                })))
            }
            #[cfg(feature = "decimal")]
            ArrowType::Decimal64(p, s) => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let data = shared.slice(off..off + len);
                Array::NumericArray(NumericArray::Decimal64(Arc::new(DecimalArray {
                    data: minarrow::Buffer::from_shared(data),
                    null_mask,
                    precision: *p,
                    scale: *s,
                })))
            }
            #[cfg(feature = "decimal")]
            ArrowType::Decimal128(p, s) => {
                let (off, len) = consume_buffer(
                    &buffers,
                    &mut buffer_idx,
                    body_start,
                    body_len,
                    &field.name,
                    corrections,
                )?;
                let data = shared.slice(off..off + len);
                Array::NumericArray(NumericArray::Decimal128(Arc::new(DecimalArray {
                    data: minarrow::Buffer::from_shared(data),
                    null_mask,
                    precision: *p,
                    scale: *s,
                })))
            }

            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unsupported type in direct decoder: {:?}", other),
                ));
            }
        };

        cols.push(FieldArray::new(field.clone(), array));
    }

    let table = Table {
        cols,
        n_rows,
        name: "RecordBatch".to_string(),
        ..Default::default()
    };
    Ok((table, shared))
}

/// Number of data buffer descriptors consumed by a given Arrow type,
/// not counting the null mask buffer which is handled separately by
/// `extract_null_mask`.
///
/// This encodes the Arrow IPC wire format's per-type buffer layout:
/// - Fixed-width types (numerics, bool, date, dictionary indices): 1 data buffer
/// - Variable-length types (string, large string): 2 buffers (offsets + data)
fn data_buffer_count(dtype: &ArrowType) -> usize {
    match dtype {
        ArrowType::String => 2,
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => 2,
        _ => 1,
    }
}

/// Read a buffer descriptor from the IPC metadata and return the absolute
/// offset and length within the SharedBuffer.
///
/// When `corrections` is provided, the offset and length come from the
/// decompression corrections map rather than from the flatbuffer metadata.
/// This is used for compressed record batches where the metadata references
/// compressed positions.
fn consume_buffer(
    buffers: &Vector<'_, Buffer>,
    buffer_idx: &mut usize,
    body_start: usize,
    body_len: usize,
    field_name: &str,
    corrections: Option<&[(usize, usize)]>,
) -> io::Result<(usize, usize)> {
    let idx = *buffer_idx;
    *buffer_idx += 1;

    if let Some(c) = corrections {
        // Corrections come from our own decompression pass and are bounded
        // by the buffer descriptor count, but check anyway so a future
        // refactor cannot regress into an OOB index panic.
        let (off, len) = *c.get(idx).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("corrections index {} out of range for {}", idx, field_name),
            )
        })?;
        let end = off.checked_add(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "buffer offset+length overflow")
        })?;
        if end > body_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Decompressed buffer out of bounds for {}", field_name),
            ));
        }
        return Ok((body_start + off, len));
    }

    // Bound-check idx against the flatbuffer Vector before calling .get;
    // Vector::get on an OOB index panics in debug and is undefined in
    // release. Reject negative i64 offset/length so a hostile cast to
    // usize cannot become a huge value that wraps past the bounds check.
    if idx >= buffers.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "buffer index {} out of range for {} (have {})",
                idx,
                field_name,
                buffers.len()
            ),
        ));
    }
    let buf = buffers.get(idx);
    let offset_i = buf.offset();
    let length_i = buf.length();
    if offset_i < 0 || length_i < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "buffer descriptor has negative offset/length for {}",
                field_name
            ),
        ));
    }
    let offset = offset_i as usize;
    let length = length_i as usize;
    let end = offset.checked_add(length).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "buffer offset+length overflow")
    })?;
    if end > body_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("Buffer out of bounds for {}", field_name),
        ));
    }
    Ok((body_start + offset, length))
}

/// Extract null mask for the streaming decode path.
///
/// Creates a Bitmask backed by a SharedBuffer slice when a real bitmap is
/// present, avoiding any data copy.
///
/// When `corrections` is provided, buffer offset and length come from the
/// decompression corrections map instead of the flatbuffer metadata.
#[allow(clippy::too_many_arguments)]
fn extract_null_mask(
    field: &Field,
    field_len: usize,
    null_count: usize,
    fbuf_meta: &Vector<'_, Buffer>,
    buffer_idx: &mut usize,
    shared: &SharedBuffer,
    body_start: usize,
    body_len: usize,
    corrections: Option<&[(usize, usize)]>,
) -> io::Result<Option<Bitmask>> {
    if !field.nullable {
        // Peek the null-mask slot length without bounds-asserting Vector::get
        // when the slot is absent. A peer that omits the slot for a
        // non-nullable column simply does not advance the buffer index.
        let len_bytes = if let Some(c) = corrections {
            match c.get(*buffer_idx) {
                Some((_, l)) => *l,
                None => return Ok(None),
            }
        } else if *buffer_idx < fbuf_meta.len() {
            let l_i = fbuf_meta.get(*buffer_idx).length();
            if l_i < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("null mask length is negative for {}", field.name),
                ));
            }
            l_i as usize
        } else {
            return Ok(None);
        };
        let expected_null_mask_len = field_len.div_ceil(8);
        if len_bytes == 0 || len_bytes == expected_null_mask_len {
            *buffer_idx += 1;
        }
        return Ok(None);
    }

    let idx = *buffer_idx;
    *buffer_idx += 1;

    let (offset, len_bytes) = if let Some(c) = corrections {
        *c.get(idx).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("corrections index {} out of range for {}", idx, field.name),
            )
        })?
    } else {
        if idx >= fbuf_meta.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "null mask index {} out of range for {} (have {})",
                    idx,
                    field.name,
                    fbuf_meta.len()
                ),
            ));
        }
        let buf = fbuf_meta.get(idx);
        let off_i = buf.offset();
        let len_i = buf.length();
        if off_i < 0 || len_i < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("null mask descriptor negative for {}", field.name),
            ));
        }
        (off_i as usize, len_i as usize)
    };

    if null_count == 0 || len_bytes == 0 {
        return Ok(Some(Bitmask::new_set_all(field_len, true)));
    }

    let end = offset.checked_add(len_bytes).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "null mask offset+length overflow",
        )
    })?;
    if end > body_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("Null buffer out of bounds for {}", field.name),
        ));
    }

    let slice = shared.slice(body_start + offset..body_start + offset + len_bytes);
    let buffer: minarrow::Buffer<u8> = minarrow::Buffer::from_shared(slice);
    Ok(Some(Bitmask::new(buffer, field_len)))
}

/// Create a numeric Array from a SharedBuffer slice.
fn make_numeric_array(
    dtype: &ArrowType,
    data: SharedBuffer,
    null_mask: Option<Bitmask>,
) -> io::Result<Array> {
    let array = match dtype {
        ArrowType::Int32 => Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        ArrowType::Int64 => Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        ArrowType::UInt32 => Array::NumericArray(NumericArray::UInt32(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        ArrowType::UInt64 => Array::NumericArray(NumericArray::UInt64(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        ArrowType::Float32 => Array::NumericArray(NumericArray::Float32(Arc::new(FloatArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        ArrowType::Float64 => Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => Array::NumericArray(NumericArray::Int8(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => Array::NumericArray(NumericArray::UInt8(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => Array::NumericArray(NumericArray::Int16(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => Array::NumericArray(NumericArray::UInt16(Arc::new(IntegerArray {
            data: minarrow::Buffer::from_shared(data),
            null_mask,
        }))),
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(dt @ (Dt::Date32 | Dt::Time32(_))) => {
            Array::TemporalArray(TemporalArray::Datetime32(Arc::new(DatetimeArray::new(
                minarrow::Buffer::from_shared(data),
                null_mask,
                dt.clone(),
            ))))
        }
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(
            dt @ (Dt::Date64 | Dt::Timestamp(_, _) | Dt::Time64(_) | Dt::Duration64(_)),
        ) => Array::TemporalArray(TemporalArray::Datetime64(Arc::new(DatetimeArray::new(
            minarrow::Buffer::from_shared(data),
            null_mask,
            dt.clone(),
        )))),
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported numeric type: {:?}", other),
            ));
        }
    };
    Ok(array)
}

/// Create a categorical Array from a SharedBuffer slice.
fn make_categorical_array(
    idx_ty: &CategoricalIndexType,
    idx_data: SharedBuffer,
    dict_values: &[String],
    null_mask: Option<Bitmask>,
) -> io::Result<Array> {
    let unique_values = Vec64::from(dict_values.to_vec());
    let array = match idx_ty {
        CategoricalIndexType::UInt32 => {
            Array::TextArray(TextArray::Categorical32(Arc::new(CategoricalArray::new(
                minarrow::Buffer::from_shared(idx_data),
                unique_values,
                null_mask,
            ))))
        }
        #[cfg(feature = "default_categorical_8")]
        CategoricalIndexType::UInt8 => {
            Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray::new(
                minarrow::Buffer::from_shared(idx_data),
                unique_values,
                null_mask,
            ))))
        }
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt16 => {
            Array::TextArray(TextArray::Categorical16(Arc::new(CategoricalArray::new(
                minarrow::Buffer::from_shared(idx_data),
                unique_values,
                null_mask,
            ))))
        }
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt64 => {
            Array::TextArray(TextArray::Categorical64(Arc::new(CategoricalArray::new(
                minarrow::Buffer::from_shared(idx_data),
                unique_values,
                null_mask,
            ))))
        }
        #[allow(unreachable_patterns)]
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported categorical index type: {:?}", idx_ty),
            ));
        }
    };
    Ok(array)
}

/// Extracts all Arrow fields from a FlatBuffers Arrow schema message.
///
/// Converts FlatBuffers fields to native [`Field`] representations.
#[inline(always)]
pub fn handle_schema_header(af_msg: &fb::Message, limits: DecodeLimits) -> io::Result<Vec<Field>> {
    // 1. Validate Flatbuffer version
    let version = af_msg.version();
    match version {
        fb::MetadataVersion::V1
        | fb::MetadataVersion::V2
        | fb::MetadataVersion::V3
        | fb::MetadataVersion::V4
        | fb::MetadataVersion::V5 => { /* Supported */ }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Arrow IPC: unsupported Flatbuffer metadata version {:?}",
                    version
                ),
            ));
        }
    }

    // 2. Parse and validate Schema header
    let schema = af_msg
        .header_as_schema()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "schema missing"))?;

    // 3. Enforce endianness = little endian
    let endianness = schema.endianness();
    if endianness != fb::Endianness::Little {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Arrow IPC: unsupported endianness {:?} - only Little Endian is supported",
                endianness
            ),
        ));
    }

    // 4. Extract fields. Cap the declared field count before allocating;
    // a peer can otherwise declare arbitrary fb_fields.len() and trigger
    // a Vec::with_capacity sized off untrusted input.
    let fb_fields = schema
        .fields()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no fields"))?;
    limits.check(fb_fields.len(), limits.max_fields, "schema fields")?;
    let mut fields = Vec::with_capacity(fb_fields.len());
    for i in 0..fb_fields.len() {
        fields.push(convert_flatbuffers_to_arrow_field(&fb_fields.get(i))?);
    }
    Ok(fields)
}

/// Converts a FlatBuffers Arrow Field definition to an Arrow [`Field`] struct.
///
/// Extracts name, nullability, user-defined metadata, and data type information.
#[inline]
pub fn convert_flatbuffers_to_arrow_field(fb_field: &fb::Field) -> io::Result<Field> {
    let name = fb_field
        .name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing field name"))?
        .to_string();

    let nullable = fb_field.nullable();
    let metadata = extract_metadata(fb_field.custom_metadata());
    let base_type = extract_base_type(fb_field)?;
    let dtype = extract_dtype(fb_field, base_type)?;

    Ok(Field {
        name,
        dtype,
        nullable,
        metadata,
    })
}

/// Extracts user-defined key-value metadata from a FlatBuffers Arrow Field.
///
/// Converts FlatBuffers custom metadata into a key-value [`BTreeMap`].
fn extract_metadata<'a>(
    meta_vec: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::KeyValue<'a>>>>,
) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    if let Some(vec) = meta_vec {
        for i in 0..vec.len() {
            let k = vec.get(i).key().unwrap_or("").to_string();
            let v = vec.get(i).value().unwrap_or("").to_string();
            map.insert(k, v);
        }
    }
    map
}

/// Determines the Arrow categorical index type from FlatBuffers dictionary index metadata.
///
/// Supports UInt32 by default, additional types under `extended_categorical`.
fn extract_categorical_index_type(
    index_type: Option<&fb::Int>,
) -> io::Result<CategoricalIndexType> {
    let idx_type = index_type.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing dictionary index type")
    })?;
    match idx_type.bitWidth() {
        32 => Ok(CategoricalIndexType::UInt32),
        #[cfg(feature = "default_categorical_8")]
        8 => Ok(CategoricalIndexType::UInt8),
        #[cfg(feature = "extended_categorical")]
        64 => Ok(CategoricalIndexType::UInt64),
        #[cfg(feature = "extended_categorical")]
        16 => Ok(CategoricalIndexType::UInt16),
        w => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported dict index width {w}"),
        )),
    }
}

/// Convert FlatBuffer DateUnit to ArrowType for the stream message schema.
#[cfg(feature = "datetime")]
fn convert_date_unit_fb(unit: fb::DateUnit) -> io::Result<ArrowType> {
    match unit {
        fb::DateUnit::DAY => Ok(ArrowType::DateTime(Dt::Date32)),
        fb::DateUnit::MILLISECOND => Ok(ArrowType::DateTime(Dt::Date64)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported Date unit {:?}", unit),
        )),
    }
}

/// Convert the stream message schema's FlatBuffer TimeUnit to Minarrow's.
#[cfg(feature = "datetime")]
fn convert_time_unit_fb(unit: fb::TimeUnit) -> io::Result<MnTimeUnit> {
    match unit {
        fb::TimeUnit::SECOND => Ok(MnTimeUnit::Seconds),
        fb::TimeUnit::MILLISECOND => Ok(MnTimeUnit::Milliseconds),
        fb::TimeUnit::MICROSECOND => Ok(MnTimeUnit::Microseconds),
        fb::TimeUnit::NANOSECOND => Ok(MnTimeUnit::Nanoseconds),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported time unit {unit:?}"),
        )),
    }
}

/// Convert the file footer schema's FlatBuffer TimeUnit to Minarrow's.
#[cfg(feature = "datetime")]
fn convert_time_unit_fbf(
    unit: crate::arrow::file::org::apache::arrow::flatbuf::TimeUnit,
) -> io::Result<MnTimeUnit> {
    use crate::arrow::file::org::apache::arrow::flatbuf::TimeUnit;
    match unit {
        TimeUnit::SECOND => Ok(MnTimeUnit::Seconds),
        TimeUnit::MILLISECOND => Ok(MnTimeUnit::Milliseconds),
        TimeUnit::MICROSECOND => Ok(MnTimeUnit::Microseconds),
        TimeUnit::NANOSECOND => Ok(MnTimeUnit::Nanoseconds),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported time unit {unit:?}"),
        )),
    }
}

/// Convert FlatBuffer DateUnit to ArrowType for the file footer schema.
#[cfg(feature = "datetime")]
fn convert_date_unit_fbf(
    unit: crate::arrow::file::org::apache::arrow::flatbuf::DateUnit,
) -> io::Result<ArrowType> {
    use crate::arrow::file::org::apache::arrow::flatbuf::DateUnit;
    match unit {
        DateUnit::DAY => Ok(ArrowType::DateTime(Dt::Date32)),
        DateUnit::MILLISECOND => Ok(ArrowType::DateTime(Dt::Date64)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported Date unit {:?}", unit),
        )),
    }
}

/// Deduces the logical Arrow type from the FlatBuffers field variant.
///
/// Maps Arrow physical types, dictionaries, and text columns.
fn extract_base_type(fb_field: &fb::Field) -> io::Result<ArrowType> {
    match fb_field.type_type() {
        fb::Type::Int => {
            let i = fb_field
                .type__as_int()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Int type"))?;
            match (i.bitWidth(), i.is_signed()) {
                #[cfg(feature = "extended_numeric_types")]
                (8, true) => Ok(ArrowType::Int8),
                #[cfg(feature = "extended_numeric_types")]
                (8, false) => Ok(ArrowType::UInt8),
                #[cfg(feature = "extended_numeric_types")]
                (16, true) => Ok(ArrowType::Int16),
                #[cfg(feature = "extended_numeric_types")]
                (16, false) => Ok(ArrowType::UInt16),
                (32, true) => Ok(ArrowType::Int32),
                (64, true) => Ok(ArrowType::Int64),
                (32, false) => Ok(ArrowType::UInt32),
                (64, false) => Ok(ArrowType::UInt64),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported int width",
                )),
            }
        }
        fb::Type::Utf8 => Ok(ArrowType::String),
        #[cfg(feature = "large_string")]
        fb::Type::LargeUtf8 => Ok(ArrowType::LargeString),
        fb::Type::FloatingPoint => {
            let f = fb_field.type__as_floating_point().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing FloatingPoint type")
            })?;
            match f.precision() {
                fb::Precision::SINGLE => Ok(ArrowType::Float32),
                fb::Precision::DOUBLE => Ok(ArrowType::Float64),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported float precision",
                )),
            }
        }
        #[cfg(feature = "datetime")]
        fb::Type::Date => {
            let d = fb_field
                .type__as_date()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Date type"))?;
            convert_date_unit_fb(d.unit())
        }
        #[cfg(feature = "datetime")]
        fb::Type::Timestamp => {
            let t = fb_field.type__as_timestamp().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing Timestamp type")
            })?;
            Ok(ArrowType::DateTime(Dt::Timestamp(
                convert_time_unit_fb(t.unit())?,
                t.timezone().map(|s| s.to_string()),
            )))
        }
        #[cfg(feature = "datetime")]
        fb::Type::Time => {
            let t = fb_field
                .type__as_time()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Time type"))?;
            let unit = convert_time_unit_fb(t.unit())?;
            match t.bitWidth() {
                32 => Ok(ArrowType::DateTime(Dt::Time32(unit))),
                64 => Ok(ArrowType::DateTime(Dt::Time64(unit))),
                w => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported Time bit width {w}"),
                )),
            }
        }
        #[cfg(feature = "datetime")]
        fb::Type::Duration => {
            let d = fb_field.type__as_duration().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing Duration type")
            })?;
            Ok(ArrowType::DateTime(Dt::Duration64(convert_time_unit_fb(d.unit())?)))
        }
        fb::Type::Bool => Ok(ArrowType::Boolean),
        #[cfg(feature = "decimal")]
        fb::Type::Decimal => {
            let d = fb_field.type__as_decimal().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing Decimal type")
            })?;
            let precision = d.precision() as u8;
            let scale = d.scale() as i8;
            match d.bitWidth() {
                32 => Ok(ArrowType::Decimal32(precision, scale)),
                64 => Ok(ArrowType::Decimal64(precision, scale)),
                128 => Ok(ArrowType::Decimal128(precision, scale)),
                w => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported Decimal bit width {w}"),
                )),
            }
        }
        other => {
            if let Some(dict) = fb_field.dictionary() {
                let idx_ty = extract_categorical_index_type(dict.indexType().as_ref())?;
                Ok(ArrowType::Dictionary(idx_ty))
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported fb type {other:?}"),
                ))
            }
        }
    }
}

/// Deduces the Arrow data type (possibly dictionary-wrapped) for an Arrow field.
///
/// Returns wrapped type if dictionary encoding present, otherwise raw type.
fn extract_dtype(fb_field: &fb::Field, base_type: ArrowType) -> io::Result<ArrowType> {
    if let Some(dict) = fb_field.dictionary() {
        // Dictionary values are strings with u32 offsets, so the field must
        // declare Utf8 as its value type.
        if fb_field.type_type() != fb::Type::Utf8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported dictionary value type {:?}",
                    fb_field.type_type()
                ),
            ));
        }
        let idx_ty = extract_categorical_index_type(dict.indexType().as_ref())?;
        Ok(ArrowType::Dictionary(idx_ty))
    } else {
        Ok(base_type)
    }
}

/// Converts a Flatbuffers `Field` to the `Minarrow` version
pub fn convert_fb_field_to_arrow(
    fbf_field: &crate::arrow::file::org::apache::arrow::flatbuf::Field,
) -> io::Result<Field> {
    use crate::arrow::file::org::apache::arrow::flatbuf as fbf;
    // name
    let name = fbf_field
        .name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing field name"))?
        .to_string();

    // nullable
    let nullable = fbf_field.nullable();

    // user metadata
    let metadata = {
        let mut map = std::collections::BTreeMap::<String, String>::new();
        if let Some(vec) = fbf_field.custom_metadata() {
            for i in 0..vec.len() {
                let kv = vec.get(i);
                map.insert(
                    kv.key().unwrap_or("").to_owned(),
                    kv.value().unwrap_or("").to_owned(),
                );
            }
        }
        map
    };

    // Check for dictionary encoding first, regardless of the underlying type
    let base_type = if let Some(dict) = fbf_field.dictionary() {
        use minarrow::ffi::arrow_dtype::CategoricalIndexType as Idx;
        // Dictionary values are strings with u32 offsets, so the field must
        // declare Utf8 as its value type.
        if fbf_field.type_type() != fbf::Type::Utf8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported dictionary value type {:?}",
                    fbf_field.type_type()
                ),
            ));
        }
        let idx = dict
            .indexType()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing dict idx"))?;
        let idx_ty = match idx.bitWidth() {
            32 => Idx::UInt32,
            #[cfg(feature = "default_categorical_8")]
            8 => Idx::UInt8,
            #[cfg(feature = "extended_categorical")]
            16 => Idx::UInt16,
            #[cfg(feature = "extended_categorical")]
            64 => Idx::UInt64,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bad dict idx width",
                ));
            }
        };
        ArrowType::Dictionary(idx_ty)
    } else {
        // base dtype (non-dictionary)
        match fbf_field.type_type() {
            fbf::Type::Int => {
                // A peer that declares type_type = Int without a matching Int
                // union table reaches this branch; previously this panicked.
                let i = fbf_field.type__as_int().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "field type tagged Int but Int table missing",
                    )
                })?;
                match (i.bitWidth(), i.is_signed()) {
                    #[cfg(feature = "extended_numeric_types")]
                    (8, true) => ArrowType::Int8,
                    #[cfg(feature = "extended_numeric_types")]
                    (8, false) => ArrowType::UInt8,
                    #[cfg(feature = "extended_numeric_types")]
                    (16, true) => ArrowType::Int16,
                    #[cfg(feature = "extended_numeric_types")]
                    (16, false) => ArrowType::UInt16,
                    (32, true) => ArrowType::Int32,
                    (64, true) => ArrowType::Int64,
                    (32, false) => ArrowType::UInt32,
                    (64, false) => ArrowType::UInt64,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unsupported int width",
                        ));
                    }
                }
            }
            fbf::Type::FloatingPoint => {
                // Same shape as the Int branch above: refuse a mismatched
                // type tag rather than panicking.
                let f = fbf_field.type__as_floating_point().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "field type tagged FloatingPoint but FloatingPoint table missing",
                    )
                })?;
                match f.precision() {
                    fbf::Precision::SINGLE => ArrowType::Float32,
                    fbf::Precision::DOUBLE => ArrowType::Float64,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unsupported float prec",
                        ));
                    }
                }
            }
            fbf::Type::Utf8 => ArrowType::String,
            #[cfg(feature = "large_string")]
            fbf::Type::LargeUtf8 => ArrowType::LargeString,
            fbf::Type::Bool => ArrowType::Boolean,
            #[cfg(feature = "datetime")]
            fbf::Type::Date => {
                let d = fbf_field.type__as_date().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing Date type")
                })?;
                convert_date_unit_fbf(d.unit())?
            }
            #[cfg(feature = "datetime")]
            fbf::Type::Timestamp => {
                let t = fbf_field.type__as_timestamp().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing Timestamp type")
                })?;
                ArrowType::DateTime(Dt::Timestamp(
                    convert_time_unit_fbf(t.unit())?,
                    t.timezone().map(|s| s.to_string()),
                ))
            }
            #[cfg(feature = "datetime")]
            fbf::Type::Time => {
                let t = fbf_field.type__as_time().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing Time type")
                })?;
                let unit = convert_time_unit_fbf(t.unit())?;
                match t.bitWidth() {
                    32 => ArrowType::DateTime(Dt::Time32(unit)),
                    64 => ArrowType::DateTime(Dt::Time64(unit)),
                    w => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unsupported Time bit width {w}"),
                        ));
                    }
                }
            }
            #[cfg(feature = "datetime")]
            fbf::Type::Duration => {
                let d = fbf_field.type__as_duration().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing Duration type")
                })?;
                ArrowType::DateTime(Dt::Duration64(convert_time_unit_fbf(d.unit())?))
            }
            #[cfg(feature = "decimal")]
            fbf::Type::Decimal => {
                let d = fbf_field.type__as_decimal().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing Decimal type")
                })?;
                let precision = d.precision() as u8;
                let scale = d.scale() as i8;
                match d.bitWidth() {
                    32 => ArrowType::Decimal32(precision, scale),
                    64 => ArrowType::Decimal64(precision, scale),
                    128 => ArrowType::Decimal128(precision, scale),
                    w => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unsupported Decimal bit width {w}"),
                        ));
                    }
                }
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported fb type {other:?}"),
                ));
            }
        }
    };

    Ok(Field {
        name,
        dtype: base_type,
        nullable,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_dictionary_strings tests --------------------------------------

    #[test]
    fn test_parse_dictionary_strings_basic() {
        // Three strings: "red", "green", "blue"
        // Offsets: [0, 3, 8, 12]
        let data = b"redgreenblue";

        let offsets: Vec<u8> = [0u32, 3, 8, 12].iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = parse_dictionary_strings(&offsets, data, DecodeLimits::default()).unwrap();
        assert_eq!(result.as_slice(), &["red", "green", "blue"]);
    }

    #[test]
    fn test_parse_dictionary_strings_single() {
        let data = b"hello";

        let offsets: Vec<u8> = [0u32, 5].iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = parse_dictionary_strings(&offsets, data, DecodeLimits::default()).unwrap();
        assert_eq!(result.as_slice(), &["hello"]);
    }

    #[test]
    fn test_parse_dictionary_strings_empty_strings() {
        // Two empty strings: offsets [0, 0, 0]
        let data = b"";

        let offsets: Vec<u8> = [0u32, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = parse_dictionary_strings(&offsets, data, DecodeLimits::default()).unwrap();
        assert_eq!(result.as_slice(), &["", ""]);
    }

    #[test]
    fn test_parse_dictionary_strings_out_of_bounds() {
        let data = b"abc";

        let offsets: Vec<u8> = [0u32, 99].iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = parse_dictionary_strings(&offsets, data, DecodeLimits::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dictionary_strings_too_few_offsets() {
        // Only one offset - need at least 2
        let offsets = 0u32.to_le_bytes().to_vec();

        let result = parse_dictionary_strings(&offsets, b"", DecodeLimits::default());
        assert!(result.is_err());
    }

    // -- handle_dictionary_batch delta tests ----------------------------------
    // The encoder does not emit delta dictionaries, so we test the handler
    // directly by building flatbuffer DictionaryBatch payloads.

    use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
    use flatbuffers::FlatBufferBuilder;

    /// Build a minimal DictionaryBatch flatbuffer and its body bytes.
    fn build_dict_batch(id: i64, is_delta: bool, strings: &[&str]) -> (Vec<u8>, Vec<u8>) {
        // Build body: validity (empty) + offsets + string data
        let concat: String = strings.iter().copied().collect();
        let data_bytes = concat.as_bytes();

        let offset_bytes: Vec<u8> = {
            let mut offs = Vec::with_capacity(strings.len() + 1);
            let mut pos = 0u32;
            offs.push(pos);
            for s in strings {
                pos += s.len() as u32;
                offs.push(pos);
            }
            offs.iter().flat_map(|v| v.to_le_bytes()).collect()
        };

        // Body layout: [validity(0 bytes)] [offsets] [data]
        let validity_len = 0usize;
        let offsets_start = validity_len;
        let offsets_len = offset_bytes.len();
        let data_start = offsets_start + offsets_len;
        let data_len = data_bytes.len();

        let mut body = vec![0u8; data_start + data_len];
        body[offsets_start..offsets_start + offsets_len].copy_from_slice(&offset_bytes);
        body[data_start..data_start + data_len].copy_from_slice(data_bytes);

        // Build flatbuffer
        let mut fbb = FlatBufferBuilder::new();

        // RecordBatch inside the DictionaryBatch
        let nodes = fbb.create_vector(&[fbm::FieldNode::new(strings.len() as i64, 0)]);
        let buffers = fbb.create_vector(&[
            fbm::Buffer::new(0, 0), // validity
            fbm::Buffer::new(offsets_start as i64, offsets_len as i64),
            fbm::Buffer::new(data_start as i64, data_len as i64),
        ]);
        let rec = fbm::RecordBatch::create(
            &mut fbb,
            &fbm::RecordBatchArgs {
                length: strings.len() as i64,
                nodes: Some(nodes),
                buffers: Some(buffers),
                compression: None,
                variadicBufferCounts: None,
            },
        );

        let dict = fbm::DictionaryBatch::create(
            &mut fbb,
            &fbm::DictionaryBatchArgs {
                id,
                data: Some(rec),
                isDelta: is_delta,
            },
        );

        fbb.finish(dict, None);
        let fb_bytes = fbb.finished_data().to_vec();

        (fb_bytes, body)
    }

    #[test]
    fn test_dictionary_delta_appends() {
        let mut dicts = HashMap::new();

        // Base dictionary
        let (fb_base, body_base) = build_dict_batch(0, false, &["red", "green", "blue"]);
        let base = flatbuffers::root::<fbm::DictionaryBatch>(&fb_base).unwrap();
        handle_dictionary_batch(&base, &body_base, &mut dicts, DecodeLimits::default()).unwrap();
        assert_eq!(dicts[&0], vec!["red", "green", "blue"]);

        // Delta appends new values
        let (fb_delta, body_delta) = build_dict_batch(0, true, &["yellow", "purple"]);
        let delta = flatbuffers::root::<fbm::DictionaryBatch>(&fb_delta).unwrap();
        handle_dictionary_batch(&delta, &body_delta, &mut dicts, DecodeLimits::default()).unwrap();
        assert_eq!(dicts[&0], vec!["red", "green", "blue", "yellow", "purple"]);
    }

    #[test]
    fn test_dictionary_delta_without_base_errors() {
        let mut dicts = HashMap::new();

        let (fb_delta, body_delta) = build_dict_batch(0, true, &["orphan"]);
        let delta = flatbuffers::root::<fbm::DictionaryBatch>(&fb_delta).unwrap();
        let result =
            handle_dictionary_batch(&delta, &body_delta, &mut dicts, DecodeLimits::default());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("before base dictionary")
        );
    }

    #[test]
    fn test_dictionary_base_replaces() {
        let mut dicts = HashMap::new();

        // First base
        let (fb1, body1) = build_dict_batch(0, false, &["a", "b"]);
        let batch1 = flatbuffers::root::<fbm::DictionaryBatch>(&fb1).unwrap();
        handle_dictionary_batch(&batch1, &body1, &mut dicts, DecodeLimits::default()).unwrap();
        assert_eq!(dicts[&0], vec!["a", "b"]);

        // Second base replaces
        let (fb2, body2) = build_dict_batch(0, false, &["x", "y", "z"]);
        let batch2 = flatbuffers::root::<fbm::DictionaryBatch>(&fb2).unwrap();
        handle_dictionary_batch(&batch2, &body2, &mut dicts, DecodeLimits::default()).unwrap();
        assert_eq!(dicts[&0], vec!["x", "y", "z"]);
    }

    // -- DecodeLimits enforcement ---------------------------------------------

    #[test]
    fn decode_record_batch_rejects_length_beyond_max_n_rows() {
        // Hand-build a flatbuffers RecordBatch claiming length = i64::MAX.
        // The decoder must refuse it under the default DecodeLimits before
        // any allocation that scales with the declared row count.
        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let node = fbm::FieldNode::new(i64::MAX, 0);
        let nodes_vec = fbb.create_vector(&[node]);
        let buf = fbm::Buffer::new(0, 0);
        let buffers_vec = fbb.create_vector(&[buf]);
        let rb = fbm::RecordBatch::create(
            &mut fbb,
            &fbm::RecordBatchArgs {
                length: i64::MAX,
                nodes: Some(nodes_vec),
                buffers: Some(buffers_vec),
                compression: None,
                variadicBufferCounts: None,
            },
        );
        fbb.finish(rb, None);
        let rb_bytes = fbb.finished_data().to_vec();
        let rec = flatbuffers::root::<fbm::RecordBatch>(&rb_bytes).unwrap();

        let fields = vec![Field {
            name: "x".into(),
            dtype: ArrowType::Int32,
            nullable: false,
            metadata: Default::default(),
        }];
        let dicts: HashMap<i64, Vec<String>> = HashMap::new();
        let shared = SharedBuffer::from_vec(Vec::new());

        let err = decode_record_batch(
            &rec,
            &fields,
            &dicts,
            shared,
            0,
            0,
            None,
            DecodeLimits::default(),
        )
        .err()
        .expect("decode_record_batch must refuse n_rows beyond max_n_rows");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("record batch rows"));
    }

    #[test]
    fn parse_dictionary_strings_rejects_u32_count_below_two() {
        // The u32-offset branch needs at least two offsets (one + sentinel).
        // A peer that hands us a single u32 offset would otherwise underflow
        // `count - 1` to usize::MAX.
        let offsets = [0u8; 4];
        let data = b"abc";
        let err = parse_dictionary_strings(&offsets, data, DecodeLimits::default())
            .err()
            .expect("count = 1 in the u32 path must surface as Err");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("offset count < 2"));
    }

    fn build_minimal_record_batch_bytes(node_length: i64, buffers: &[fbm::Buffer]) -> Vec<u8> {
        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let node = fbm::FieldNode::new(node_length, 0);
        let nodes_vec = fbb.create_vector(&[node]);
        let buffers_vec = fbb.create_vector(buffers);
        let rb = fbm::RecordBatch::create(
            &mut fbb,
            &fbm::RecordBatchArgs {
                length: node_length,
                nodes: Some(nodes_vec),
                buffers: Some(buffers_vec),
                compression: None,
                variadicBufferCounts: None,
            },
        );
        fbb.finish(rb, None);
        fbb.finished_data().to_vec()
    }

    fn single_int32_field() -> Vec<Field> {
        vec![Field {
            name: "x".into(),
            dtype: ArrowType::Int32,
            nullable: false,
            metadata: Default::default(),
        }]
    }

    #[test]
    fn decode_record_batch_rejects_negative_buffer_offset() {
        // Non-nullable Int32, two buffer descriptors: an empty validity
        // slot at idx 0 and a data slot at idx 1 with a negative offset.
        // The cast-to-usize wraparound classic, caught at descriptor read.
        let validity = fbm::Buffer::new(0, 0);
        let bad_data = fbm::Buffer::new(-1, 0);
        let bytes = build_minimal_record_batch_bytes(0, &[validity, bad_data]);
        let rec = flatbuffers::root::<fbm::RecordBatch>(&bytes).unwrap();
        let fields = single_int32_field();
        let dicts: HashMap<i64, Vec<String>> = HashMap::new();
        let shared = SharedBuffer::from_vec(Vec::new());

        let err = decode_record_batch(
            &rec,
            &fields,
            &dicts,
            shared,
            0,
            0,
            None,
            DecodeLimits::default(),
        )
        .err()
        .expect("negative buffer offset must surface as Err");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("negative"));
    }

    #[test]
    fn decode_record_batch_rejects_buffer_index_overrun() {
        // One Int32 field requires one data buffer descriptor; supply
        // zero so the column decoder runs off the end of the Vector.
        let bytes = build_minimal_record_batch_bytes(0, &[]);
        let rec = flatbuffers::root::<fbm::RecordBatch>(&bytes).unwrap();
        let fields = single_int32_field();
        let dicts: HashMap<i64, Vec<String>> = HashMap::new();
        let shared = SharedBuffer::from_vec(Vec::new());

        let err = decode_record_batch(
            &rec,
            &fields,
            &dicts,
            shared,
            0,
            0,
            None,
            DecodeLimits::default(),
        )
        .err()
        .expect("missing buffer descriptor must surface as Err, not a Vector::get panic");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("out of range"));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn decompress_ipc_body_rejects_bomb_prefix() {
        // Build a single-buffer RecordBatch with a ZSTD-compressed body where
        // the per-buffer 8-byte uncompressed_len prefix claims i64::MAX bytes.
        // The cap inside decompress_ipc_body must fire before any allocation
        // sized off that claim, and well before the decompressor is invoked.
        use crate::compression::ipc::decompress_ipc_body;

        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let nodes_vec = fbb.create_vector(&[fbm::FieldNode::new(0, 0)]);
        // One buffer descriptor covering 9 body bytes: 8 prefix + 1 payload.
        let buffers_vec = fbb.create_vector(&[fbm::Buffer::new(0, 9)]);
        let body_compression = fbm::BodyCompression::create(
            &mut fbb,
            &fbm::BodyCompressionArgs {
                codec: fbm::CompressionType::ZSTD,
                method: fbm::BodyCompressionMethod::BUFFER,
            },
        );
        let rb = fbm::RecordBatch::create(
            &mut fbb,
            &fbm::RecordBatchArgs {
                length: 0,
                nodes: Some(nodes_vec),
                buffers: Some(buffers_vec),
                compression: Some(body_compression),
                variadicBufferCounts: None,
            },
        );
        fbb.finish(rb, None);
        let rb_bytes = fbb.finished_data().to_vec();
        let rec = flatbuffers::root::<fbm::RecordBatch>(&rb_bytes).unwrap();

        // The malicious body: prefix claims i64::MAX uncompressed bytes,
        // followed by a single payload byte so length checks all pass.
        let mut body = Vec::with_capacity(9);
        body.extend_from_slice(&i64::MAX.to_le_bytes());
        body.push(0u8);

        let buffers = rec.buffers().unwrap();
        let compression = rec.compression().unwrap();
        let start = std::time::Instant::now();
        let err = decompress_ipc_body::<Vec64<u8>>(
            &body,
            &buffers,
            &compression,
            DecodeLimits::default(),
        )
        .err()
        .expect("bomb prefix must surface as Err");
        let elapsed = start.elapsed();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("decompressed buffer length"));
        // The cap must fire on the prefix - no decompression attempted, no
        // allocation sized off the claim.
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "decompression cap should fire promptly, took {:?}",
            elapsed
        );
    }

    #[test]
    fn convert_fb_field_rejects_int_tag_without_table() {
        // Build a file-format Field with type_type = Int but no Int union
        // table attached; the mismatched tag must surface as Err rather
        // than aborting the process. flatbuffers::root verifies the union
        // discriminant against the table, so we use the unchecked entry to
        // exercise the conversion path itself.
        use crate::arrow::file::org::apache::arrow::flatbuf as fbf;
        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let name = fbb.create_string("x");
        let field = fbf::Field::create(
            &mut fbb,
            &fbf::FieldArgs {
                name: Some(name),
                nullable: false,
                type_type: fbf::Type::Int,
                type_: None,
                dictionary: None,
                children: None,
                custom_metadata: None,
            },
        );
        fbb.finish(field, None);
        let bytes = fbb.finished_data().to_vec();
        // SAFETY: bytes were just produced by FlatBufferBuilder; only the
        // union discriminant/table consistency is intentionally violated.
        let fb_field = unsafe { flatbuffers::root_unchecked::<fbf::Field>(&bytes) };

        let err = convert_fb_field_to_arrow(&fb_field)
            .err()
            .expect("missing Int union table must surface as Err, not a panic");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("Int table missing"));
    }
}
