// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Row windowing over decoded Arrow IPC record batches.
//!
//! Shared by the file and mmap table readers. A window is a standalone
//! `Table` whose column buffers view the parent batch through its shared
//! owner, so cutting a batch into row windows bumps reference counts
//! rather than copying data.

use std::io;
use std::sync::Arc;

use minarrow::{
    Array, Bitmask, BooleanArray, Buffer, CategoricalArray, FieldArray, FloatArray, Integer,
    IntegerArray, MaskedArray, NumericArray, StringArray, Table, TextArray, Vec64,
};
#[cfg(feature = "decimal")]
use minarrow::DecimalArray;
#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, TemporalArray};

/// Build a standalone table for the row window `[offset, offset + len)`
/// of a decoded batch. A window covering the whole table returns a
/// clone, which bumps the columns' reference counts without touching
/// data.
pub(crate) fn window_table(table: &Table, offset: usize, len: usize) -> io::Result<Table> {
    if offset == 0 && len == table.n_rows {
        return Ok(table.clone());
    }
    if offset % 512 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("window start {offset} is not a multiple of 512 rows"),
        ));
    }
    if offset + len > table.n_rows {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "window {offset}..{} exceeds the table's {} rows",
                offset + len,
                table.n_rows
            ),
        ));
    }
    let mut cols = Vec::with_capacity(table.cols.len());
    for col in &table.cols {
        cols.push(FieldArray {
            field: col.field.clone(),
            array: window_array(&col.array, offset, len)?,
            null_count: 0,
        });
    }
    Ok(Table::new(table.name.clone(), Some(cols)))
}

/// Window one column's array by buffer arithmetic.
fn window_array(array: &Array, offset: usize, len: usize) -> io::Result<Array> {
    Ok(match array {
        Array::NumericArray(num) => {
            macro_rules! win_int {
                ($variant:ident, $arr:expr) => {
                    NumericArray::$variant(Arc::new(IntegerArray {
                        data: window_buffer(&$arr.data, offset, len),
                        null_mask: window_mask($arr.null_mask.as_ref(), offset, len),
                    }))
                };
            }
            macro_rules! win_float {
                ($variant:ident, $arr:expr) => {
                    NumericArray::$variant(Arc::new(FloatArray {
                        data: window_buffer(&$arr.data, offset, len),
                        null_mask: window_mask($arr.null_mask.as_ref(), offset, len),
                    }))
                };
            }
            Array::NumericArray(match num {
                NumericArray::Int32(arr) => win_int!(Int32, arr),
                NumericArray::Int64(arr) => win_int!(Int64, arr),
                NumericArray::UInt32(arr) => win_int!(UInt32, arr),
                NumericArray::UInt64(arr) => win_int!(UInt64, arr),
                NumericArray::Float32(arr) => win_float!(Float32, arr),
                NumericArray::Float64(arr) => win_float!(Float64, arr),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => win_int!(Int8, arr),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => win_int!(UInt8, arr),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => win_int!(Int16, arr),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => win_int!(UInt16, arr),
                #[cfg(feature = "decimal")]
                NumericArray::Decimal32(arr) => {
                    NumericArray::Decimal32(Arc::new(DecimalArray {
                        data: window_buffer(&arr.data, offset, len),
                        null_mask: window_mask(arr.null_mask.as_ref(), offset, len),
                        precision: arr.precision,
                        scale: arr.scale,
                    }))
                }
                #[cfg(feature = "decimal")]
                NumericArray::Decimal64(arr) => {
                    NumericArray::Decimal64(Arc::new(DecimalArray {
                        data: window_buffer(&arr.data, offset, len),
                        null_mask: window_mask(arr.null_mask.as_ref(), offset, len),
                        precision: arr.precision,
                        scale: arr.scale,
                    }))
                }
                #[cfg(feature = "decimal")]
                NumericArray::Decimal128(arr) => {
                    NumericArray::Decimal128(Arc::new(DecimalArray {
                        data: window_buffer(&arr.data, offset, len),
                        null_mask: window_mask(arr.null_mask.as_ref(), offset, len),
                        precision: arr.precision,
                        scale: arr.scale,
                    }))
                }
                NumericArray::Null => NumericArray::Null,
            })
        }

        Array::BooleanArray(arr) => {
            let bits = Bitmask::new(
                window_buffer(&arr.data.bits, offset / 8, len.div_ceil(8)),
                len,
            );
            Array::BooleanArray(Arc::new(BooleanArray::new(
                bits,
                window_mask(arr.null_mask.as_ref(), offset, len),
            )))
        }

        Array::TextArray(text) => Array::TextArray(match text {
            TextArray::String32(arr) => {
                TextArray::String32(Arc::new(window_string(arr, offset, len)))
            }
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => {
                TextArray::String64(Arc::new(window_string(arr, offset, len)))
            }
            TextArray::Categorical32(arr) => {
                TextArray::Categorical32(Arc::new(window_categorical(arr, offset, len)))
            }
            #[cfg(feature = "default_categorical_8")]
            TextArray::Categorical8(arr) => {
                TextArray::Categorical8(Arc::new(window_categorical(arr, offset, len)))
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => {
                TextArray::Categorical16(Arc::new(window_categorical(arr, offset, len)))
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => {
                TextArray::Categorical64(Arc::new(window_categorical(arr, offset, len)))
            }
            TextArray::Null => TextArray::Null,
        }),

        #[cfg(feature = "datetime")]
        Array::TemporalArray(temp) => {
            Array::TemporalArray(match temp {
                TemporalArray::Datetime32(arr) => {
                    TemporalArray::Datetime32(Arc::new(DatetimeArray {
                        data: window_buffer(&arr.data, offset, len),
                        null_mask: window_mask(arr.null_mask.as_ref(), offset, len),
                        time_unit: arr.time_unit,
                    }))
                }
                TemporalArray::Datetime64(arr) => {
                    TemporalArray::Datetime64(Arc::new(DatetimeArray {
                        data: window_buffer(&arr.data, offset, len),
                        null_mask: window_mask(arr.null_mask.as_ref(), offset, len),
                        time_unit: arr.time_unit,
                    }))
                }
                TemporalArray::Null => TemporalArray::Null,
            })
        }

        Array::Null => Array::Null,
    })
}

/// Window a string array: zero-copy values sub-slice, offsets borrowed
/// at row zero or rebased against the window base otherwise.
fn window_string<T>(arr: &StringArray<T>, offset: usize, len: usize) -> StringArray<T>
where
    T: Integer + Into<u64> + std::ops::Sub<Output = T>,
{
    let offs = arr.offsets.as_slice();
    let base = offs[offset];
    let base_bytes: u64 = base.into();
    let end_bytes: u64 = offs[offset + len].into();
    // A len-row window spans len + 1 offsets entries, the fence-posts
    // around its rows.
    let offsets = if offset == 0 {
        window_buffer(&arr.offsets, 0, len + 1)
    } else {
        Buffer::from_vec64(rebase_offsets(&offs[offset..offset + len + 1], base))
    };
    StringArray::new(
        window_buffer(
            &arr.data,
            base_bytes as usize,
            (end_bytes - base_bytes) as usize,
        ),
        window_mask(arr.null_mask.as_ref(), offset, len),
        offsets,
    )
}

/// Window a categorical array: indices sub-slice with the dictionary
/// carried across.
fn window_categorical<T: Integer>(
    arr: &CategoricalArray<T>,
    offset: usize,
    len: usize,
) -> CategoricalArray<T> {
    arr.slice_clone(offset, len)
}

/// Window a buffer to elements `[offset, offset + len)`. Shared-backed
/// buffers window through their owner with a reference-count bump.
/// Owned buffers copy the window, which arises only for columns that
/// were not decoded from shared memory.
fn window_buffer<T: Clone>(buf: &Buffer<T>, offset: usize, len: usize) -> Buffer<T> {
    match buf.shared_parts() {
        Some((owner, base, _)) => Buffer::from_shared_column(owner.clone(), base + offset, len),
        None => Buffer::from_slice(&buf.as_slice()[offset..offset + len]),
    }
}

/// Window a null mask, cut at the same boundaries as the data buffers.
fn window_mask(mask: Option<&Bitmask>, offset: usize, len: usize) -> Option<Bitmask> {
    mask.map(|m| Bitmask::new(window_buffer(&m.bits, offset / 8, len.div_ceil(8)), len))
}

/// Subtract the window base from a window's string offsets so the
/// values buffer the receiver sees starts at zero. One exact-size
/// buffer holds the rewritten offsets. Windows starting at row zero
/// keep their offsets zero-copy instead of coming through here.
fn rebase_offsets<T: Copy + std::ops::Sub<Output = T>>(offs: &[T], base: T) -> Vec64<T> {
    let mut rebased: Vec64<T> = Vec64::with_capacity(offs.len());
    // SAFETY: the capacity was just allocated for `offs.len()` values and
    // every element is written before the length is set.
    unsafe {
        let out = std::slice::from_raw_parts_mut(rebased.as_mut_ptr(), offs.len());
        for (o, v) in out.iter_mut().zip(offs) {
            *o = *v - base;
        }
        rebased.set_len(offs.len());
    }
    rebased
}
