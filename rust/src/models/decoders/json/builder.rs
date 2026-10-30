// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # ColumnBuilder
//!
//! Thin dispatcher that grows a [`minarrow`] typed array via the
//! [`MaskedArray`] trait or each variant's inherent `push_str` method.
//!
//! Construction goes through `<Array>::with_capacity(n_rows, nullable)`,
//! which pre-reserves the [`Vec64`](minarrow::Vec64)-backed data buffer and (when nullable)
//! the [`Bitmask`](minarrow::Bitmask) - so the JSON row loop performs zero reallocations on
//! the hot path. `push_null` lazily materialises the mask in the
//! `MaskedArray` default implementation when the column is non-nullable
//! but a null shows up at runtime.

use std::sync::Arc;

use minarrow::ffi::arrow_dtype::CategoricalIndexType;
#[cfg(feature = "datetime")]
use minarrow::ffi::arrow_dtype::Dt;
use minarrow::traits::masked_array::MaskedArray;
use minarrow::{
    Array, ArrowType, BooleanArray, CategoricalArray, Field, FieldArray, FloatArray, IntegerArray,
    NumericArray, StringArray, TextArray,
};

/// Strategy for handling a JSON value whose type does not match the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TypeMismatchPolicy {
    /// Return an error identifying the row and column.
    #[default]
    Error,
    /// Attempt to coerce (string<->number, bool<->0/1). On failure, write null.
    Coerce,
    /// Silently record the cell as null.
    Null,
}

/// Per-column accumulator that wraps a typed minarrow array. The decoder calls
/// the typed `push_*` methods or [`push_null`](Self::push_null) for missing
/// or `null` cells, then [`finish`](Self::finish) to obtain a [`FieldArray`].
pub enum ColumnBuilder {
    Int32(IntegerArray<i32>),
    Int64(IntegerArray<i64>),
    UInt32(IntegerArray<u32>),
    UInt64(IntegerArray<u64>),
    Float32(FloatArray<f32>),
    Float64(FloatArray<f64>),
    Boolean(BooleanArray<()>),
    String32(StringArray<u32>),
    #[cfg(feature = "large_string")]
    String64(StringArray<u64>),
    Categorical32(CategoricalArray<u32>),
    #[cfg(feature = "default_categorical_8")]
    Categorical8(CategoricalArray<u8>),
    #[cfg(feature = "extended_categorical")]
    Categorical16(CategoricalArray<u16>),
    #[cfg(feature = "extended_categorical")]
    Categorical64(CategoricalArray<u64>),
    #[cfg(feature = "datetime")]
    Date32(minarrow::DatetimeArray<i32>),
    #[cfg(feature = "datetime")]
    Date64(minarrow::DatetimeArray<i64>),
    #[cfg(feature = "extended_numeric_types")]
    Int8(IntegerArray<i8>),
    #[cfg(feature = "extended_numeric_types")]
    Int16(IntegerArray<i16>),
    #[cfg(feature = "extended_numeric_types")]
    UInt8(IntegerArray<u8>),
    #[cfg(feature = "extended_numeric_types")]
    UInt16(IntegerArray<u16>),
}

impl ColumnBuilder {
    /// Construct a builder matching the field's Arrow type, reserving
    /// capacity for `n_rows` cells so the row loop performs zero
    /// reallocations. `string_bytes_per_row` sizes the data buffer for
    /// String / LargeString columns at `n_rows * string_bytes_per_row`
    /// up front; tune this for long-string columns to avoid growth
    /// reallocations during the decode.
    pub fn for_field(
        field: &Field,
        n_rows: usize,
        string_bytes_per_row: usize,
    ) -> std::io::Result<Self> {
        let nullable = field.nullable;
        let b = match &field.dtype {
            ArrowType::Int32 => ColumnBuilder::Int32(IntegerArray::with_capacity(n_rows, nullable)),
            ArrowType::Int64 => ColumnBuilder::Int64(IntegerArray::with_capacity(n_rows, nullable)),
            ArrowType::UInt32 => {
                ColumnBuilder::UInt32(IntegerArray::with_capacity(n_rows, nullable))
            }
            ArrowType::UInt64 => {
                ColumnBuilder::UInt64(IntegerArray::with_capacity(n_rows, nullable))
            }
            ArrowType::Float32 => {
                ColumnBuilder::Float32(FloatArray::with_capacity(n_rows, nullable))
            }
            ArrowType::Float64 => {
                ColumnBuilder::Float64(FloatArray::with_capacity(n_rows, nullable))
            }
            ArrowType::Boolean => {
                ColumnBuilder::Boolean(BooleanArray::with_capacity(n_rows, nullable))
            }
            ArrowType::String => ColumnBuilder::String32(StringArray::<u32>::with_capacity(
                n_rows,
                n_rows * string_bytes_per_row,
                nullable,
            )),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => ColumnBuilder::String64(StringArray::<u64>::with_capacity(
                n_rows,
                n_rows * string_bytes_per_row,
                nullable,
            )),
            ArrowType::Dictionary(idx_ty) => match idx_ty {
                CategoricalIndexType::UInt32 => ColumnBuilder::Categorical32(
                    CategoricalArray::<u32>::with_capacity(n_rows, None, nullable),
                ),
                #[cfg(feature = "default_categorical_8")]
                CategoricalIndexType::UInt8 => ColumnBuilder::Categorical8(
                    CategoricalArray::<u8>::with_capacity(n_rows, None, nullable),
                ),
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt16 => ColumnBuilder::Categorical16(
                    CategoricalArray::<u16>::with_capacity(n_rows, None, nullable),
                ),
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt64 => ColumnBuilder::Categorical64(
                    CategoricalArray::<u64>::with_capacity(n_rows, None, nullable),
                ),
            },
            #[cfg(feature = "datetime")]
            ArrowType::DateTime(Dt::Date32) => ColumnBuilder::Date32(
                minarrow::DatetimeArray::with_capacity(n_rows, nullable, Dt::Date32),
            ),
            #[cfg(feature = "datetime")]
            ArrowType::DateTime(Dt::Date64) => ColumnBuilder::Date64(
                minarrow::DatetimeArray::with_capacity(n_rows, nullable, Dt::Date64),
            ),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => ColumnBuilder::Int8(IntegerArray::with_capacity(n_rows, nullable)),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => ColumnBuilder::Int16(IntegerArray::with_capacity(n_rows, nullable)),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => ColumnBuilder::UInt8(IntegerArray::with_capacity(n_rows, nullable)),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => {
                ColumnBuilder::UInt16(IntegerArray::with_capacity(n_rows, nullable))
            }
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unsupported JSON column type: {:?}", other),
                ));
            }
        };
        Ok(b)
    }

    /// Append a null cell using minarrow's per-type push_null fast path.
    #[inline]
    pub fn push_null(&mut self) {
        match self {
            ColumnBuilder::Int32(a) => a.push_null(),
            ColumnBuilder::Int64(a) => a.push_null(),
            ColumnBuilder::UInt32(a) => a.push_null(),
            ColumnBuilder::UInt64(a) => a.push_null(),
            ColumnBuilder::Float32(a) => a.push_null(),
            ColumnBuilder::Float64(a) => a.push_null(),
            ColumnBuilder::Boolean(a) => a.push_null(),
            ColumnBuilder::String32(a) => a.push_null(),
            #[cfg(feature = "large_string")]
            ColumnBuilder::String64(a) => a.push_null(),
            ColumnBuilder::Categorical32(a) => a.push_null(),
            #[cfg(feature = "default_categorical_8")]
            ColumnBuilder::Categorical8(a) => a.push_null(),
            #[cfg(feature = "extended_categorical")]
            ColumnBuilder::Categorical16(a) => a.push_null(),
            #[cfg(feature = "extended_categorical")]
            ColumnBuilder::Categorical64(a) => a.push_null(),
            #[cfg(feature = "datetime")]
            ColumnBuilder::Date32(a) => a.push_null(),
            #[cfg(feature = "datetime")]
            ColumnBuilder::Date64(a) => a.push_null(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::Int8(a) => a.push_null(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::Int16(a) => a.push_null(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::UInt8(a) => a.push_null(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::UInt16(a) => a.push_null(),
        }
    }

    /// Number of cells appended so far.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ColumnBuilder::Int32(a) => a.len(),
            ColumnBuilder::Int64(a) => a.len(),
            ColumnBuilder::UInt32(a) => a.len(),
            ColumnBuilder::UInt64(a) => a.len(),
            ColumnBuilder::Float32(a) => a.len(),
            ColumnBuilder::Float64(a) => a.len(),
            ColumnBuilder::Boolean(a) => a.len(),
            ColumnBuilder::String32(a) => a.len(),
            #[cfg(feature = "large_string")]
            ColumnBuilder::String64(a) => a.len(),
            ColumnBuilder::Categorical32(a) => a.len(),
            #[cfg(feature = "default_categorical_8")]
            ColumnBuilder::Categorical8(a) => a.len(),
            #[cfg(feature = "extended_categorical")]
            ColumnBuilder::Categorical16(a) => a.len(),
            #[cfg(feature = "extended_categorical")]
            ColumnBuilder::Categorical64(a) => a.len(),
            #[cfg(feature = "datetime")]
            ColumnBuilder::Date32(a) => a.len(),
            #[cfg(feature = "datetime")]
            ColumnBuilder::Date64(a) => a.len(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::Int8(a) => a.len(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::Int16(a) => a.len(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::UInt8(a) => a.len(),
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::UInt16(a) => a.len(),
        }
    }

    /// Returns `true` when no cells have been appended.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wrap the inner typed array in a [`FieldArray`] tagged with `field`.
    pub fn finish(self, field: Arc<Field>) -> FieldArray {
        let (array, null_count) = match self {
            ColumnBuilder::Int32(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::Int32(Arc::new(a))), nc)
            }
            ColumnBuilder::Int64(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::Int64(Arc::new(a))), nc)
            }
            ColumnBuilder::UInt32(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::UInt32(Arc::new(a))), nc)
            }
            ColumnBuilder::UInt64(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::UInt64(Arc::new(a))), nc)
            }
            ColumnBuilder::Float32(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::Float32(Arc::new(a))), nc)
            }
            ColumnBuilder::Float64(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::Float64(Arc::new(a))), nc)
            }
            ColumnBuilder::Boolean(a) => {
                let nc = a.null_count();
                (Array::BooleanArray(Arc::new(a)), nc)
            }
            ColumnBuilder::String32(a) => {
                let nc = a.null_count();
                (Array::TextArray(TextArray::String32(Arc::new(a))), nc)
            }
            #[cfg(feature = "large_string")]
            ColumnBuilder::String64(a) => {
                let nc = a.null_count();
                (Array::TextArray(TextArray::String64(Arc::new(a))), nc)
            }
            ColumnBuilder::Categorical32(a) => {
                let nc = a.null_count();
                (Array::TextArray(TextArray::Categorical32(Arc::new(a))), nc)
            }
            #[cfg(feature = "default_categorical_8")]
            ColumnBuilder::Categorical8(a) => {
                let nc = a.null_count();
                (Array::TextArray(TextArray::Categorical8(Arc::new(a))), nc)
            }
            #[cfg(feature = "extended_categorical")]
            ColumnBuilder::Categorical16(a) => {
                let nc = a.null_count();
                (Array::TextArray(TextArray::Categorical16(Arc::new(a))), nc)
            }
            #[cfg(feature = "extended_categorical")]
            ColumnBuilder::Categorical64(a) => {
                let nc = a.null_count();
                (Array::TextArray(TextArray::Categorical64(Arc::new(a))), nc)
            }
            #[cfg(feature = "datetime")]
            ColumnBuilder::Date32(a) => {
                let nc = a.null_count();
                (
                    Array::TemporalArray(minarrow::TemporalArray::Datetime32(Arc::new(a))),
                    nc,
                )
            }
            #[cfg(feature = "datetime")]
            ColumnBuilder::Date64(a) => {
                let nc = a.null_count();
                (
                    Array::TemporalArray(minarrow::TemporalArray::Datetime64(Arc::new(a))),
                    nc,
                )
            }
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::Int8(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::Int8(Arc::new(a))), nc)
            }
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::Int16(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::Int16(Arc::new(a))), nc)
            }
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::UInt8(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::UInt8(Arc::new(a))), nc)
            }
            #[cfg(feature = "extended_numeric_types")]
            ColumnBuilder::UInt16(a) => {
                let nc = a.null_count();
                (Array::NumericArray(NumericArray::UInt16(Arc::new(a))), nc)
            }
        };
        FieldArray {
            field,
            array,
            null_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minarrow::traits::masked_array::MaskedArray;
    use minarrow::{ArrowType, Field};
    use std::sync::Arc;

    fn field(name: &str, dtype: ArrowType, nullable: bool) -> Arc<Field> {
        Arc::new(Field::new(name, dtype, nullable, None))
    }

    #[test]
    fn int32_no_nulls_roundtrip() {
        let mut b =
            ColumnBuilder::for_field(&Field::new("x", ArrowType::Int32, false, None), 4, 16)
                .unwrap();
        if let ColumnBuilder::Int32(a) = &mut b {
            a.push(1);
            a.push(2);
            a.push(3);
            a.push(4);
        } else {
            panic!();
        }
        let fa = b.finish(field("x", ArrowType::Int32, false));
        assert_eq!(fa.null_count, 0);
        match &fa.array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_ref(), &[1, 2, 3, 4]);
                assert!(arr.null_mask.is_none());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn int32_with_nulls_lazy_mask() {
        let mut b = ColumnBuilder::for_field(&Field::new("x", ArrowType::Int32, true, None), 4, 16)
            .unwrap();
        if let ColumnBuilder::Int32(a) = &mut b {
            a.push(1);
            a.push(2);
            a.push_null();
            a.push(4);
        } else {
            panic!();
        }
        let fa = b.finish(field("x", ArrowType::Int32, true));
        assert_eq!(fa.null_count, 1);
        match &fa.array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                let mask = arr.null_mask.as_ref().unwrap();
                assert!(mask.get(0));
                assert!(mask.get(1));
                assert!(!mask.get(2));
                assert!(mask.get(3));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn string_offsets_lockstep() {
        let mut b =
            ColumnBuilder::for_field(&Field::new("s", ArrowType::String, true, None), 4, 16)
                .unwrap();
        if let ColumnBuilder::String32(a) = &mut b {
            a.push_str("hello");
            a.push_str("world");
            a.push_null();
            a.push_str("!");
        } else {
            panic!();
        }
        let fa = b.finish(field("s", ArrowType::String, true));
        assert_eq!(fa.null_count, 1);
        match &fa.array {
            Array::TextArray(TextArray::String32(arr)) => {
                assert_eq!(arr.offsets.as_ref(), &[0u32, 5, 10, 10, 11]);
                assert_eq!(arr.data.as_ref(), b"helloworld!");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn boolean_bit_packed() {
        let mut b =
            ColumnBuilder::for_field(&Field::new("flag", ArrowType::Boolean, false, None), 8, 16)
                .unwrap();
        if let ColumnBuilder::Boolean(a) = &mut b {
            for v in [true, false, true, true, false, false, true, false] {
                a.push(v);
            }
        } else {
            panic!();
        }
        let fa = b.finish(field("flag", ArrowType::Boolean, false));
        match &fa.array {
            Array::BooleanArray(arr) => {
                assert_eq!(arr.len(), 8);
                let bits: Vec<bool> = (0..8).map(|i| arr.data.get(i)).collect();
                assert_eq!(
                    bits,
                    vec![true, false, true, true, false, false, true, false]
                );
            }
            _ => panic!(),
        }
    }

    #[cfg(feature = "default_categorical_8")]
    #[test]
    fn categorical_dedup_via_push_str() {
        let mut b = ColumnBuilder::for_field(
            &Field::new(
                "c",
                ArrowType::Dictionary(CategoricalIndexType::UInt8),
                false,
                None,
            ),
            6,
            16,
        )
        .unwrap();
        if let ColumnBuilder::Categorical8(a) = &mut b {
            a.push_str("red");
            a.push_str("green");
            a.push_str("red");
            a.push_str("blue");
            a.push_str("green");
            a.push_str("red");
        } else {
            panic!();
        }
        let fa = b.finish(field(
            "c",
            ArrowType::Dictionary(CategoricalIndexType::UInt8),
            false,
        ));
        match &fa.array {
            Array::TextArray(TextArray::Categorical8(arr)) => {
                assert_eq!(arr.unique_values().len(), 3);
                assert_eq!(arr.data.as_ref(), &[0u8, 1, 0, 2, 1, 0]);
            }
            _ => panic!(),
        }
    }
}
