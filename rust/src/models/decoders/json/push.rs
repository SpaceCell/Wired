// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Shared value -> ColumnBuilder dispatch
//!
//! The row decoder translates parser output into [`JsonValueRef`] and
//! calls [`push_value_into`] to land each cell in the right typed
//! builder. Type mismatches route through [`TypeMismatchPolicy`].

use std::io;

#[cfg(feature = "datetime")]
use minarrow::DatetimeArray;
use minarrow::traits::masked_array::MaskedArray;
use minarrow::traits::type_unions::Integer;
use minarrow::{BooleanArray, CategoricalArray, StringArray};

use crate::models::decoders::json::row_decoder::{MismatchAction, handle_type_mismatch};
use crate::models::decoders::json::value::JsonValueRef;
use crate::models::decoders::json::builder::{ColumnBuilder, TypeMismatchPolicy};

/// Push a parser-borrowed value into the destination builder, applying the
/// supplied [`TypeMismatchPolicy`] on type mismatch.
#[inline]
pub fn push_value_into(
    builder: &mut ColumnBuilder,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
) -> io::Result<()> {
    if matches!(value, JsonValueRef::Null) {
        builder.push_null();
        return Ok(());
    }

    match builder {
        ColumnBuilder::Int32(a) => {
            push_int(a, value, policy, row, field, |v| i32::try_from(v).ok())?
        }
        ColumnBuilder::Int64(a) => push_int(a, value, policy, row, field, Some)?,
        ColumnBuilder::UInt32(a) => {
            push_uint(a, value, policy, row, field, |v| u32::try_from(v).ok())?
        }
        ColumnBuilder::UInt64(a) => push_uint(a, value, policy, row, field, Some)?,
        ColumnBuilder::Float32(a) => push_float(a, value, policy, row, field, |v| v as f32)?,
        ColumnBuilder::Float64(a) => push_float(a, value, policy, row, field, |v| v)?,
        ColumnBuilder::Boolean(a) => push_bool(a, value, policy, row, field)?,
        ColumnBuilder::String32(a) => push_string(a, value, policy, row, field)?,
        #[cfg(feature = "large_string")]
        ColumnBuilder::String64(a) => push_string(a, value, policy, row, field)?,
        ColumnBuilder::Categorical32(a) => push_cat(a, value, policy, row, field)?,
        #[cfg(feature = "default_categorical_8")]
        ColumnBuilder::Categorical8(a) => push_cat(a, value, policy, row, field)?,
        #[cfg(feature = "extended_categorical")]
        ColumnBuilder::Categorical16(a) => push_cat(a, value, policy, row, field)?,
        #[cfg(feature = "extended_categorical")]
        ColumnBuilder::Categorical64(a) => push_cat(a, value, policy, row, field)?,
        #[cfg(feature = "datetime")]
        ColumnBuilder::Date32(a) => {
            push_date(a, value, policy, row, field, |v| i32::try_from(v).ok())?
        }
        #[cfg(feature = "datetime")]
        ColumnBuilder::Date64(a) => push_date(a, value, policy, row, field, |v| Some(v))?,
        #[cfg(feature = "extended_numeric_types")]
        ColumnBuilder::Int8(a) => push_int(a, value, policy, row, field, |v| i8::try_from(v).ok())?,
        #[cfg(feature = "extended_numeric_types")]
        ColumnBuilder::Int16(a) => {
            push_int(a, value, policy, row, field, |v| i16::try_from(v).ok())?
        }
        #[cfg(feature = "extended_numeric_types")]
        ColumnBuilder::UInt8(a) => {
            push_uint(a, value, policy, row, field, |v| u8::try_from(v).ok())?
        }
        #[cfg(feature = "extended_numeric_types")]
        ColumnBuilder::UInt16(a) => {
            push_uint(a, value, policy, row, field, |v| u16::try_from(v).ok())?
        }
    }
    Ok(())
}

/// Push a null cell on type mismatch, after applying the policy.
#[inline]
fn mismatch_null<A>(
    a: &mut A,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
    detail: &str,
) -> io::Result<()>
where
    A: MaskedArray,
{
    match handle_type_mismatch(policy, row, field, detail)? {
        MismatchAction::Coerce | MismatchAction::PushNull => {
            a.push_null();
            Ok(())
        }
    }
}

#[inline]
fn push_int<T, A, F>(
    a: &mut A,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
    narrow: F,
) -> io::Result<()>
where
    A: MaskedArray<LogicalType = T>,
    T: Copy,
    F: Fn(i64) -> Option<T>,
{
    let raw: i64 = match value {
        JsonValueRef::I64(v) => v,
        JsonValueRef::U64(v) => match i64::try_from(v) {
            Ok(x) => x,
            Err(_) => return mismatch_null(a, policy, row, field, "u64 out of i64 range"),
        },
        JsonValueRef::F64(v) => {
            if v.fract() == 0.0 && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                v as i64
            } else {
                return mismatch_null(a, policy, row, field, "expected integer");
            }
        }
        JsonValueRef::Bool(b) => b as i64,
        JsonValueRef::Str(s) => match policy {
            TypeMismatchPolicy::Coerce => match s.parse::<i64>() {
                Ok(x) => x,
                Err(_) => return mismatch_null(a, policy, row, field, "string not integer"),
            },
            _ => return mismatch_null(a, policy, row, field, "expected integer"),
        },
        JsonValueRef::Null => unreachable!("null handled before dispatch"),
    };
    match narrow(raw) {
        Some(v) => a.push(v),
        None => return mismatch_null(a, policy, row, field, "integer overflows target width"),
    }
    Ok(())
}

#[inline]
fn push_uint<T, A, F>(
    a: &mut A,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
    narrow: F,
) -> io::Result<()>
where
    A: MaskedArray<LogicalType = T>,
    T: Copy,
    F: Fn(u64) -> Option<T>,
{
    let raw: u64 = match value {
        JsonValueRef::U64(v) => v,
        JsonValueRef::I64(v) => match u64::try_from(v) {
            Ok(x) => x,
            Err(_) => return mismatch_null(a, policy, row, field, "negative for unsigned"),
        },
        JsonValueRef::F64(v) => {
            if v.fract() == 0.0 && v >= 0.0 && v <= u64::MAX as f64 {
                v as u64
            } else {
                return mismatch_null(a, policy, row, field, "expected unsigned integer");
            }
        }
        JsonValueRef::Bool(b) => b as u64,
        JsonValueRef::Str(s) => match policy {
            TypeMismatchPolicy::Coerce => match s.parse::<u64>() {
                Ok(x) => x,
                Err(_) => {
                    return mismatch_null(a, policy, row, field, "string not unsigned integer");
                }
            },
            _ => return mismatch_null(a, policy, row, field, "expected unsigned integer"),
        },
        JsonValueRef::Null => unreachable!("null handled before dispatch"),
    };
    match narrow(raw) {
        Some(v) => a.push(v),
        None => return mismatch_null(a, policy, row, field, "value overflows target width"),
    }
    Ok(())
}

#[inline]
fn push_float<T, A, F>(
    a: &mut A,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
    narrow: F,
) -> io::Result<()>
where
    A: MaskedArray<LogicalType = T>,
    T: Copy,
    F: Fn(f64) -> T,
{
    let raw: f64 = match value {
        JsonValueRef::F64(v) => v,
        JsonValueRef::I64(v) => v as f64,
        JsonValueRef::U64(v) => v as f64,
        JsonValueRef::Bool(b) => {
            if b {
                1.0
            } else {
                0.0
            }
        }
        JsonValueRef::Str(s) => match policy {
            TypeMismatchPolicy::Coerce => match s.parse::<f64>() {
                Ok(x) => x,
                Err(_) => return mismatch_null(a, policy, row, field, "string not number"),
            },
            _ => return mismatch_null(a, policy, row, field, "expected number"),
        },
        JsonValueRef::Null => unreachable!("null handled before dispatch"),
    };
    a.push(narrow(raw));
    Ok(())
}

#[inline]
fn push_bool(
    a: &mut BooleanArray<()>,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
) -> io::Result<()> {
    let v: bool = match value {
        JsonValueRef::Bool(v) => v,
        JsonValueRef::I64(v) => v != 0,
        JsonValueRef::U64(v) => v != 0,
        JsonValueRef::F64(v) => v != 0.0,
        JsonValueRef::Str(s) => match policy {
            TypeMismatchPolicy::Coerce => match s {
                "true" | "1" | "t" | "T" => true,
                "false" | "0" | "f" | "F" => false,
                _ => return mismatch_null(a, policy, row, field, "string not bool"),
            },
            _ => return mismatch_null(a, policy, row, field, "expected bool"),
        },
        JsonValueRef::Null => unreachable!("null handled before dispatch"),
    };
    a.push(v);
    Ok(())
}

#[inline]
fn push_string<O>(
    a: &mut StringArray<O>,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
) -> io::Result<()>
where
    O: Integer,
{
    if let JsonValueRef::Str(s) = value {
        a.push_str(s);
        return Ok(());
    }
    if policy == TypeMismatchPolicy::Coerce {
        let s = stringify_scalar(&value);
        a.push_str(&s);
        return Ok(());
    }
    mismatch_null(a, policy, row, field, "expected string")
}

#[inline]
fn push_cat<T>(
    a: &mut CategoricalArray<T>,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
) -> io::Result<()>
where
    T: Integer,
{
    if let JsonValueRef::Str(s) = value {
        a.push_str(s);
        return Ok(());
    }
    if policy == TypeMismatchPolicy::Coerce {
        let s = stringify_scalar(&value);
        a.push_str(&s);
        return Ok(());
    }
    mismatch_null(a, policy, row, field, "expected string for categorical")
}

#[cfg(feature = "datetime")]
#[inline]
fn push_date<T, F>(
    a: &mut DatetimeArray<T>,
    value: JsonValueRef<'_>,
    policy: TypeMismatchPolicy,
    row: usize,
    field: &str,
    narrow: F,
) -> io::Result<()>
where
    T: Integer,
    DatetimeArray<T>: MaskedArray<LogicalType = T>,
    F: Fn(i64) -> Option<T>,
{
    let raw: i64 = match value {
        JsonValueRef::I64(v) => v,
        JsonValueRef::U64(v) => match i64::try_from(v) {
            Ok(x) => x,
            Err(_) => return mismatch_null(a, policy, row, field, "Datetime out of i64 range"),
        },
        _ => return mismatch_null(a, policy, row, field, "expected integer for Datetime"),
    };
    match narrow(raw) {
        Some(v) => a.push(v),
        None => return mismatch_null(a, policy, row, field, "Datetime overflows target width"),
    }
    Ok(())
}

/// Stringify a non-string scalar for `Coerce` policy on string columns.
fn stringify_scalar(v: &JsonValueRef<'_>) -> String {
    match v {
        JsonValueRef::Bool(b) => b.to_string(),
        JsonValueRef::I64(v) => v.to_string(),
        JsonValueRef::U64(v) => v.to_string(),
        JsonValueRef::F64(v) => v.to_string(),
        JsonValueRef::Str(s) => (*s).to_string(),
        JsonValueRef::Null => String::new(),
    }
}
