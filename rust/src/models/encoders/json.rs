// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # JSON Encoder for Minarrow Tables and SuperTables
//!
//! Serialises a [`minarrow::Table`] or [`minarrow::SuperTable`] to either:
//! - **Array-of-objects**: `[{"col_a": 1, "col_b": "x"}, ...]`
//! - **NDJSON** (newline-delimited): `{"col_a": 1, "col_b": "x"}\n...`
//!
//! ## Supported types
//! - Integers: `Int8/16/32/64`, `UInt8/16/32/64` (extended widths behind `extended_numeric_types`)
//! - Floats: `Float32`, `Float64`
//! - Boolean
//! - Strings: `String32`, `String64` (behind `large_string`)
//! - Categorical: `Categorical8/16/32/64` - emitted as strings via dictionary lookup
//! - Temporal: `Date32`, `Date64` (behind `datetime`) - emitted as integers
//!
//! Nulls are emitted as JSON `null`. Numeric NaN/Infinity are emitted as `null`
//! since standard JSON has no representation for them.

use std::io::{self, Write};

use minarrow::traits::type_unions::Float;
use minarrow::{Array, Bitmask, NumericArray, SuperTable, Table, TextArray};

/// Output shape for JSON encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonFormat {
    /// Emit a single JSON array containing one object per row.
    ///
    /// When `pretty` is `true`, rows are indented on separate lines with a
    /// two-space indent. NDJSON is line-delimited by definition and does
    /// not support indentation.
    Array { pretty: bool },
    /// Emit one JSON object per line, newline-delimited (NDJSON / JSON Lines).
    Ndjson,
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat::Array { pretty: false }
    }
}

/// Options for JSON encoding.
#[derive(Debug, Clone)]
pub struct JsonEncodeOptions {
    /// Output shape - array-of-objects or NDJSON.
    pub format: JsonFormat,
    /// If true, include null values in output. If false, omit keys whose
    /// values are null to produce more compact output.
    pub include_nulls: bool,
}

impl Default for JsonEncodeOptions {
    fn default() -> Self {
        JsonEncodeOptions {
            format: JsonFormat::default(),
            include_nulls: true,
        }
    }
}

/// Serialise a [`Table`] to JSON.
pub fn encode_table_json<W: Write>(
    table: &Table,
    writer: W,
    options: &JsonEncodeOptions,
) -> io::Result<()> {
    match options.format {
        JsonFormat::Array { pretty } => encode_table_as_array(table, writer, options, pretty),
        JsonFormat::Ndjson => encode_table_as_ndjson(table, writer, options),
    }
}

/// Serialise a [`SuperTable`] to JSON by concatenating all batches.
pub fn encode_supertable_json<W: Write>(
    supertable: &SuperTable,
    mut writer: W,
    options: &JsonEncodeOptions,
) -> io::Result<()> {
    match options.format {
        JsonFormat::Array { pretty } => {
            writer.write_all(b"[")?;
            let mut first_row = true;
            for batch in supertable.batches.iter() {
                first_row =
                    write_batch_rows_as_array(batch, &mut writer, options, first_row, pretty)?;
            }
            if pretty && !first_row {
                writer.write_all(b"\n")?;
            }
            writer.write_all(b"]")?;
            if pretty {
                writer.write_all(b"\n")?;
            }
            Ok(())
        }
        JsonFormat::Ndjson => {
            for batch in supertable.batches.iter() {
                encode_table_as_ndjson(batch, &mut writer, options)?;
            }
            Ok(())
        }
    }
}

/// Write table rows as a JSON array-of-objects.
fn encode_table_as_array<W: Write>(
    table: &Table,
    mut writer: W,
    options: &JsonEncodeOptions,
    pretty: bool,
) -> io::Result<()> {
    writer.write_all(b"[")?;
    let first_row = write_batch_rows_as_array(table, &mut writer, options, true, pretty)?;
    if pretty && !first_row {
        writer.write_all(b"\n")?;
    }
    writer.write_all(b"]")?;
    if pretty {
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Write each row of the table as a single line of NDJSON.
fn encode_table_as_ndjson<W: Write>(
    table: &Table,
    mut writer: W,
    options: &JsonEncodeOptions,
) -> io::Result<()> {
    let null_masks = collect_null_masks(table);
    let cat_maps = collect_cat_maps(table);
    let key_prefixes = cache_key_prefixes(table);
    for row in 0..table.n_rows {
        write_row_object(
            &mut writer,
            table,
            row,
            &null_masks,
            &cat_maps,
            &key_prefixes,
            options.include_nulls,
        )?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Write table rows into an in-progress JSON array. Returns updated `first_row`
/// so subsequent batches can continue the array without emitting a leading comma.
fn write_batch_rows_as_array<W: Write>(
    table: &Table,
    writer: &mut W,
    options: &JsonEncodeOptions,
    mut first_row: bool,
    pretty: bool,
) -> io::Result<bool> {
    let null_masks = collect_null_masks(table);
    let cat_maps = collect_cat_maps(table);
    let key_prefixes = cache_key_prefixes(table);
    let include_nulls = options.include_nulls;

    for row in 0..table.n_rows {
        if !first_row {
            writer.write_all(b",")?;
        }
        first_row = false;
        if pretty {
            writer.write_all(b"\n  ")?;
        }
        write_row_object(
            writer,
            table,
            row,
            &null_masks,
            &cat_maps,
            &key_prefixes,
            include_nulls,
        )?;
    }
    Ok(first_row)
}

/// Collect null masks for all columns in one pass so per-row lookups are cheap.
fn collect_null_masks(table: &Table) -> Vec<Option<&Bitmask>> {
    let mut null_masks: Vec<Option<&Bitmask>> = Vec::with_capacity(table.cols.len());
    for col in &table.cols {
        match &col.array {
            Array::NumericArray(arr) => null_masks.push(arr.null_mask()),
            Array::BooleanArray(arr) => null_masks.push(arr.null_mask.as_ref()),
            Array::TextArray(TextArray::String32(arr)) => null_masks.push(arr.null_mask.as_ref()),
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => null_masks.push(arr.null_mask.as_ref()),
            Array::TextArray(TextArray::Categorical32(arr)) => {
                null_masks.push(arr.null_mask.as_ref())
            }
            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(arr)) => {
                null_masks.push(arr.null_mask.as_ref())
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => {
                null_masks.push(arr.null_mask.as_ref())
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => {
                null_masks.push(arr.null_mask.as_ref())
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => {
                let null_mask = match arr {
                    minarrow::TemporalArray::Datetime32(a) => a.null_mask.as_ref(),
                    minarrow::TemporalArray::Datetime64(a) => a.null_mask.as_ref(),
                    minarrow::TemporalArray::Null => None,
                };
                null_masks.push(null_mask);
            }
            _ => null_masks.push(None),
        }
    }
    null_masks
}

/// Collect categorical dictionaries up front for fast per-row lookups.
fn collect_cat_maps(table: &Table) -> Vec<Option<&[String]>> {
    let mut cat_maps: Vec<Option<&[String]>> = Vec::with_capacity(table.cols.len());
    for col in &table.cols {
        match &col.array {
            Array::TextArray(TextArray::Categorical32(arr)) => {
                cat_maps.push(Some(&arr.unique_values()))
            }
            #[cfg(feature = "default_categorical_8")]
            Array::TextArray(TextArray::Categorical8(arr)) => {
                cat_maps.push(Some(&arr.unique_values()))
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => {
                cat_maps.push(Some(&arr.unique_values()))
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => {
                cat_maps.push(Some(&arr.unique_values()))
            }
            _ => cat_maps.push(None),
        }
    }
    cat_maps
}

/// Pre-encode each column name as `"name":` once so the per-row write
/// loop emits the key with a single `write_all` instead of repeating
/// `write_json_string` + a colon for every cell.
fn cache_key_prefixes(table: &Table) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(table.cols.len());
    for col in &table.cols {
        let mut buf = Vec::with_capacity(col.field.name.len() + 3);
        // write_json_string handles escaping for control chars / quotes
        // in the name itself - reusing it here means the cached bytes
        // are correct even for unusual column names.
        write_json_string(&mut buf, &col.field.name).expect("Vec<u8> never errors");
        buf.push(b':');
        keys.push(buf);
    }
    keys
}

/// Write a single row as a JSON object. Caller controls surrounding commas
/// and whitespace (array separator, newline, etc.).
fn write_row_object<W: Write>(
    writer: &mut W,
    table: &Table,
    row: usize,
    null_masks: &[Option<&Bitmask>],
    cat_maps: &[Option<&[String]>],
    key_prefixes: &[Vec<u8>],
    include_nulls: bool,
) -> io::Result<()> {
    writer.write_all(b"{")?;
    let mut first_key = true;
    for (col_idx, col) in table.cols.iter().enumerate() {
        let is_null = if col.null_count == 0 {
            false
        } else {
            match null_masks[col_idx] {
                Some(mask) => !mask.get(row),
                None => false,
            }
        };

        if is_null && !include_nulls {
            continue;
        }

        if !first_key {
            writer.write_all(b",")?;
        }
        first_key = false;

        writer.write_all(&key_prefixes[col_idx])?;
        if is_null {
            writer.write_all(b"null")?;
        } else {
            write_cell_value(writer, col, row, cat_maps[col_idx])?;
        }
    }
    writer.write_all(b"}")?;
    Ok(())
}

/// Write a single cell value.
fn write_cell_value<W: Write>(
    writer: &mut W,
    col: &minarrow::FieldArray,
    row: usize,
    cat_map: Option<&[String]>,
) -> io::Result<()> {
    match &col.array {
        Array::NumericArray(n) => match n {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            NumericArray::Int32(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            NumericArray::Int64(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            NumericArray::UInt32(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            NumericArray::UInt64(arr) => write_integer(writer, arr.data.as_ref()[row])?,
            NumericArray::Float32(arr) => write_float(writer, arr.data.as_ref()[row])?,
            NumericArray::Float64(arr) => write_float(writer, arr.data.as_ref()[row])?,
            _ => writer.write_all(b"null")?,
        },
        Array::BooleanArray(arr) => {
            writer.write_all(if arr.data.get(row) { b"true" } else { b"false" })?;
        }
        Array::TextArray(TextArray::String32(arr)) => {
            let start = arr.offsets.as_ref()[row] as usize;
            let end = arr.offsets.as_ref()[row + 1] as usize;
            let s = std::str::from_utf8(&arr.data.as_ref()[start..end]).unwrap_or("");
            write_json_string(writer, s)?;
        }
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(arr)) => {
            let start = arr.offsets.as_ref()[row] as usize;
            let end = arr.offsets.as_ref()[row + 1] as usize;
            let s = std::str::from_utf8(&arr.data.as_ref()[start..end]).unwrap_or("");
            write_json_string(writer, s)?;
        }
        Array::TextArray(TextArray::Categorical32(arr)) => {
            let idx = arr.data.as_ref()[row] as usize;
            let val = cat_map
                .and_then(|m| m.get(idx))
                .map(String::as_str)
                .unwrap_or("");
            write_json_string(writer, val)?;
        }
        #[cfg(feature = "default_categorical_8")]
        Array::TextArray(TextArray::Categorical8(arr)) => {
            let idx = arr.data.as_ref()[row] as usize;
            let val = cat_map
                .and_then(|m| m.get(idx))
                .map(String::as_str)
                .unwrap_or("");
            write_json_string(writer, val)?;
        }
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical16(arr)) => {
            let idx = arr.data.as_ref()[row] as usize;
            let val = cat_map
                .and_then(|m| m.get(idx))
                .map(String::as_str)
                .unwrap_or("");
            write_json_string(writer, val)?;
        }
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical64(arr)) => {
            let idx = arr.data.as_ref()[row] as usize;
            let val = cat_map
                .and_then(|m| m.get(idx))
                .map(String::as_str)
                .unwrap_or("");
            write_json_string(writer, val)?;
        }
        #[cfg(feature = "datetime")]
        Array::TemporalArray(temp) => match temp {
            minarrow::TemporalArray::Datetime32(arr) => {
                write_integer(writer, arr.data.as_ref()[row])?
            }
            minarrow::TemporalArray::Datetime64(arr) => {
                write_integer(writer, arr.data.as_ref()[row])?
            }
            minarrow::TemporalArray::Null => writer.write_all(b"null")?,
        },
        _ => writer.write_all(b"null")?,
    }
    Ok(())
}

/// Emit an integer as JSON via the vendored `int_ascii` formatter, which
/// writes straight into a small stack buffer and bypasses `fmt::Display`'s
/// formatter machinery.
#[inline]
fn write_integer<W: Write, I: super::int_ascii::Integer>(writer: &mut W, v: I) -> io::Result<()> {
    let mut buf = super::int_ascii::Buffer::new();
    writer.write_all(buf.format(v).as_bytes())
}

/// Emit a float as JSON via `ryu`, which produces the shortest correct
/// round-trip including a trailing `.0` for integer-valued floats. NaN
/// and Infinity become `null` since JSON has no representation for them.
#[inline]
fn write_float<W: Write, F: Float + ryu::Float>(writer: &mut W, v: F) -> io::Result<()> {
    if !v.is_finite() {
        return writer.write_all(b"null");
    }
    let mut buf = ryu::Buffer::new();
    writer.write_all(buf.format_finite(v).as_bytes())
}

/// Emit a UTF-8 string as a JSON string literal with escapes for `"`, `\`,
/// and ASCII control characters per RFC 8259. Walks bytes, emits unescaped
/// runs in bulk via `write_all`, and individual escape sequences inline.
fn write_json_string<W: Write>(writer: &mut W, s: &str) -> io::Result<()> {
    writer.write_all(b"\"")?;
    let bytes = s.as_bytes();
    let mut start = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b >= 0x20 && b != b'"' && b != b'\\' {
            continue;
        }
        if start < i {
            writer.write_all(&bytes[start..i])?;
        }
        match b {
            b'"' => writer.write_all(b"\\\"")?,
            b'\\' => writer.write_all(b"\\\\")?,
            0x08 => writer.write_all(b"\\b")?,
            0x09 => writer.write_all(b"\\t")?,
            0x0a => writer.write_all(b"\\n")?,
            0x0c => writer.write_all(b"\\f")?,
            0x0d => writer.write_all(b"\\r")?,
            _ => write!(writer, "\\u{:04x}", b)?,
        }
        start = i + 1;
    }
    if start < bytes.len() {
        writer.write_all(&bytes[start..])?;
    }
    writer.write_all(b"\"")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use minarrow::{
        Array, ArrowType, Bitmask, Buffer, Field, FieldArray, IntegerArray, NumericArray,
        StringArray, Table, TextArray, vec64,
    };
    use simd_json::prelude::ValueAsArray;

    /// Parse JSON output with simd-json's owned-value type for assertions.
    fn parse(bytes: Vec<u8>) -> simd_json::OwnedValue {
        let mut buf = bytes;
        simd_json::to_owned_value(&mut buf).unwrap()
    }

    fn make_test_table() -> Table {
        let int_col = FieldArray {
            field: Field {
                name: "ints".to_string(),
                dtype: ArrowType::Int32,
                nullable: true,
                metadata: Default::default(),
            }
            .into(),
            array: Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
                data: Buffer::from(vec64![1, 2, 3, 4]),
                null_mask: Some(Bitmask::from_bools(&[true, false, true, true])),
            }))),
            null_count: 1,
        };
        let str_col = FieldArray {
            field: Field {
                name: "strings".to_string(),
                dtype: ArrowType::String,
                nullable: true,
                metadata: Default::default(),
            }
            .into(),
            array: Array::TextArray(TextArray::String32(Arc::new(StringArray {
                offsets: Buffer::from(vec64![0u32, 5, 9, 14, 18]),
                data: Buffer::from_vec64("helloabcdworldrust".as_bytes().into()),
                null_mask: Some(Bitmask::from_bools(&[true, false, true, true])),
            }))),
            null_count: 1,
        };
        Table::new("test".to_string(), Some(vec![int_col, str_col]))
    }

    #[test]
    fn encode_array_basic() {
        let table = make_test_table();
        let mut out = Vec::new();
        encode_table_json(&table, &mut out, &JsonEncodeOptions::default()).unwrap();
        let v = parse(out);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn encode_ndjson_one_row_per_line() {
        let table = make_test_table();
        let mut out = Vec::new();
        let opts = JsonEncodeOptions {
            format: JsonFormat::Ndjson,
            ..Default::default()
        };
        encode_table_json(&table, &mut out, &opts).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s.lines().count(), 4);
        for line in s.lines() {
            let mut bytes = line.as_bytes().to_vec();
            simd_json::to_owned_value(&mut bytes).unwrap();
        }
    }

    #[test]
    fn encode_omit_nulls() {
        let table = make_test_table();
        let mut out = Vec::new();
        let opts = JsonEncodeOptions {
            format: JsonFormat::Ndjson,
            include_nulls: false,
            ..Default::default()
        };
        encode_table_json(&table, &mut out, &opts).unwrap();
        let s = String::from_utf8(out).unwrap();
        // Row 1 has both null fields - empty object expected.
        assert!(s.lines().any(|l| l == "{}"));
    }

    #[test]
    fn encode_escapes_special_chars() {
        let str_col = FieldArray {
            field: Field {
                name: "msg".to_string(),
                dtype: ArrowType::String,
                nullable: false,
                metadata: Default::default(),
            }
            .into(),
            array: Array::TextArray(TextArray::String32(Arc::new(StringArray {
                offsets: Buffer::from(vec64![0u32, 15]),
                data: Buffer::from_vec64(b"quote:\" slash:\\".to_vec().into()),
                null_mask: None,
            }))),
            null_count: 0,
        };
        let tbl = Table::new("t".to_string(), Some(vec![str_col]));
        let mut out = Vec::new();
        encode_table_json(&tbl, &mut out, &JsonEncodeOptions::default()).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains(r#"\""#));
        assert!(s.contains(r#"\\"#));
    }

    #[test]
    fn encode_pretty_is_parseable() {
        let table = make_test_table();
        let mut out = Vec::new();
        let opts = JsonEncodeOptions {
            format: JsonFormat::Array { pretty: true },
            ..Default::default()
        };
        encode_table_json(&table, &mut out, &opts).unwrap();
        let v = parse(out);
        assert_eq!(v.as_array().unwrap().len(), 4);
    }

    #[test]
    fn encode_control_char_as_uescape() {
        let str_col = FieldArray {
            field: Field {
                name: "ctl".to_string(),
                dtype: ArrowType::String,
                nullable: false,
                metadata: Default::default(),
            }
            .into(),
            array: Array::TextArray(TextArray::String32(Arc::new(StringArray {
                offsets: Buffer::from(vec64![0u32, 3]),
                data: Buffer::from_vec64(vec64![b'a', 0x01, b'b']),
                null_mask: None,
            }))),
            null_count: 0,
        };
        let tbl = Table::new("t".to_string(), Some(vec![str_col]));
        let mut out = Vec::new();
        encode_table_json(&tbl, &mut out, &JsonEncodeOptions::default()).unwrap();
        let s = String::from_utf8(out).unwrap();
        // 0x01 must be escaped as \u0001 in JSON output, not emitted as a raw byte.
        assert!(s.contains("\\u0001"));
        assert!(!s.as_bytes().contains(&0x01u8));
    }
}
