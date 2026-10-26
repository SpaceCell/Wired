// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! CSV Encoder for Minarrow Tables/SuperTables.
//! - Handles all supported types: Int32, Int64, UInt32, UInt64, Float32, Float64, Boolean, String32, Categorical32.
//! - Supports custom delimiter, header row, quoting, and null representation.
//! - Serialises a Table or SuperTable to any Write or `Vec<u8>`.
//!
//! ## Fast path
//!
//! `encode_table_csv` writes the entire table into a single contiguous
//! `Vec<u8>` and emits one `write_all` to the underlying sink at the
//! end. Numeric formatting goes through `int_ascii::Buffer` and
//! `ryu::Buffer` (no allocation, ~10x faster than `core::fmt::Display`).
//! String quoting uses a single `memchr` pass to decide whether quoting
//! is needed; bytes are copied straight from the Arrow data buffer
//! without going through `String`.

use minarrow::{Array, Bitmask, Integer, NumericArray, SuperTable, Table, TextArray};
use std::io::{self, Write};

/// Options for CSV encoding.
#[derive(Debug, Clone)]
pub struct CsvEncodeOptions {
    /// Delimiter (e.g., b',' for CSV, b'\t' for TSV).
    pub delimiter: u8,
    /// Whether to write header row.
    pub write_header: bool,
    /// String to represent nulls.
    pub null_repr: &'static str,
    /// Quote character to use (default: '"').
    pub quote: u8,
}

impl Default for CsvEncodeOptions {
    fn default() -> Self {
        CsvEncodeOptions {
            delimiter: b',',
            write_header: true,
            null_repr: "",
            quote: b'"',
        }
    }
}

/// Returns true if the bytes need to be wrapped in quotes for CSV
/// output. One SIMD-accelerated pass via `memchr3` covers three of the
/// four hot bytes; `\r` and leading/trailing space are cheap edge
/// checks.
#[inline]
fn needs_quoting(bytes: &[u8], delimiter: u8, quote: u8) -> bool {
    if bytes.first() == Some(&b' ') || bytes.last() == Some(&b' ') {
        return true;
    }
    memchr::memchr3(delimiter, quote, b'\n', bytes).is_some()
        || memchr::memchr(b'\r', bytes).is_some()
}

/// Append the CSV-encoded form of `bytes` to `out`: either the bytes
/// verbatim, or wrapped in quotes with embedded quotes doubled. Walks
/// the bytes in runs delimited by `quote`, copying whole runs via
/// `extend_from_slice` so the common case is a memcpy.
#[inline]
fn append_csv_string(out: &mut Vec<u8>, bytes: &[u8], delimiter: u8, quote: u8) {
    if !needs_quoting(bytes, delimiter, quote) {
        out.extend_from_slice(bytes);
        return;
    }
    out.push(quote);
    let mut start = 0;
    while let Some(pos) = memchr::memchr(quote, &bytes[start..]) {
        let abs = start + pos;
        out.extend_from_slice(&bytes[start..abs]);
        out.push(quote);
        out.push(quote);
        start = abs + 1;
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(quote);
}

/// Append a String<T> cell payload to `out` without going through a
/// String allocation. The Arrow String layout stores n+1 offsets, and
/// each cell's bytes live at `data[offsets[row]..offsets[row+1]]`.
/// Generic over the offset width so the same code path covers
/// `StringArray<u32>` and `StringArray<u64>`.
#[inline]
fn append_str_cell<T: Integer>(
    out: &mut Vec<u8>,
    data: &[u8],
    offsets: &[T],
    row: usize,
    delimiter: u8,
    quote: u8,
) {
    let start = offsets[row].to_usize();
    let mut end = offsets[row + 1].to_usize();
    // Defensive: minarrow's String layout sometimes truncates the
    // trailing offset; clamp to the real data length on the last row.
    if row + 1 == offsets.len() - 1 && end < data.len() {
        end = data.len();
    }
    append_csv_string(out, &data[start..end], delimiter, quote);
}

/// Format one decimal storage integer with its scale into the CSV output
/// buffer, placing the decimal point at the position determined by scale.
#[cfg(feature = "decimal")]
fn write_decimal_cell(out: &mut Vec<u8>, raw: i128, scale: i8) {
    use std::fmt::Write as _;
    let mut buf = String::with_capacity(48);
    if scale == 0 {
        let _ = write!(buf, "{}", raw);
    } else if scale < 0 {
        let _ = write!(buf, "{}", raw);
        for _ in 0..(-scale) {
            buf.push('0');
        }
    } else {
        let s = scale as usize;
        let raw_str = format!("{}", raw);
        let (is_negative, digits) = if raw_str.starts_with('-') {
            (true, &raw_str[1..])
        } else {
            (false, raw_str.as_str())
        };
        let padded = if digits.len() <= s {
            format!("{:0>width$}", digits, width = s + 1)
        } else {
            digits.to_string()
        };
        let (integer_part, frac_part) = padded.split_at(padded.len() - s);
        if is_negative {
            buf.push('-');
        }
        buf.push_str(integer_part);
        buf.push('.');
        buf.push_str(frac_part);
    }
    out.extend_from_slice(buf.as_bytes());
}

/// Average bytes per cell heuristic per column type; used to size the
/// scratch buffer up front so the encode loop hits no Vec growth.
fn estimate_cell_width(arr: &Array, n_rows: usize) -> usize {
    match arr {
        Array::NumericArray(n) => match n {
            NumericArray::Int32(_) | NumericArray::UInt32(_) => 6,
            NumericArray::Int64(_) | NumericArray::UInt64(_) => 10,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(_) | NumericArray::UInt8(_) => 4,
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(_) | NumericArray::UInt16(_) => 5,
            NumericArray::Float32(_) | NumericArray::Float64(_) => 16,
            _ => 12,
        },
        Array::BooleanArray(_) => 5,
        Array::TextArray(TextArray::String32(arr)) => {
            arr.data.len().checked_div(n_rows).map_or(8, |v| v + 4)
        }
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(arr)) => {
            if n_rows == 0 {
                8
            } else {
                arr.data.len() / n_rows + 4
            }
        }
        Array::TextArray(_) => {
            // Categorical columns: rough average over dictionary entries.
            12
        }
        #[cfg(feature = "datetime")]
        Array::TemporalArray(_) => 12,
        _ => 12,
    }
}

/// Serialises a Minarrow `Table` (i.e., Arrow `RecordBatch`) to any `Write` as CSV.
/// - Supports custom delimiter, null representation, header.
/// - Escapes/quotes fields as needed.
/// - Errors propagate from writer.
pub fn encode_table_csv<W: Write>(
    table: &Table,
    mut writer: W,
    options: &CsvEncodeOptions,
) -> io::Result<()> {
    let CsvEncodeOptions {
        delimiter,
        write_header,
        null_repr,
        quote,
    } = *options;

    let n_rows = table.n_rows;

    // Pre-extract null masks once per column so the row loop skips the
    // typed match for null detection. Mirrors the original encoder's
    // null_masks vector.
    let mut null_masks: Vec<Option<&Bitmask>> = Vec::with_capacity(table.cols.len());
    for col in &table.cols {
        match &col.array {
            Array::NumericArray(arr) => null_masks.push(arr.null_mask()),
            Array::BooleanArray(arr) => null_masks.push(arr.null_mask.as_ref()),
            Array::TextArray(TextArray::String32(arr)) => null_masks.push(arr.null_mask.as_ref()),
            Array::TextArray(TextArray::Categorical32(arr)) => {
                null_masks.push(arr.null_mask.as_ref())
            }
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => null_masks.push(arr.null_mask.as_ref()),
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
                    minarrow::TemporalArray::Datetime32(arr) => arr.null_mask.as_ref(),
                    minarrow::TemporalArray::Datetime64(arr) => arr.null_mask.as_ref(),
                    minarrow::TemporalArray::Null => None,
                };
                null_masks.push(null_mask)
            }
            _ => null_masks.push(None),
        }
    }

    // Size the scratch buffer up front from per-column width estimates
    // so the row loop doesn't trigger Vec reallocation.
    let est_row: usize = table
        .cols
        .iter()
        .map(|c| estimate_cell_width(&c.array, n_rows) + 1)
        .sum::<usize>();
    let header_est = if write_header {
        table
            .cols
            .iter()
            .map(|c| c.field.name.len() + 1)
            .sum::<usize>()
    } else {
        0
    };
    let mut out: Vec<u8> = Vec::with_capacity(header_est + est_row * n_rows + 16);

    // Header
    if write_header {
        for (i, col) in table.cols.iter().enumerate() {
            if i > 0 {
                out.push(delimiter);
            }
            append_csv_string(&mut out, col.field.name.as_bytes(), delimiter, quote);
        }
        out.push(b'\n');
    }

    let null_bytes = null_repr.as_bytes();
    let mut itoa_buf = super::int_ascii::Buffer::new();
    let mut ryu_buf = ryu::Buffer::new();

    for row in 0..n_rows {
        for (col_idx, col) in table.cols.iter().enumerate() {
            if col_idx > 0 {
                out.push(delimiter);
            }
            // Optimise for the common case of no nulls in the column.
            let is_null = if col.null_count == 0 {
                false
            } else {
                match null_masks[col_idx] {
                    Some(mask) => !mask.get(row),
                    None => false,
                }
            };
            if is_null {
                out.extend_from_slice(null_bytes);
                continue;
            }
            match &col.array {
                Array::NumericArray(n) => match n {
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    NumericArray::Int32(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    NumericArray::Int64(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    NumericArray::UInt32(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    NumericArray::UInt64(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    NumericArray::Float32(arr) => {
                        out.extend_from_slice(ryu_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    NumericArray::Float64(arr) => {
                        out.extend_from_slice(ryu_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    #[cfg(feature = "decimal")]
                    NumericArray::Decimal32(arr) => {
                        write_decimal_cell(&mut out, arr.data.as_ref()[row] as i128, arr.scale);
                    }
                    #[cfg(feature = "decimal")]
                    NumericArray::Decimal64(arr) => {
                        write_decimal_cell(&mut out, arr.data.as_ref()[row] as i128, arr.scale);
                    }
                    #[cfg(feature = "decimal")]
                    NumericArray::Decimal128(arr) => {
                        write_decimal_cell(&mut out, arr.data.as_ref()[row], arr.scale);
                    }
                    _ => {
                        out.extend_from_slice(b"<unsupported>");
                    }
                },
                Array::BooleanArray(arr) => {
                    out.extend_from_slice(if arr.data.get(row) { b"true" } else { b"false" });
                }
                Array::TextArray(TextArray::String32(arr)) => {
                    append_str_cell(
                        &mut out,
                        arr.data.as_ref(),
                        arr.offsets.as_ref(),
                        row,
                        delimiter,
                        quote,
                    );
                }
                #[cfg(feature = "large_string")]
                Array::TextArray(TextArray::String64(arr)) => {
                    append_str_cell(
                        &mut out,
                        arr.data.as_ref(),
                        arr.offsets.as_ref(),
                        row,
                        delimiter,
                        quote,
                    );
                }
                Array::TextArray(TextArray::Categorical32(arr)) => {
                    // dictionary lookup - always clean UTF-8
                    let idx = arr.data.as_ref()[row] as usize;
                    let s = arr
                        .unique_values()
                        .get(idx)
                        .map(String::as_str)
                        .unwrap_or("<invalid>");
                    append_csv_string(&mut out, s.as_bytes(), delimiter, quote);
                }
                #[cfg(feature = "default_categorical_8")]
                Array::TextArray(TextArray::Categorical8(arr)) => {
                    let idx = arr.data.as_ref()[row] as usize;
                    let s = arr
                        .unique_values()
                        .get(idx)
                        .map(String::as_str)
                        .unwrap_or("<invalid>");
                    append_csv_string(&mut out, s.as_bytes(), delimiter, quote);
                }
                #[cfg(feature = "extended_categorical")]
                Array::TextArray(TextArray::Categorical16(arr)) => {
                    let idx = arr.data.as_ref()[row] as usize;
                    let s = arr
                        .unique_values()
                        .get(idx)
                        .map(String::as_str)
                        .unwrap_or("<invalid>");
                    append_csv_string(&mut out, s.as_bytes(), delimiter, quote);
                }
                #[cfg(feature = "extended_categorical")]
                Array::TextArray(TextArray::Categorical64(arr)) => {
                    let idx = arr.data.as_ref()[row] as usize;
                    let s = arr
                        .unique_values()
                        .get(idx)
                        .map(String::as_str)
                        .unwrap_or("<invalid>");
                    append_csv_string(&mut out, s.as_bytes(), delimiter, quote);
                }
                #[cfg(feature = "datetime")]
                Array::TemporalArray(temp) => match temp {
                    minarrow::TemporalArray::Datetime32(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    minarrow::TemporalArray::Datetime64(arr) => {
                        out.extend_from_slice(itoa_buf.format(arr.data.as_ref()[row]).as_bytes());
                    }
                    minarrow::TemporalArray::Null => {
                        out.extend_from_slice(b"<null_temporal>");
                    }
                },
                _ => {
                    out.extend_from_slice(b"<unsupported>");
                }
            }
        }
        out.push(b'\n');
    }

    writer.write_all(&out)
}

/// Serialises a *Minarrow* `SuperTable` (i.e., *Arrow* multiple *RecordBatches*) as a CSV, with all batches concatenated.
/// Each batch will write headers only if `write_header` is set and is the first batch.
/// Use for multi-batch output.
///
/// # Arguments
/// - `supertable`: The SuperTable to encode.
/// - `mut writer`: Any io::Write.
/// - `options`: Encoding options.
///
/// # Errors
/// Returns any io error from the writer.
pub fn encode_supertable_csv<W: Write>(
    supertable: &SuperTable,
    mut writer: W,
    options: &CsvEncodeOptions,
) -> io::Result<()> {
    let mut opts = options.clone();
    for (i, batch) in supertable.batches.iter().enumerate() {
        opts.write_header = if i == 0 { options.write_header } else { false };
        encode_table_csv(batch, &mut writer, &opts)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use minarrow::{
        Array, ArrowType, Bitmask, Buffer, Field, FieldArray, NumericArray, Table, TextArray, vec64,
    };

    use super::*;

    fn make_test_table() -> Table {
        let int_col = FieldArray {
            field: Field {
                name: "ints".to_string(),
                dtype: minarrow::ArrowType::Int32,
                nullable: true, // Change to true to allow nulls
                metadata: Default::default(),
            }
            .into(),
            array: Array::NumericArray(NumericArray::Int32(
                minarrow::IntegerArray {
                    data: Buffer::from(vec64![1, 2, 3, 4]),
                    null_mask: Some(Bitmask::from_bools(&[true, false, true, true])), // row 1 is null
                }
                .into(),
            )),
            null_count: 1,
        };
        let str_col = FieldArray {
            field: Field {
                name: "strings".to_string(),
                dtype: minarrow::ArrowType::String,
                nullable: true,
                metadata: Default::default(),
            }
            .into(),
            array: Array::TextArray(TextArray::String32(
                minarrow::StringArray {
                    offsets: Buffer::from(vec64![0u32, 5, 9, 14, 18]),
                    data: Buffer::from_vec64("helloabcdworldrust".as_bytes().into()),
                    null_mask: Some(Bitmask::from_bools(&[true, false, true, true])),
                }
                .into(),
            )),
            null_count: 1,
        };
        Table {
            name: "test".to_string(),
            cols: vec![int_col, str_col],
            n_rows: 4,
            ..Default::default()
        }
    }

    #[test]
    fn test_encode_table_csv_basic() {
        let table = make_test_table();
        let mut out = Vec::new();
        let opts = CsvEncodeOptions::default();
        encode_table_csv(&table, &mut out, &opts).unwrap();
        let csv = String::from_utf8(out).unwrap();
        assert!(csv.contains("ints,strings"));
        assert!(csv.contains("hello"));
        assert!(csv.contains("\n,\n"));
    }

    #[test]
    fn test_encode_table_csv_custom_delim() {
        let table = make_test_table();
        let mut out = Vec::new();
        let mut opts = CsvEncodeOptions::default();
        opts.delimiter = b'\t';
        encode_table_csv(&table, &mut out, &opts).unwrap();
        let csv = String::from_utf8(out).unwrap();
        assert!(csv.contains("\t"));
    }

    #[test]
    fn encode_quotes_field_with_delimiter() {
        use minarrow::{Array, Buffer, Field, FieldArray, NumericArray, Table, TextArray, vec64};

        use crate::models::encoders::csv::{CsvEncodeOptions, encode_table_csv};
        let col1 = FieldArray {
            field: Field::new("id", minarrow::ArrowType::Int32, false, None).into(),
            array: Array::NumericArray(NumericArray::Int32(
                minarrow::IntegerArray {
                    data: Buffer::from(vec64![1]),
                    null_mask: None,
                }
                .into(),
            )),
            null_count: 0,
        };

        let col2_str = "needs,quotes"; // contains delimiter
        let col2 = FieldArray {
            field: Field::new("txt", minarrow::ArrowType::String, false, None).into(),
            array: Array::TextArray(TextArray::String32(
                minarrow::StringArray {
                    offsets: Buffer::from(vec64![0u32, col2_str.len() as u32]),
                    data: Buffer::from_vec64(col2_str.as_bytes().into()),
                    null_mask: None,
                }
                .into(),
            )),
            null_count: 0,
        };

        let tbl = Table {
            name: "".into(),
            cols: vec![col1, col2],
            n_rows: 1,
            ..Default::default()
        };
        let mut out = Vec::new();
        encode_table_csv(&tbl, &mut out, &CsvEncodeOptions::default()).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\"needs,quotes\"")); // quoted and preserved
    }

    #[test]
    fn encode_doubles_embedded_quotes() {
        let col = FieldArray {
            field: Field::new("txt", minarrow::ArrowType::String, false, None).into(),
            array: Array::TextArray(TextArray::String32(
                minarrow::StringArray {
                    offsets: Buffer::from(vec64![0u32, 9]),
                    data: Buffer::from_vec64(b"he\"llo\",x".to_vec().into()),
                    null_mask: None,
                }
                .into(),
            )),
            null_count: 0,
        };
        let tbl = Table {
            name: "".into(),
            cols: vec![col],
            n_rows: 1,
            ..Default::default()
        };
        let mut out = Vec::new();
        encode_table_csv(&tbl, &mut out, &CsvEncodeOptions::default()).unwrap();
        let s = String::from_utf8(out).unwrap();
        // Embedded `"` must be doubled and the whole field quoted.
        assert!(s.contains("\"he\"\"llo\"\",x\""), "got: {s}");
    }

    #[test]
    fn encode_decode_custom_null() {
        use crate::models::decoders::csv::*;
        use crate::models::encoders::csv::*;
        let mut opts = CsvEncodeOptions::default();
        opts.null_repr = "NULL";
        // build a 1-row table with a null value in the first column
        use minarrow::{
            Array, ArrowType, Bitmask, Field, FieldArray, IntegerArray, NumericArray, Table,
        };
        use std::sync::Arc;

        let field = Field {
            name: "int32".to_string(),
            dtype: ArrowType::Int32,
            nullable: true,
            metadata: Default::default(),
        };

        let null_mask = Bitmask::from_bytes(&[0b00000000], 1); // First bit is 0 = null
        let array = Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(minarrow::Vec64::from_slice(&[42i32])), // Value doesn't matter since it's null
            null_mask: Some(null_mask),
        })));

        let col = FieldArray::new(field, array);
        let tbl = Table {
            cols: vec![col],
            n_rows: 1,
            name: "test_null".to_string(),
            ..Default::default()
        };
        let mut buf = Vec::new();
        encode_table_csv(&tbl, &mut buf, &opts).unwrap();

        // decode with matching null option
        let mut dec = CsvDecodeOptions::default();
        dec.nulls = vec!["NULL"];
        let parsed = decode_csv(std::io::Cursor::new(&buf), &dec).unwrap();
        assert_eq!(parsed.cols[0].null_count, 1);
    }

    #[test]
    fn test_csv_decoder_mask_semantics() {
        // Verify CSV decoder uses Arrow-semantic null masks: 1 = valid, 0 = null.
        use crate::models::decoders::csv::*;
        use minarrow::MaskedArray;

        let csv = b"col\nvalid\n\nvalid2\n"; // 2 valid, 1 null (empty string)
        let opts = CsvDecodeOptions::default();
        let table = decode_csv(std::io::Cursor::new(csv.as_ref()), &opts).unwrap();

        assert_eq!(table.cols[0].null_count, 1);

        let Array::TextArray(TextArray::String32(arr)) = &table.cols[0].array else {
            panic!("Expected String32 array");
        };
        let mask = arr.null_mask.as_ref().unwrap();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.count_ones(), 2, "2 valid rows");
        assert_eq!(mask.count_zeros(), 1, "1 null row");
        assert_eq!(arr.null_count(), 1);
        // [valid, null, valid] = [true, false, true]
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert!(mask.get(2));
    }

    #[test]
    fn test_null_mask_interpretation_mixed_nulls() {
        // Test with a mix of null and valid values to ensure mask interpretation is correct
        use minarrow::{
            Array, ArrowType, Bitmask, Field, FieldArray, IntegerArray, NumericArray, Table,
        };
        use std::sync::Arc;

        let field = Field {
            name: "mixed_nulls".to_string(),
            dtype: ArrowType::Int32,
            nullable: true,
            metadata: Default::default(),
        };

        // Create mask: [valid, null, valid, null] = [1, 0, 1, 0] = 0b0101 = 5
        let null_mask = Bitmask::from_bytes(&[0b00000101], 4);
        let array = Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(minarrow::Vec64::from_slice(&[10i32, 999i32, 30i32, 999i32])),
            null_mask: Some(null_mask),
        })));

        let col = FieldArray::new(field, array);
        let tbl = Table {
            cols: vec![col],
            n_rows: 4,
            name: "mixed_null_test".to_string(),
            ..Default::default()
        };

        // Verify null_count is correct
        assert_eq!(tbl.cols[0].null_count, 2, "Expected 2 nulls");

        let mut opts = CsvEncodeOptions::default();
        opts.null_repr = "NULL";
        let mut buf = Vec::new();
        encode_table_csv(&tbl, &mut buf, &opts).unwrap();
        let csv_output = String::from_utf8(buf).unwrap();

        // Should be: "mixed_nulls\n10\nNULL\n30\nNULL\n"
        assert_eq!(csv_output, "mixed_nulls\n10\nNULL\n30\nNULL\n");
    }

    #[test]
    fn test_null_mask_interpretation_all_nulls() {
        // Test with all nulls to verify mask interpretation
        use minarrow::{
            Array, ArrowType, Bitmask, Field, FieldArray, IntegerArray, NumericArray, Table,
        };
        use std::sync::Arc;

        let field = Field {
            name: "all_nulls".to_string(),
            dtype: ArrowType::Int32,
            nullable: true,
            metadata: Default::default(),
        };

        // Create mask with all nulls: [0, 0, 0] = 0b000 = 0
        let null_mask = Bitmask::from_bytes(&[0b00000000], 3);
        let array = Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(minarrow::Vec64::from_slice(&[999i32, 999i32, 999i32])),
            null_mask: Some(null_mask),
        })));

        let col = FieldArray::new(field, array);
        let tbl = Table {
            cols: vec![col],
            n_rows: 3,
            name: "all_null_test".to_string(),
            ..Default::default()
        };

        // Verify null_count is correct
        assert_eq!(tbl.cols[0].null_count, 3, "Expected 3 nulls");

        let mut opts = CsvEncodeOptions::default();
        opts.null_repr = "NULL";
        let mut buf = Vec::new();
        encode_table_csv(&tbl, &mut buf, &opts).unwrap();
        let csv_output = String::from_utf8(buf).unwrap();

        // Should be: "all_nulls\nNULL\nNULL\nNULL\n"
        assert_eq!(csv_output, "all_nulls\nNULL\nNULL\nNULL\n");
    }

    #[test]
    fn categorical_roundtrip() {
        use crate::models::decoders::csv::*;
        use crate::models::encoders::csv::*;
        let csv = b"id,fruit\n1,apple\n2,banana\n3,apple\n";
        let mut opts = CsvDecodeOptions::default();
        opts.categorical_cols.insert("fruit".into());
        let tbl = decode_csv(std::io::Cursor::new(csv.as_ref()), &opts).unwrap();

        // ensure dictionary detected
        assert!(matches!(tbl.cols[1].field.dtype, ArrowType::Dictionary(_)));

        let mut out = Vec::new();
        encode_table_csv(&tbl, &mut out, &CsvEncodeOptions::default()).unwrap();
        let out_str = String::from_utf8(out).unwrap();
        assert!(out_str.contains("apple"));
        assert!(out_str.contains("banana"));
    }
}
