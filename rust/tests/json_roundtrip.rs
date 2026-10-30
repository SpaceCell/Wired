// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Integration tests for JSON format support: array-of-objects and NDJSON.

#![cfg(feature = "json")]

use std::io::Cursor;
use std::sync::Arc;

use lightstream::models::decoders::json::{JsonDecodeOptions, decode_json, decode_ndjson};
use lightstream::models::encoders::json::{JsonEncodeOptions, JsonFormat, encode_table_json};
use lightstream::models::readers::json::JsonReader;
use lightstream::models::writers::json::JsonWriter;
use minarrow::{
    Array, ArrowType, Bitmask, Buffer, Field, FieldArray, FloatArray, IntegerArray, NumericArray,
    StringArray, Table, TextArray, Vec64, vec64,
};
use simd_json::prelude::{ValueAsArray, ValueAsObject, ValueAsScalar};

fn parse(bytes: Vec<u8>) -> simd_json::OwnedValue {
    let mut buf = bytes;
    simd_json::to_owned_value(&mut buf).unwrap()
}

fn mixed_schema() -> Vec<Field> {
    vec![
        Field::new("id", ArrowType::Int32, false, None),
        Field::new("score", ArrowType::Float64, false, None),
        Field::new("name", ArrowType::String, true, None),
    ]
}

fn make_mixed_table() -> Table {
    let id = FieldArray {
        field: Field::new("id", ArrowType::Int32, false, None).into(),
        array: Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::<i32>::from_slice(&[10, 20, 30])),
            null_mask: None,
        }))),
        null_count: 0,
    };
    let score = FieldArray {
        field: Field::new("score", ArrowType::Float64, false, None).into(),
        array: Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: Buffer::from(Vec64::<f64>::from_slice(&[1.5, 2.5, 3.5])),
            null_mask: None,
        }))),
        null_count: 0,
    };
    let name = FieldArray {
        field: Field::new("name", ArrowType::String, true, None).into(),
        array: Array::TextArray(TextArray::String32(Arc::new(StringArray {
            offsets: Buffer::from(vec64![0u32, 5, 5, 12]),
            data: Buffer::from_vec64(b"alicecharlie".to_vec().into()),
            null_mask: Some(Bitmask::from_bools(&[true, false, true])),
        }))),
        null_count: 1,
    };
    Table::new("mixed".to_string(), Some(vec![id, score, name]))
}

#[test]
fn roundtrip_array_of_objects() {
    let table = make_mixed_table();

    let mut out = Vec::new();
    encode_table_json(&table, &mut out, &JsonEncodeOptions::default()).unwrap();

    let opts = JsonDecodeOptions {
        schema: Some(mixed_schema()),
        ..Default::default()
    };
    let decoded = decode_json(Cursor::new(&out[..]), &opts).unwrap();

    assert_eq!(decoded.n_rows, 3);
    assert_eq!(decoded.cols.len(), 3);

    match &decoded.cols[0].array {
        Array::NumericArray(NumericArray::Int32(arr)) => {
            assert_eq!(arr.data.as_ref(), &[10, 20, 30]);
        }
        _ => panic!("expected Int32"),
    }
    match &decoded.cols[1].array {
        Array::NumericArray(NumericArray::Float64(arr)) => {
            assert_eq!(arr.data.as_ref(), &[1.5, 2.5, 3.5]);
        }
        _ => panic!("expected Float64"),
    }
    assert_eq!(decoded.cols[2].null_count, 1);
}

#[test]
fn roundtrip_ndjson() {
    let table = make_mixed_table();

    let opts = JsonEncodeOptions {
        format: JsonFormat::Ndjson,
        ..Default::default()
    };
    let mut out = Vec::new();
    encode_table_json(&table, &mut out, &opts).unwrap();

    let line_count = out.iter().filter(|&&b| b == b'\n').count();
    assert_eq!(line_count, 3);

    let dec_opts = JsonDecodeOptions {
        schema: Some(mixed_schema()),
        ..Default::default()
    };
    let decoded = decode_ndjson(Cursor::new(&out[..]), &dec_opts).unwrap();
    assert_eq!(decoded.n_rows, 3);
    assert_eq!(decoded.cols.len(), 3);
}

#[test]
fn roundtrip_via_reader_writer_ndjson() {
    let table = make_mixed_table();

    let opts = JsonEncodeOptions {
        format: JsonFormat::Ndjson,
        ..Default::default()
    };
    let mut writer = JsonWriter::new(Vec64::<u8>::new(), opts);
    writer.write_table(&table).unwrap();
    let bytes = writer.into_inner();

    let dec_opts = JsonDecodeOptions {
        schema: Some(mixed_schema()),
        ..Default::default()
    };
    let reader = JsonReader::<std::io::BufReader<&[u8]>>::from_slice(
        &bytes,
        JsonFormat::Ndjson,
        dec_opts,
        10,
    );
    let out = reader.load_table().unwrap();
    assert_eq!(out.n_rows, 3);
}

#[test]
fn writer_default_output_is_64_byte_aligned() {
    let table = make_mixed_table();
    let mut writer = JsonWriter::new(Vec64::<u8>::new(), JsonEncodeOptions::default());
    writer.write_table(&table).unwrap();
    let bytes = writer.into_inner();
    assert_eq!(
        bytes.as_ptr() as usize % 64,
        0,
        "Vec64 output must be 64-byte aligned"
    );
}

#[test]
fn array_omit_nulls_produces_compact_objects() {
    let table = make_mixed_table();
    let opts = JsonEncodeOptions {
        format: JsonFormat::Ndjson,
        include_nulls: false,
        ..Default::default()
    };
    let mut writer = JsonWriter::new(Vec64::<u8>::new(), opts);
    writer.write_table(&table).unwrap();
    let bytes = writer.into_inner();
    let s = std::str::from_utf8(&bytes).unwrap().to_string();

    let lines: Vec<_> = s.lines().collect();
    let parsed_row1 = parse(lines[1].as_bytes().to_vec());
    let obj = parsed_row1.as_object().unwrap();
    assert!(!obj.contains_key("name"));
    assert_eq!(obj.get("id").and_then(|v| v.as_i64()), Some(20));
    assert_eq!(obj.get("score").and_then(|v| v.as_f64()), Some(2.5));
}

#[test]
fn ndjson_batched_reader_end_to_end() {
    let ids = FieldArray {
        field: Field::new("id", ArrowType::Int32, false, None).into(),
        array: Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::<i32>::from_slice(&[1, 2, 3, 4, 5])),
            null_mask: None,
        }))),
        null_count: 0,
    };
    let table = Table::new("ids".to_string(), Some(vec![ids]));

    let opts = JsonEncodeOptions {
        format: JsonFormat::Ndjson,
        ..Default::default()
    };
    let mut writer = JsonWriter::new(Vec64::<u8>::new(), opts);
    writer.write_table(&table).unwrap();
    let bytes = writer.into_inner();

    let dec_opts = JsonDecodeOptions {
        schema: Some(vec![Field::new("id", ArrowType::Int32, false, None)]),
        ..Default::default()
    };
    let mut reader = JsonReader::<std::io::BufReader<&[u8]>>::from_slice(
        &bytes,
        JsonFormat::Ndjson,
        dec_opts,
        2,
    );

    let b1 = reader.next_batch().unwrap().unwrap();
    assert_eq!(b1.n_rows, 2);
    let b2 = reader.next_batch().unwrap().unwrap();
    assert_eq!(b2.n_rows, 2);
    let b3 = reader.next_batch().unwrap().unwrap();
    assert_eq!(b3.n_rows, 1);
    let b4 = reader.next_batch().unwrap();
    assert!(b4.is_none());
}

#[test]
fn pretty_output_is_parseable() {
    let table = make_mixed_table();
    let opts = JsonEncodeOptions {
        format: JsonFormat::Array { pretty: true },
        ..Default::default()
    };
    let mut writer = JsonWriter::new(Vec64::<u8>::new(), opts);
    writer.write_table(&table).unwrap();
    let bytes = writer.into_inner();
    let s = std::str::from_utf8(&bytes).unwrap();
    assert!(s.contains('\n'));
    let v = parse(bytes.0.into_iter().collect());
    assert_eq!(v.as_array().unwrap().len(), 3);
}

#[test]
fn explicit_schema_controls_types() {
    let json = br#"[{"x":"001"},{"x":"002"},{"x":"003"}]"#;
    let schema = vec![Field::new("x", ArrowType::String, false, None)];
    let opts = JsonDecodeOptions {
        schema: Some(schema),
        ..Default::default()
    };
    let tbl = decode_json(Cursor::new(&json[..]), &opts).unwrap();
    assert!(matches!(tbl.cols[0].field.dtype, ArrowType::String));
    assert_eq!(tbl.n_rows, 3);
}

#[cfg(feature = "datetime")]
#[test]
fn date_columns_roundtrip_with_their_dt() {
    use minarrow::{DatetimeArray, Dt, TemporalArray};

    let mask = Bitmask::from_bools(&[true, false, true]);
    let date32 = FieldArray::new(
        Field::new("d32", ArrowType::DateTime(Dt::Date32), true, None),
        Array::from_datetime_i32(DatetimeArray::from_vec64(
            Vec64::from_slice(&[19_000, 0, -5]),
            Some(mask.clone()),
            Dt::Date32,
        )),
    );
    let date64 = FieldArray::new(
        Field::new("d64", ArrowType::DateTime(Dt::Date64), true, None),
        Array::from_datetime_i64(DatetimeArray::from_vec64(
            Vec64::from_slice(&[1_641_600_000_000, 0, -86_400_000]),
            Some(mask),
            Dt::Date64,
        )),
    );
    let table = Table::new("dates".to_string(), Some(vec![date32, date64]));
    let schema: Vec<Field> = table.cols.iter().map(|c| c.field.as_ref().clone()).collect();

    let mut out = Vec::new();
    encode_table_json(&table, &mut out, &JsonEncodeOptions::default()).unwrap();
    let opts = JsonDecodeOptions {
        schema: Some(schema),
        ..Default::default()
    };
    let decoded = decode_json(Cursor::new(&out[..]), &opts).unwrap();

    assert_eq!(decoded.n_rows, 3);
    assert_eq!(decoded.cols[0].field.dtype, ArrowType::DateTime(Dt::Date32));
    assert_eq!(decoded.cols[1].field.dtype, ArrowType::DateTime(Dt::Date64));
    match &decoded.cols[0].array {
        Array::TemporalArray(TemporalArray::Datetime32(a)) => {
            assert_eq!(a.dtype, Dt::Date32);
            assert_eq!(a.data.as_ref()[0], 19_000);
            assert_eq!(a.data.as_ref()[2], -5);
            assert!(a.null_mask.as_ref().is_some_and(|m| !m.get(1)));
        }
        other => panic!("d32 decoded as {other:?}"),
    }
    match &decoded.cols[1].array {
        Array::TemporalArray(TemporalArray::Datetime64(a)) => {
            assert_eq!(a.dtype, Dt::Date64);
            assert_eq!(a.data.as_ref()[0], 1_641_600_000_000);
            assert_eq!(a.data.as_ref()[2], -86_400_000);
            assert!(a.null_mask.as_ref().is_some_and(|m| !m.get(1)));
        }
        other => panic!("d64 decoded as {other:?}"),
    }
}
