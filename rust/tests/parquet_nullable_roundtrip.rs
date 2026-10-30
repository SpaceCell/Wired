// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Parquet write-then-read over every supported column type with nulls.
//!
//! Each column spans several data pages with nulls placed on both sides of
//! every page boundary, so the value sections hold non-null entries only
//! and the reader has to scatter them back against the definition levels
//! page by page.
//!
//! The types that come back are the ones the Parquet schema names, which
//! is not always the Arrow type that went in: categoricals are UTF8
//! strings, Date64 is a DATE day count, seconds become milliseconds, and
//! TIME columns take the storage width their unit requires.

#[cfg(feature = "parquet")]
mod parquet_nullable_roundtrip_tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use lightstream::compression::Compression;
    use lightstream::models::readers::parquet::load_parquet_table;
    use lightstream::models::writers::parquet::{PARQUET_PAGE_CHUNK_SIZE, write_parquet_table};
    use minarrow::ffi::arrow_dtype::CategoricalIndexType;
    #[cfg(feature = "datetime")]
    use minarrow::Dt;
    use minarrow::{
        Array, ArrowType, Bitmask, BooleanArray, CategoricalArray, Field, FieldArray, FloatArray,
        IntegerArray, MaskedArray, NumericArray, StringArray, Table, TextArray, Vec64,
    };
    #[cfg(feature = "datetime")]
    use minarrow::{DatetimeArray, TemporalArray, TimeUnit};
    #[cfg(feature = "decimal")]
    use minarrow::DecimalArray;

    /// Rows per column: two full pages plus a partial third page.
    const N_ROWS: usize = 2 * PARQUET_PAGE_CHUNK_SIZE + 13;

    /// Every seventh row from row three is null, plus the rows either side of
    /// each page boundary.
    fn is_valid(i: usize) -> bool {
        let boundary = [
            PARQUET_PAGE_CHUNK_SIZE - 1,
            PARQUET_PAGE_CHUNK_SIZE,
            2 * PARQUET_PAGE_CHUNK_SIZE - 1,
            2 * PARQUET_PAGE_CHUNK_SIZE,
        ];
        i % 7 != 3 && !boundary.contains(&i)
    }

    fn null_mask() -> Bitmask {
        let bits: Vec<bool> = (0..N_ROWS).map(is_valid).collect();
        Bitmask::from_bools(&bits)
    }

    fn expected<T>(value: impl Fn(usize) -> T) -> Vec<Option<T>> {
        (0..N_ROWS)
            .map(|i| is_valid(i).then(|| value(i)))
            .collect()
    }

    fn int32_value(i: usize) -> i32 {
        i as i32 - 1000
    }
    fn uint32_value(i: usize) -> u32 {
        i as u32 * 3
    }
    fn int64_value(i: usize) -> i64 {
        -(i as i64) * 7
    }
    fn uint64_value(i: usize) -> u64 {
        i as u64 + (1 << 40)
    }
    fn float32_value(i: usize) -> f32 {
        i as f32 * 0.5
    }
    fn float64_value(i: usize) -> f64 {
        i as f64 * -0.25
    }
    fn bool_value(i: usize) -> bool {
        i % 3 == 0
    }
    fn string_value(i: usize) -> String {
        format!("s{}", i % 101)
    }
    fn category_value(i: usize) -> &'static str {
        ["red", "green", "blue"][i % 3]
    }
    #[cfg(feature = "datetime")]
    fn date32_value(i: usize) -> i32 {
        i as i32 * 3
    }
    /// Milliseconds with a sub-day remainder, so the DATE day count is the
    /// truncated quotient.
    #[cfg(feature = "datetime")]
    fn date64_value(i: usize) -> i64 {
        i as i64 * 86_400_000 + 12_345
    }
    #[cfg(feature = "datetime")]
    fn timestamp_value(i: usize) -> i64 {
        1_700_000_000_000 + i as i64 * 1_001
    }
    #[cfg(feature = "datetime")]
    fn time_value(i: usize) -> i64 {
        (i % 86_400) as i64 * 7
    }
    #[cfg(feature = "decimal")]
    fn decimal32_value(i: usize) -> i32 {
        i as i32 * 125 - 50_000
    }
    #[cfg(feature = "decimal")]
    fn decimal64_value(i: usize) -> i64 {
        i as i64 * 1_000_003 - 7
    }
    #[cfg(feature = "decimal")]
    fn decimal128_value(i: usize) -> i128 {
        (i as i128 - 5) * 1_000_000_000_000_000_000_000
    }

    fn column(name: &str, dtype: ArrowType, array: Array) -> FieldArray {
        FieldArray::new(Field::new(name, dtype, true, None), array)
    }

    #[cfg(feature = "datetime")]
    fn temporal32(name: &str, dt: Dt, value: fn(usize) -> i32) -> FieldArray {
        column(
            name,
            ArrowType::DateTime(dt.clone()),
            Array::from_datetime_i32(DatetimeArray::from_vec64(
                (0..N_ROWS).map(value).collect::<Vec64<_>>(),
                Some(null_mask()),
                dt,
            )),
        )
    }

    #[cfg(feature = "datetime")]
    fn temporal64(name: &str, dt: Dt, value: fn(usize) -> i64) -> FieldArray {
        column(
            name,
            ArrowType::DateTime(dt.clone()),
            Array::from_datetime_i64(DatetimeArray::from_vec64(
                (0..N_ROWS).map(value).collect::<Vec64<_>>(),
                Some(null_mask()),
                dt,
            )),
        )
    }

    fn all_types_table() -> Table {
        let mask = null_mask();
        let strings: Vec<String> = (0..N_ROWS).map(string_value).collect();
        let categories: Vec<&str> = (0..N_ROWS).map(category_value).collect();

        let mut cols = vec![
            column(
                "int32",
                ArrowType::Int32,
                Array::from_int32(IntegerArray::from_vec64(
                    (0..N_ROWS).map(int32_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                )),
            ),
            column(
                "uint32",
                ArrowType::UInt32,
                Array::from_uint32(IntegerArray::from_vec64(
                    (0..N_ROWS).map(uint32_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                )),
            ),
            column(
                "int64",
                ArrowType::Int64,
                Array::from_int64(IntegerArray::from_vec64(
                    (0..N_ROWS).map(int64_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                )),
            ),
            column(
                "uint64",
                ArrowType::UInt64,
                Array::from_uint64(IntegerArray::from_vec64(
                    (0..N_ROWS).map(uint64_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                )),
            ),
            column(
                "float32",
                ArrowType::Float32,
                Array::from_float32(FloatArray::from_vec64(
                    (0..N_ROWS).map(float32_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                )),
            ),
            column(
                "float64",
                ArrowType::Float64,
                Array::from_float64(FloatArray::from_vec64(
                    (0..N_ROWS).map(float64_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                )),
            ),
            column(
                "bool",
                ArrowType::Boolean,
                Array::from_bool(BooleanArray::new(
                    Bitmask::from_bools(&(0..N_ROWS).map(bool_value).collect::<Vec<_>>()),
                    Some(mask.clone()),
                )),
            ),
            column(
                "string",
                ArrowType::String,
                Array::from_string32(StringArray::from_vec(
                    strings.iter().map(String::as_str).collect(),
                    Some(mask.clone()),
                )),
            ),
        ];

        #[cfg(feature = "default_categorical_8")]
        cols.push(column(
            "category",
            ArrowType::Dictionary(CategoricalIndexType::UInt8),
            Array::from_categorical8(CategoricalArray::<u8>::from_vec(
                categories,
                Some(mask.clone()),
            )),
        ));
        #[cfg(not(feature = "default_categorical_8"))]
        cols.push(column(
            "category",
            ArrowType::Dictionary(CategoricalIndexType::UInt32),
            Array::from_categorical32(CategoricalArray::<u32>::from_vec(
                categories,
                Some(mask.clone()),
            )),
        ));

        #[cfg(feature = "datetime")]
        {
            cols.push(temporal32("date32", Dt::Date32, date32_value));
            cols.push(temporal64("date64", Dt::Date64, date64_value));
            cols.push(temporal64(
                "ts_s",
                Dt::Timestamp(TimeUnit::Seconds, None),
                timestamp_value,
            ));
            cols.push(temporal64(
                "ts_ms",
                Dt::Timestamp(TimeUnit::Milliseconds, None),
                timestamp_value,
            ));
            cols.push(temporal64(
                "ts_us",
                Dt::Timestamp(TimeUnit::Microseconds, None),
                timestamp_value,
            ));
            cols.push(temporal64(
                "ts_ns",
                Dt::Timestamp(TimeUnit::Nanoseconds, None),
                timestamp_value,
            ));
            cols.push(temporal64(
                "ts_us_utc",
                Dt::Timestamp(TimeUnit::Microseconds, Some("UTC".to_string())),
                timestamp_value,
            ));
            cols.push(temporal32(
                "time32_ms",
                Dt::Time32(TimeUnit::Milliseconds),
                |i| time_value(i) as i32,
            ));
            cols.push(temporal32(
                "time32_us",
                Dt::Time32(TimeUnit::Microseconds),
                |i| time_value(i) as i32,
            ));
            cols.push(temporal64(
                "time64_ms",
                Dt::Time64(TimeUnit::Milliseconds),
                time_value,
            ));
            cols.push(temporal64(
                "time64_ns",
                Dt::Time64(TimeUnit::Nanoseconds),
                time_value,
            ));
            cols.push(temporal32(
                "time32_s",
                Dt::Time32(TimeUnit::Seconds),
                |i| time_value(i) as i32,
            ));
            cols.push(temporal32(
                "time32_ns",
                Dt::Time32(TimeUnit::Nanoseconds),
                |i| time_value(i) as i32,
            ));
            cols.push(temporal64(
                "time64_s",
                Dt::Time64(TimeUnit::Seconds),
                time_value,
            ));
            cols.push(temporal64(
                "time64_us",
                Dt::Time64(TimeUnit::Microseconds),
                time_value,
            ));
        }

        #[cfg(feature = "decimal")]
        {
            cols.push(column(
                "dec32",
                ArrowType::Decimal32(7, 2),
                Array::from_decimal32(DecimalArray::from_vec64(
                    (0..N_ROWS).map(decimal32_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                    7,
                    2,
                )),
            ));
            cols.push(column(
                "dec64",
                ArrowType::Decimal64(18, 4),
                Array::from_decimal64(DecimalArray::from_vec64(
                    (0..N_ROWS).map(decimal64_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                    18,
                    4,
                )),
            ));
            cols.push(column(
                "dec128",
                ArrowType::Decimal128(38, 6),
                Array::from_decimal128(DecimalArray::from_vec64(
                    (0..N_ROWS).map(decimal128_value).collect::<Vec64<_>>(),
                    Some(mask.clone()),
                    38,
                    6,
                )),
            ));
        }

        Table::new("all_types".to_string(), Some(cols))
    }

    fn roundtrip(table: &Table, compression: Option<Compression>) -> Table {
        let mut buf = Cursor::new(Vec::new());
        write_parquet_table(table, &mut buf, compression).expect("write");
        buf.seek(SeekFrom::Start(0)).unwrap();
        load_parquet_table(&mut buf).expect("read")
    }

    fn col<'a>(table: &'a Table, name: &str) -> &'a FieldArray {
        table
            .cols
            .iter()
            .find(|c| c.field.name == name)
            .unwrap_or_else(|| panic!("column {name} missing"))
    }

    #[cfg(feature = "datetime")]
    fn assert_temporal32(out: &Table, name: &str, dtype: ArrowType, unit: TimeUnit, value: impl Fn(usize) -> i32) {
        assert_eq!(col(out, name).field.dtype, dtype, "{name} dtype");
        match &col(out, name).array {
            Array::TemporalArray(TemporalArray::Datetime32(a)) => {
                assert_eq!(a.time_unit(), unit, "{name} unit");
                assert_eq!(ArrowType::DateTime(a.dtype.clone()), dtype, "{name} array dtype");
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(value), "{name}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }

    #[cfg(feature = "datetime")]
    fn assert_temporal64(out: &Table, name: &str, dtype: ArrowType, unit: TimeUnit, value: impl Fn(usize) -> i64) {
        assert_eq!(col(out, name).field.dtype, dtype, "{name} dtype");
        match &col(out, name).array {
            Array::TemporalArray(TemporalArray::Datetime64(a)) => {
                assert_eq!(a.time_unit(), unit, "{name} unit");
                assert_eq!(ArrowType::DateTime(a.dtype.clone()), dtype, "{name} array dtype");
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(value), "{name}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }

    fn assert_all_types(out: &Table) {
        let expected_nulls = (0..N_ROWS).filter(|&i| !is_valid(i)).count();
        assert_eq!(out.n_rows, N_ROWS);
        for c in &out.cols {
            assert!(c.field.nullable, "{} must be nullable", c.field.name);
            assert_eq!(c.null_count, expected_nulls, "{} null count", c.field.name);
            assert_eq!(c.array.len(), N_ROWS, "{} length", c.field.name);
        }

        match &col(out, "int32").array {
            Array::NumericArray(NumericArray::Int32(a)) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(int32_value))
            }
            other => panic!("int32: {other:?}"),
        }
        match &col(out, "uint32").array {
            Array::NumericArray(NumericArray::UInt32(a)) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(uint32_value))
            }
            other => panic!("uint32: {other:?}"),
        }
        match &col(out, "int64").array {
            Array::NumericArray(NumericArray::Int64(a)) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(int64_value))
            }
            other => panic!("int64: {other:?}"),
        }
        match &col(out, "uint64").array {
            Array::NumericArray(NumericArray::UInt64(a)) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(uint64_value))
            }
            other => panic!("uint64: {other:?}"),
        }
        match &col(out, "float32").array {
            Array::NumericArray(NumericArray::Float32(a)) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(float32_value))
            }
            other => panic!("float32: {other:?}"),
        }
        match &col(out, "float64").array {
            Array::NumericArray(NumericArray::Float64(a)) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(float64_value))
            }
            other => panic!("float64: {other:?}"),
        }
        match &col(out, "bool").array {
            Array::BooleanArray(a) => {
                assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(bool_value))
            }
            other => panic!("bool: {other:?}"),
        }
        assert_eq!(col(out, "string").field.dtype, ArrowType::String);
        match &col(out, "string").array {
            Array::TextArray(TextArray::String32(a)) => assert_eq!(
                (0..N_ROWS).map(|i| a.get(i).map(str::to_owned)).collect::<Vec<_>>(),
                expected(string_value)
            ),
            other => panic!("string: {other:?}"),
        }

        // A categorical column is a UTF8 string column in Parquet, so it
        // reads back as one.
        assert_eq!(col(out, "category").field.dtype, ArrowType::String);
        match &col(out, "category").array {
            Array::TextArray(TextArray::String32(a)) => assert_eq!(
                (0..N_ROWS).map(|i| a.get(i).map(str::to_owned)).collect::<Vec<_>>(),
                expected(|i| category_value(i).to_owned())
            ),
            other => panic!("category: {other:?}"),
        }

        #[cfg(feature = "datetime")]
        {
            assert_temporal32(out, "date32", ArrowType::DateTime(Dt::Date32), TimeUnit::Days, date32_value);
            // DATE stores days, so Date64 milliseconds come back as Date32.
            assert_temporal32(out, "date64", ArrowType::DateTime(Dt::Date32), TimeUnit::Days, |i| {
                (date64_value(i) / 86_400_000) as i32
            });
            // Parquet has no seconds unit, so seconds come back as milliseconds.
            assert_temporal64(
                out,
                "ts_s",
                ArrowType::DateTime(Dt::Timestamp(TimeUnit::Milliseconds, None)),
                TimeUnit::Milliseconds,
                |i| timestamp_value(i) * 1000,
            );
            assert_temporal64(
                out,
                "ts_ms",
                ArrowType::DateTime(Dt::Timestamp(TimeUnit::Milliseconds, None)),
                TimeUnit::Milliseconds,
                timestamp_value,
            );
            assert_temporal64(
                out,
                "ts_us",
                ArrowType::DateTime(Dt::Timestamp(TimeUnit::Microseconds, None)),
                TimeUnit::Microseconds,
                timestamp_value,
            );
            assert_temporal64(
                out,
                "ts_ns",
                ArrowType::DateTime(Dt::Timestamp(TimeUnit::Nanoseconds, None)),
                TimeUnit::Nanoseconds,
                timestamp_value,
            );
            // A zoned timestamp is stored as adjusted to UTC and reads back
            // with the UTC zone.
            assert_temporal64(
                out,
                "ts_us_utc",
                ArrowType::DateTime(Dt::Timestamp(TimeUnit::Microseconds, Some("UTC".to_string()))),
                TimeUnit::Microseconds,
                timestamp_value,
            );
            assert_temporal32(
                out,
                "time32_ms",
                ArrowType::DateTime(Dt::Time32(TimeUnit::Milliseconds)),
                TimeUnit::Milliseconds,
                |i| time_value(i) as i32,
            );
            // TIME(MICROS) is INT64 in Parquet, so Time32 microseconds widen.
            assert_temporal64(
                out,
                "time32_us",
                ArrowType::DateTime(Dt::Time64(TimeUnit::Microseconds)),
                TimeUnit::Microseconds,
                time_value,
            );
            // TIME(MILLIS) is INT32 in Parquet, so Time64 milliseconds narrow.
            assert_temporal32(
                out,
                "time64_ms",
                ArrowType::DateTime(Dt::Time32(TimeUnit::Milliseconds)),
                TimeUnit::Milliseconds,
                |i| time_value(i) as i32,
            );
            assert_temporal64(
                out,
                "time64_ns",
                ArrowType::DateTime(Dt::Time64(TimeUnit::Nanoseconds)),
                TimeUnit::Nanoseconds,
                time_value,
            );
            // Parquet has no seconds unit, so Time32 seconds come back as
            // Time32 milliseconds.
            assert_temporal32(
                out,
                "time32_s",
                ArrowType::DateTime(Dt::Time32(TimeUnit::Milliseconds)),
                TimeUnit::Milliseconds,
                |i| time_value(i) as i32 * 1000,
            );
            // TIME(NANOS) is INT64 in Parquet, so Time32 nanoseconds widen.
            assert_temporal64(
                out,
                "time32_ns",
                ArrowType::DateTime(Dt::Time64(TimeUnit::Nanoseconds)),
                TimeUnit::Nanoseconds,
                time_value,
            );
            // Time64 seconds scale to milliseconds and narrow to INT32.
            assert_temporal32(
                out,
                "time64_s",
                ArrowType::DateTime(Dt::Time32(TimeUnit::Milliseconds)),
                TimeUnit::Milliseconds,
                |i| time_value(i) as i32 * 1000,
            );
            assert_temporal64(
                out,
                "time64_us",
                ArrowType::DateTime(Dt::Time64(TimeUnit::Microseconds)),
                TimeUnit::Microseconds,
                time_value,
            );
        }

        #[cfg(feature = "decimal")]
        {
            assert_eq!(col(out, "dec32").field.dtype, ArrowType::Decimal32(7, 2));
            match &col(out, "dec32").array {
                Array::NumericArray(NumericArray::Decimal32(a)) => {
                    assert_eq!((a.precision, a.scale), (7, 2));
                    assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(decimal32_value));
                }
                other => panic!("dec32: {other:?}"),
            }
            assert_eq!(col(out, "dec64").field.dtype, ArrowType::Decimal64(18, 4));
            match &col(out, "dec64").array {
                Array::NumericArray(NumericArray::Decimal64(a)) => {
                    assert_eq!((a.precision, a.scale), (18, 4));
                    assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(decimal64_value));
                }
                other => panic!("dec64: {other:?}"),
            }
            assert_eq!(col(out, "dec128").field.dtype, ArrowType::Decimal128(38, 6));
            match &col(out, "dec128").array {
                Array::NumericArray(NumericArray::Decimal128(a)) => {
                    assert_eq!((a.precision, a.scale), (38, 6));
                    assert_eq!((0..N_ROWS).map(|i| a.get(i)).collect::<Vec<_>>(), expected(decimal128_value));
                }
                other => panic!("dec128: {other:?}"),
            }
        }
    }

    #[test]
    fn nullable_all_types_uncompressed() {
        let table = all_types_table();
        assert_all_types(&roundtrip(&table, None));
    }

    #[cfg(feature = "snappy")]
    #[test]
    fn nullable_all_types_snappy() {
        let table = all_types_table();
        assert_all_types(&roundtrip(&table, Some(Compression::Snappy)));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn nullable_all_types_zstd() {
        let table = all_types_table();
        assert_all_types(&roundtrip(&table, Some(Compression::Zstd)));
    }
}
