// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Test Helpers - *Column and Table Generators*
//!
//! Constructors for default `minarrow` columns/tables plus in-memory/file
//! writers used in integration tests. Each function returns deterministic,
//! 4-row fixtures spanning the supported logical types.
//!
//! Feature flags gate optional types:
//! - `datetime` - date/datetime arrays
//! - `large_string` - 64-bit offset strings
//! - `extended_categorical` - 8/16/64-bit categorical indices
//! - `extended_numeric_types` - 8/16-bit integer families

use std::sync::Arc;

use minarrow::BooleanArray;
use minarrow::{
    Array, ArrowType, Bitmask, Buffer, CategoricalArray, Field, FieldArray, FloatArray,
    IntegerArray, NumericArray, StringArray, Table, TextArray, Vec64,
    ffi::arrow_dtype::CategoricalIndexType,
};
use tempfile::NamedTempFile;

#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, Dt, TemporalArray};

use tokio::io::{AsyncReadExt, AsyncWriteExt, duplex};

use crate::{
    enums::IPCMessageProtocol,
    models::writers::ipc::{
        table_stream::write_tables_to_stream, table::write_tables_to_file,
    },
};

// -------------------- Column Generators -------------------- //

/// Build a non-nullable `Int32` column: `[1, 2, 3, 4]`.
pub(crate) fn int32_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "int32".into(),
            dtype: ArrowType::Int32,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[1, 2, 3, 4])),
            null_mask: None,
        }))),
    )
}

/// Build a non-nullable `Int64` column: `[11, 12, 13, 14]`.
pub(crate) fn int64_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "int64".into(),
            dtype: ArrowType::Int64,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[11, 12, 13, 14])),
            null_mask: None,
        }))),
    )
}

/// Build a non-nullable `UInt32` column: `[21, 22, 23, 24]`.
pub(crate) fn uint32_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "uint32".into(),
            dtype: ArrowType::UInt32,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::UInt32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[21u32, 22, 23, 24])),
            null_mask: None,
        }))),
    )
}

/// Build a non-nullable `UInt64` column: `[31, 32, 33, 34]`.
pub(crate) fn uint64_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "uint64".into(),
            dtype: ArrowType::UInt64,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::UInt64(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[31u64, 32, 33, 34])),
            null_mask: None,
        }))),
    )
}

/// Build a non-nullable `Float32` column: `[0.1, 0.2, 0.3, 0.4]`.
pub(crate) fn float32_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "float32".into(),
            dtype: ArrowType::Float32,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Float32(Arc::new(FloatArray {
            data: Buffer::from(Vec64::from_slice(&[0.1f32, 0.2, 0.3, 0.4])),
            null_mask: None,
        }))),
    )
}

/// Build a non-nullable `Float64` column: `[1.1, 2.2, 3.3, 4.4]`.
pub(crate) fn float64_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "float64".into(),
            dtype: ArrowType::Float64,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: Buffer::from(Vec64::from_slice(&[1.1f64, 2.2, 3.3, 4.4])),
            null_mask: None,
        }))),
    )
}

/// Build a nullable `Boolean` column with bitmask and all-valid null mask.
pub(crate) fn bool_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "bool".into(),
            dtype: ArrowType::Boolean,
            nullable: true,
            metadata: Default::default(),
        },
        Array::BooleanArray(Arc::new(BooleanArray::new(
            Bitmask::from_bytes(&[0b0000_1101], 4),
            Some(Bitmask::new_set_all(4, true)),
        ))),
    )
}

/// Build a nullable `String32` column with offsets `[0, 4, 9, 11, 12]` over `"helloworldxx"`.
pub(crate) fn string32_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "string32".into(),
            dtype: ArrowType::String,
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::String32(Arc::new(StringArray::new(
            Buffer::from(Vec64::from_slice("helloworldxx".as_bytes())),
            Some(Bitmask::new_set_all(4, true)),
            Buffer::from(Vec64::from_slice(&[0u32, 4, 9, 11, 12])),
        )))),
    )
}

/// Build a nullable dictionary column with values `[1,0,2,1]` and uniques `["apple","banana","pear"]`.
#[cfg(not(feature = "default_categorical_8"))]
pub(crate) fn dict32_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "dict32".into(),
            dtype: ArrowType::Dictionary(CategoricalIndexType::UInt32),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical32(Arc::new(CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u32, 0, 2, 1])),
            Vec64::from(vec![
                "apple".to_string(),
                "banana".to_string(),
                "pear".to_string(),
            ]),
            Some(Bitmask::new_set_all(4, true)),
        )))),
    )
}

/// Build a nullable dictionary column with values `[1,0,2,1]` and uniques `["apple","banana","pear"]`.
#[cfg(feature = "default_categorical_8")]
pub(crate) fn dict8_default_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "dict8".into(),
            dtype: ArrowType::Dictionary(CategoricalIndexType::UInt8),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u8, 0, 2, 1])),
            Vec64::from(vec![
                "apple".to_string(),
                "banana".to_string(),
                "pear".to_string(),
            ]),
            Some(Bitmask::new_set_all(4, true)),
        )))),
    )
}

#[cfg(feature = "datetime")]
/// Build a non-nullable `Date32` column - unit is days
pub(crate) fn dt32_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "dt32".into(),
            dtype: ArrowType::DateTime(Dt::Date32),
            nullable: false,
            metadata: Default::default(),
        },
        Array::TemporalArray(TemporalArray::Datetime32(Arc::new(DatetimeArray {
            data: Buffer::from(Vec64::from_slice(&[111, 222, 333, 444])),
            null_mask: None,
            dtype: Dt::Date32,
        }))),
    )
}

#[cfg(feature = "datetime")]
/// Build a non-nullable `Date64` column - unit is milliseconds
pub(crate) fn dt64_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "dt64".into(),
            dtype: ArrowType::DateTime(Dt::Date64),
            nullable: false,
            metadata: Default::default(),
        },
        Array::TemporalArray(TemporalArray::Datetime64(Arc::new(DatetimeArray {
            data: Buffer::from(Vec64::from_slice(&[
                1_111_111, 2_222_222, 3_333_333, 4_444_444,
            ])),
            null_mask: None,
            dtype: Dt::Date64,
        }))),
    )
}

#[cfg(feature = "large_string")]
/// Build a nullable `LargeString` column with 64-bit offsets.
pub(crate) fn string64_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "string64".into(),
            dtype: ArrowType::LargeString,
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::String64(Arc::new(StringArray::new(
            Buffer::from(Vec64::from_slice("bigbigstringlarge".as_bytes())),
            Some(Bitmask::new_set_all(4, true)),
            Buffer::from(Vec64::from_slice(&[0u64, 5, 11, 16, 17])),
        )))),
    )
}

#[cfg(feature = "extended_categorical")]
/// Build a nullable `Dictionary8` column with small categorical index for extended tests.
pub(crate) fn dict8_extended_col() -> FieldArray {
    use minarrow::ffi::arrow_dtype::CategoricalIndexType::*;
    FieldArray::new(
        Field {
            name: "dict8_ext".into(),
            dtype: ArrowType::Dictionary(UInt8),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u8, 0, 2, 1])),
            Vec64::from(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            Some(Bitmask::new_set_all(4, true)),
        )))),
    )
}

#[cfg(feature = "extended_categorical")]
/// Build a nullable `Dictionary16` column with medium categorical index.
pub(crate) fn dict16_col() -> FieldArray {
    use minarrow::ffi::arrow_dtype::CategoricalIndexType::*;
    FieldArray::new(
        Field {
            name: "dict16".into(),
            dtype: ArrowType::Dictionary(UInt16),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical16(Arc::new(CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u16, 0, 2, 1])),
            Vec64::from(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
            Some(Bitmask::new_set_all(4, true)),
        )))),
    )
}

#[cfg(feature = "extended_categorical")]
/// Build a nullable `Dictionary64` column with large categorical index.
pub(crate) fn dict64_col() -> FieldArray {
    use minarrow::ffi::arrow_dtype::CategoricalIndexType::*;
    FieldArray::new(
        Field {
            name: "dict64".into(),
            dtype: ArrowType::Dictionary(UInt64),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical64(Arc::new(CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u64, 0, 2, 1])),
            Vec64::from(vec!["long".to_string(), "lo".to_string(), "l".to_string()]),
            Some(Bitmask::new_set_all(4, true)),
        )))),
    )
}

#[cfg(feature = "extended_numeric_types")]
/// Build a non-nullable `Int8` column.
pub(crate) fn int8_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "int8".into(),
            dtype: ArrowType::Int8,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Int8(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[5i8, 6, 7, 8])),
            null_mask: None,
        }))),
    )
}

#[cfg(feature = "extended_numeric_types")]
/// Build a non-nullable `Int16` column.
pub(crate) fn int16_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "int16".into(),
            dtype: ArrowType::Int16,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Int16(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[15i16, 16, 17, 18])),
            null_mask: None,
        }))),
    )
}

#[cfg(feature = "extended_numeric_types")]
/// Build a non-nullable `UInt8` column.
pub(crate) fn uint8_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "uint8".into(),
            dtype: ArrowType::UInt8,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::UInt8(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[25u8, 26, 27, 28])),
            null_mask: None,
        }))),
    )
}

#[cfg(feature = "extended_numeric_types")]
/// Build a non-nullable `UInt16` column.
pub(crate) fn uint16_col() -> FieldArray {
    FieldArray::new(
        Field {
            name: "uint16".into(),
            dtype: ArrowType::UInt16,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::UInt16(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[35u16, 36, 37, 38])),
            null_mask: None,
        }))),
    )
}

// -------------------- Table Generator -------------------- //

/// Construct a 4-row `Table` covering all available types under the current feature set.
///
/// Always includes base numeric, boolean, string32, and dictionary32 columns.
/// feature-flag conditionally includes `datetime`, `large_string`, `extended_categorical`,
/// and `extended_numeric_types` columns.
pub(crate) fn make_all_types_table() -> Table {
    #[allow(unused_mut)]
    let mut cols = vec![
        int32_col(),
        int64_col(),
        uint32_col(),
        uint64_col(),
        float32_col(),
        float64_col(),
        bool_col(),
        string32_col(),
    ];
    #[cfg(not(feature = "default_categorical_8"))]
    cols.push(dict32_col());
    #[cfg(feature = "default_categorical_8")]
    cols.push(dict8_default_col());
    #[cfg(feature = "datetime")]
    {
        cols.push(dt32_col());
        cols.push(dt64_col());
    }
    #[cfg(feature = "large_string")]
    cols.push(string64_col());
    #[cfg(feature = "extended_categorical")]
    {
        cols.push(dict8_extended_col());
        cols.push(dict16_col());
        cols.push(dict64_col());
    }
    #[cfg(feature = "extended_numeric_types")]
    {
        cols.push(int8_col());
        cols.push(int16_col());
        cols.push(uint8_col());
        cols.push(uint16_col());
    }
    Table {
        cols,
        n_rows: 4,
        name: "all_types".to_owned(),
        ..Default::default()
    }
}

/// Extract a schema (fields) from `make_all_types_table()`.
pub(crate) fn make_schema_all_types() -> Vec<Field> {
    make_all_types_table()
        .cols
        .iter()
        .map(|fa| fa.field.as_ref().clone())
        .collect()
}

// -------------------- File Writers -------------------- //

/// Write the provided tables to a temporary Arrow IPC *file* and return the handle.
///
/// Errors on file creation or write are bubbled via panics in tests for clarity.
pub(crate) async fn write_test_table_to_file(tables: &[Table]) -> NamedTempFile {
    let temp = NamedTempFile::new().expect("Failed to create NamedTempFile");
    let path = temp.path().to_str().unwrap();
    let schema = make_schema_all_types();

    write_tables_to_file(path, tables, schema)
        .await
        .expect("Failed to write tables");
    temp
}

/// Write tables to an in-memory Arrow IPC *stream* and return chunked payloads.
///
/// - `tables`: borrowed slice, not moved or cloned.
/// - `schema`: borrowed slice (field definitions).
/// - `protocol`: stream vs file framing.
/// - `chunk_size`: number of bytes per simulated stream chunk.
#[allow(dead_code)]
pub async fn write_test_table_to_ipc_stream(
    tables: &[Table],
    schema: &[Field],
    protocol: IPCMessageProtocol,
    chunk_size: usize,
) -> std::io::Result<Vec<Vec<u8>>> {
    let (mut tx, mut rx) = duplex(16 * 1024);

    write_tables_to_stream::<_, Vec64<u8>>(&mut tx, tables, schema.to_vec(), protocol).await?;
    tx.shutdown().await.ok();

    // Read all bytes from the reader end.
    let mut all_bytes = Vec::new();
    let mut buf = vec![0u8; 8192];
    loop {
        let n = rx.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        all_bytes.extend_from_slice(&buf[..n]);
    }

    // Split into Vec<Vec<u8>> chunks for streaming.
    let mut chunks = Vec::new();
    let mut idx = 0;
    while idx < all_bytes.len() {
        let end = usize::min(idx + chunk_size, all_bytes.len());
        chunks.push(all_bytes[idx..end].to_vec());
        idx = end;
    }
    Ok(chunks)
}
