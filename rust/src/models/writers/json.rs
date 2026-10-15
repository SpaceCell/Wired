// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # JSON Writer
//!
//! Utilities for serialising [`minarrow::Table`] and [`minarrow::SuperTable`]
//! to JSON. Wraps any [`std::io::Write`] and streams records in the chosen
//! [`JsonFormat`](crate::models::encoders::json::JsonFormat).
//!
//! ## Features
//! - Pluggable destination: in-memory `Vec<u8>` or [`minarrow::Vec64`] (64-byte aligned),
//!   files, or any user-supplied [`Write`] impl
//! - Configurable output shape and formatting via [`JsonEncodeOptions`]
//! - Writes single tables or multi-batch [`SuperTable`]s
//!
//! JSON is a text format; output bytes are not consumed as SIMD payloads, so
//! a standard `Vec<u8>` is the ergonomic default. A 64-byte aligned `Vec64<u8>`
//! is also supported for callers that want consistent wire alignment across
//! formats. Alignment matters on the read side - see
//! [`JsonReader`](crate::models::readers::json::JsonReader).
//!
//! ## Quick start
//! ```no_run
//! # use minarrow::Table;
//! use lightstream::models::encoders::json::{JsonEncodeOptions, JsonFormat};
//! use lightstream::models::writers::json::JsonWriter;
//! use minarrow::Vec64;
//!
//! # let table = Table::default();
//! // In-memory output (Vec<u8>)
//! let mut w = JsonWriter::new_vec();
//! w.write_table(&table)?;
//! let bytes = w.into_inner();
//!
//! // 64-byte aligned in-memory output
//! let mut w = JsonWriter::new(Vec64::<u8>::new(), JsonEncodeOptions::default());
//! w.write_table(&table)?;
//! let aligned = w.into_inner();
//! assert_eq!(aligned.as_ptr() as usize % 64, 0);
//!
//! // NDJSON to a file
//! let opts = JsonEncodeOptions { format: JsonFormat::Ndjson, ..Default::default() };
//! let mut w = JsonWriter::to_path("out.ndjson", opts)?;
//! w.write_table(&table)?;
//! w.flush()?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::io::{self, Write};
use std::path::Path;

use minarrow::{SuperTable, Table};

use crate::models::encoders::json::{JsonEncodeOptions, encode_supertable_json, encode_table_json};
use crate::models::writers::object_storage::StorageWriter;

/// A streaming JSON writer for [`minarrow::Table`] and [`minarrow::SuperTable`].
///
/// Wraps any [`Write`] sink. Both `Vec<u8>` and `Vec64<u8>` are supported
/// in-memory since `Vec64<u8>` implements `Write` directly.
pub struct JsonWriter<W: Write> {
    writer: W,
    options: JsonEncodeOptions,
}

impl JsonWriter<Vec<u8>> {
    /// Create a `JsonWriter` backed by an in-memory `Vec<u8>` with default options.
    pub fn new_vec() -> Self {
        Self::new(Vec::new(), JsonEncodeOptions::default())
    }
}

impl<W: Write + IntoInner> JsonWriter<W> {
    /// Consume the writer and return the underlying sink. Available for sinks
    /// that define an `IntoInner::Inner` type (`Vec<u8>`, `Vec64<u8>`).
    pub fn into_inner(self) -> W::Inner {
        self.writer.into_inner_buf()
    }
}

impl<W: Write> JsonWriter<W> {
    /// Create a new `JsonWriter` wrapping the given sink with the given options.
    /// Pass `JsonEncodeOptions::default()` for defaults.
    pub fn new(writer: W, options: JsonEncodeOptions) -> Self {
        JsonWriter { writer, options }
    }

    /// Write a single [`Table`] as JSON.
    pub fn write_table(&mut self, table: &Table) -> io::Result<()> {
        encode_table_json(table, &mut self.writer, &self.options)
    }

    /// Write a [`SuperTable`] as JSON, concatenating batches per the
    /// configured [`JsonFormat`](crate::models::encoders::json::JsonFormat).
    pub fn write_supertable(&mut self, st: &SuperTable) -> io::Result<()> {
        encode_supertable_json(st, &mut self.writer, &self.options)
    }

    /// Flush the underlying sink.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl JsonWriter<StorageWriter> {
    /// Create the given file path and return a `JsonWriter`.
    ///
    /// `path` is a local file path or, with the `object_storage` feature, an
    /// object URL. An object is completed by [`JsonWriter::flush`].
    pub fn to_path<P: AsRef<Path>>(path: P, options: JsonEncodeOptions) -> io::Result<Self> {
        Ok(Self::new(StorageWriter::create(path)?, options))
    }
}

/// Lets [`JsonWriter::into_inner`] unwrap in-memory sinks without a
/// match at the call site. Implemented for `Vec<u8>` and `Vec64<u8>`.
pub trait IntoInner {
    type Inner;
    fn into_inner_buf(self) -> Self::Inner;
}

impl IntoInner for Vec<u8> {
    type Inner = Vec<u8>;
    fn into_inner_buf(self) -> Vec<u8> {
        self
    }
}

impl IntoInner for minarrow::Vec64<u8> {
    type Inner = minarrow::Vec64<u8>;
    fn into_inner_buf(self) -> minarrow::Vec64<u8> {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::models::encoders::json::JsonFormat;
    use minarrow::{
        Array, ArrowType, Buffer, Field, FieldArray, IntegerArray, NumericArray, Table, Vec64,
    };
    use simd_json::prelude::ValueAsArray;

    fn tiny_table() -> Table {
        let col = FieldArray {
            field: Field::new("n", ArrowType::Int32, false, None).into(),
            array: Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
                data: Buffer::from(Vec64::<i32>::from_slice(&[1, 2, 3])),
                null_mask: None,
            }))),
            null_count: 0,
        };
        Table::new("t".to_string(), Some(vec![col]))
    }

    fn parse(bytes: Vec<u8>) -> simd_json::OwnedValue {
        let mut buf = bytes;
        simd_json::to_owned_value(&mut buf).unwrap()
    }

    #[test]
    fn writer_vec_roundtrip() {
        let table = tiny_table();
        let mut w = JsonWriter::new_vec();
        w.write_table(&table).unwrap();
        let v = parse(w.into_inner());
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn writer_vec64_roundtrip_and_alignment() {
        let table = tiny_table();
        let mut w = JsonWriter::new(Vec64::<u8>::new(), JsonEncodeOptions::default());
        w.write_table(&table).unwrap();
        let bytes = w.into_inner();
        assert_eq!(bytes.as_ptr() as usize % 64, 0);
        let v = parse(bytes.0.into_iter().collect());
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn writer_ndjson_roundtrip() {
        let table = tiny_table();
        let opts = JsonEncodeOptions {
            format: JsonFormat::Ndjson,
            ..Default::default()
        };
        let mut w = JsonWriter::new(Vec::new(), opts);
        w.write_table(&table).unwrap();
        let s = String::from_utf8(w.into_inner()).unwrap();
        assert_eq!(s.lines().count(), 3);
    }

    #[test]
    fn writer_to_path() {
        let table = tiny_table();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut w = JsonWriter::to_path(tmp.path(), JsonEncodeOptions::default()).unwrap();
            w.write_table(&table).unwrap();
            w.flush().unwrap();
        }
        let contents = std::fs::read(tmp.path()).unwrap();
        let v = parse(contents);
        assert_eq!(v.as_array().unwrap().len(), 3);
    }
}
