// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # CSV Writer
//!
//! Utilities for serialising [`minarrow::Table`] and [`minarrow::SuperTable`] to CSV.
//! Wraps any [`std::io::Write`] and streams rows with configurable options.
//!
//! ## Features
//! - Pluggable destination: in-memory (`Vec<u8>`) or files (`std::fs::File`)
//! - Configurable delimiter, header emission, and null representation via [`CsvEncodeOptions`]
//! - RFC 4180-style quoting/escaping handled by the encoders
//! - Write single tables or concatenate multi-batch [`SuperTable`]s - header on first batch only
//!
//! ## Quick start
//! ```no_run
//! # use minarrow::{Field, Table};
//! # use minarrow::ArrowType::*;
//! # use minarrow::Field as F;
//! use lightstream::models::encoders::csv::{CsvEncodeOptions};
//! use lightstream::models::writers::csv::CsvWriter;
//!
//! // In-memory
//! let mut w = CsvWriter::new_vec();
//! # let table = Table::default();
//! w.write_table(&table)?;
//! let bytes = w.into_inner(); // CSV bytes
//!
//! // To a file - default options
//! let mut wf = CsvWriter::to_path("out.csv")?;
//! wf.write_table(&table)?;
//! wf.flush()?;
//!
//! // Custom options
//! let opts = CsvEncodeOptions { delimiter: b';', null_repr: "NULL", ..Default::default() };
//! let mut wc = CsvWriter::with_options(Vec::new(), opts);
//! wc.write_table(&table)?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::io::{self, Write};
use std::path::Path;

use minarrow::{SuperTable, Table};

use crate::models::encoders::csv::{CsvEncodeOptions, encode_supertable_csv, encode_table_csv};
use crate::models::writers::object_storage::StorageWriter;

/// A streaming CSV writer for [`minarrow::Table`] and [`minarrow::SuperTable`].
///
/// Wraps any type implementing [`std::io::Write`] and serialises rows to CSV
/// using configurable options (`delimiter`, `null_repr`, header control, etc.).
///
/// ## Features
/// - Works with arbitrary sinks: `Vec<u8>`, `File`, sockets, etc.
/// - Customisable via [`CsvEncodeOptions`]
/// - Supports both single-batch tables and multi-batch supertables
pub struct CsvWriter<W: Write> {
    writer: W,
    options: CsvEncodeOptions,
}

impl CsvWriter<Vec<u8>> {
    /// Create a `CsvWriter` that writes into an in-memory `Vec<u8>`,
    /// with default CSV options.
    pub fn new_vec() -> Self {
        Self::with_options(Vec::new(), CsvEncodeOptions::default())
    }

    /// Consume the writer and return the underlying `Vec<u8>` containing the CSV.
    pub fn into_inner(self) -> Vec<u8> {
        self.writer
    }
}

impl<W: Write> CsvWriter<W> {
    /// Create a new `CsvWriter` wrapping the given writer, with default options.
    pub fn new(writer: W) -> Self {
        Self::with_options(writer, CsvEncodeOptions::default())
    }

    /// Create with custom `CsvEncodeOptions`.
    pub fn with_options(writer: W, options: CsvEncodeOptions) -> Self {
        CsvWriter { writer, options }
    }

    /// Write a single `Table` as CSV to the underlying writer.
    ///
    /// This writes headers if enabled and all rows, then leaves the writer
    /// ready for more writes or flushing.
    pub fn write_table(&mut self, table: &Table) -> io::Result<()> {
        encode_table_csv(table, &mut self.writer, &self.options)
    }

    /// Write a `SuperTable` (i.e. multiple batches) as CSV.
    ///
    /// Only the first batch will include headers if enabled; subsequent
    /// batches are concatenated without headers.
    pub fn write_supertable(&mut self, st: &SuperTable) -> io::Result<()> {
        encode_supertable_csv(st, &mut self.writer, &self.options)
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl CsvWriter<StorageWriter> {
    /// Create the given file path and return a `CsvWriter` using default
    /// options.
    ///
    /// `path` is a local file path or, with the `object_storage` feature, an
    /// object URL. An object is completed by [`CsvWriter::flush`].
    pub fn to_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self::new(StorageWriter::create(path)?))
    }

    /// Create the given file path and return a `CsvWriter` with custom
    /// options.
    ///
    /// `path` is a local file path or, with the `object_storage` feature, an
    /// object URL. An object is completed by [`CsvWriter::flush`].
    pub fn to_path_with_options<P: AsRef<Path>>(
        path: P,
        options: CsvEncodeOptions,
    ) -> io::Result<Self> {
        Ok(Self::with_options(StorageWriter::create(path)?, options))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::models::encoders::csv::CsvEncodeOptions;
    use minarrow::{
        Array, Bitmask, Buffer, Field, FieldArray, NumericArray, Table, TextArray, vec64,
    };

    fn make_test_table() -> Table {
        let int_col = FieldArray {
            field: Field {
                name: "ints".to_string(),
                dtype: minarrow::ArrowType::Int32,
                nullable: false,
                metadata: Default::default(),
            }
            .into(),
            array: Array::NumericArray(NumericArray::Int32(
                minarrow::IntegerArray {
                    data: Buffer::from(vec64![1, 2, 3]),
                    null_mask: None,
                }
                .into(),
            )),
            null_count: 0,
        };
        let str_col = FieldArray {
            field: Field {
                name: "strs".to_string(),
                dtype: minarrow::ArrowType::String,
                nullable: true,
                metadata: Default::default(),
            }
            .into(),
            array: Array::TextArray(TextArray::String32(
                minarrow::StringArray {
                    // Canonical Arrow string layout: offsets bracket each
                    // cell's bytes, no padding NULs between cells.
                    offsets: Buffer::from(vec64![0u32, 3, 3, 9]),
                    data: Buffer::from_vec64(b"foobarbaz".to_vec().into()),
                    null_mask: Some(Bitmask::from_bools(&[true, false, true])),
                }
                .into(),
            )),
            null_count: 1,
        };
        Table {
            name: "test".to_string(),
            cols: vec![int_col, str_col],
            n_rows: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_csv_writer_table_default() {
        let table = make_test_table();
        let mut w = CsvWriter::new_vec();
        w.write_table(&table).unwrap();
        w.flush().unwrap();
        let csv = String::from_utf8(w.into_inner()).unwrap();

        // header + 3 rows
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "ints,strs");
        assert_eq!(lines[1], "1,foo");
        assert_eq!(lines[2], "2,");
        assert_eq!(lines[3], "3,barbaz");
    }

    #[test]
    fn test_csv_writer_table_custom_options() {
        let table = make_test_table();
        let mut opts = CsvEncodeOptions::default();
        opts.delimiter = b';';
        opts.null_repr = "NULL";
        let mut w = CsvWriter::with_options(Vec::new(), opts.clone());
        w.write_table(&table).unwrap();
        let out = w.into_inner();
        let csv = String::from_utf8(out).unwrap();
        // header uses ';'
        assert!(csv.starts_with("ints;strs\n"));
        // null row uses "NULL"
        assert!(csv.contains("2;NULL"));
    }

    #[test]
    fn test_csv_writer_supertable() {
        let t1 = make_test_table();
        let t2 = make_test_table();
        let supertbl =
            SuperTable::from_batches(vec![Arc::new(t1.clone()), Arc::new(t2.clone())], None);
        let mut writer = CsvWriter::new_vec();
        writer.write_supertable(&supertbl).unwrap();
        let csv = String::from_utf8(writer.into_inner()).unwrap();
        let lines: Vec<_> = csv.lines().collect();
        // header + 3 rows + 3 rows = 7 lines
        assert_eq!(lines.len(), 1 + 3 + 3);
        assert_eq!(lines[0], "ints,strs");
        assert_eq!(lines[4], "1,foo"); // first row of second batch, no header
    }

    #[test]
    fn test_csv_writer_to_path() {
        let table = make_test_table();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut writer = CsvWriter::to_path(tmp.path()).unwrap();
            writer.write_table(&table).unwrap();
            writer.flush().unwrap();
        }
        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(contents.contains("ints,strs"));
    }
}
