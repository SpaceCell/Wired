// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # JsonReader - *Read JSON into `Table`*
//!
//! High-level API for reading JSON files or streams into Minarrow Tables.
//! Supports either a single JSON array-of-objects or streaming NDJSON
//! (newline-delimited), chosen via [`JsonFormat`].
//!
//! For NDJSON, records can be pulled in fixed-size batches via
//! [`JsonReader::next_batch`]; array-of-objects must be fully read in one pass.
//!
//! See [`JsonDecodeOptions`] for schema handling and type control.

use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use minarrow::{Field, SuperTable, Table};

use crate::models::decoders::json::row_decoder::JsonRowDecoder;
use crate::models::decoders::json::builder::ColumnBuilder;
use crate::models::decoders::json::simd::TapeDecoder;
use crate::models::decoders::json::{
    JsonDecodeOptions, append_ndjson_line, decode_json, finish_table, make_builders, make_field_map,
};
use crate::models::encoders::json::JsonFormat;
use crate::models::readers::object_storage::StorageReader;

/// Reads JSON data into Minarrow Tables.
///
/// Use `from_path`, `from_reader`, or `from_slice`. For NDJSON, iterate with
/// `next_batch`. For array-of-objects, call `load_table` to materialise the
/// whole payload at once.
///
/// The reader holds reusable simd-json `Buffers` and line/chunk vectors
/// across NDJSON batches so the steady-state path does not touch the
/// allocator.
pub struct JsonReader<R: BufRead> {
    reader: R,
    options: JsonDecodeOptions,
    format: JsonFormat,
    batch_size: usize,
    /// True once the underlying reader has signalled EOF. Flipped only
    /// by `read_until` returning 0 - never inferred from row count, so a
    /// chunk-full short batch does not look like end-of-stream.
    finished: bool,
    /// Reused across NDJSON batches; the simd-json `Buffers` inside grow
    /// once and are then reused for every subsequent batch.
    decoder: TapeDecoder,
    /// Reused chunk buffer for the `[obj1,obj2,...]` array form that
    /// simd-json's tape parser consumes. Cleared between batches.
    chunk: Vec<u8>,
    /// Reused line buffer for `read_until` in the NDJSON loop.
    line: Vec<u8>,
}

impl JsonReader<StorageReader> {
    /// Open a JSON file at the given path.
    ///
    /// `path` is a local file path or, with the `object_storage` feature, an
    /// object URL.
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        format: JsonFormat,
        options: JsonDecodeOptions,
        batch_size: usize,
    ) -> io::Result<Self> {
        Ok(Self::from_reader(
            StorageReader::open(path)?,
            format,
            options,
            batch_size,
        ))
    }
}

impl<R: BufRead> JsonReader<R> {
    /// Create from any `BufRead` source.
    pub fn from_reader(
        reader: R,
        format: JsonFormat,
        options: JsonDecodeOptions,
        batch_size: usize,
    ) -> Self {
        JsonReader {
            reader,
            options,
            format,
            batch_size,
            finished: false,
            decoder: TapeDecoder::new(),
            chunk: Vec::new(),
            line: Vec::with_capacity(4096),
        }
    }
}

impl<'a> JsonReader<BufReader<&'a [u8]>> {
    /// Create from a byte slice. The format controls how the input is parsed.
    pub fn from_slice(
        slice: &'a [u8],
        format: JsonFormat,
        options: JsonDecodeOptions,
        batch_size: usize,
    ) -> JsonReader<BufReader<&'a [u8]>> {
        JsonReader::from_reader(BufReader::new(slice), format, options, batch_size)
    }
}

impl<R: BufRead> JsonReader<R> {
    /// Caller-provided schema, if any. Both formats currently require a
    /// schema in [`JsonDecodeOptions`], so this is a thin accessor.
    pub fn schema(&self) -> &[Field] {
        self.options.schema.as_deref().unwrap_or(&[])
    }

    /// Read the next batch of rows as a `Table`.
    ///
    /// For NDJSON, returns up to `batch_size` rows per call and `None` at EOS.
    /// For array-of-objects, the first call returns the entire table and all
    /// subsequent calls return `None`.
    pub fn next_batch(&mut self) -> io::Result<Option<Table>> {
        if self.finished {
            return Ok(None);
        }

        match self.format {
            JsonFormat::Ndjson => self.next_ndjson_batch(),
            JsonFormat::Array { .. } => {
                // Array is a single parse over the whole input.
                let table = decode_json(&mut self.reader, &self.options)?;
                self.finished = true;
                Ok(Some(table))
            }
        }
    }

    /// Consume the entire input and return a single `Table`.
    ///
    /// For NDJSON this drains successive chunks into a single set of
    /// builders so the result is one Table covering every row. For
    /// array-of-objects it parses the whole input in one go.
    pub fn load_table(mut self) -> io::Result<Table> {
        match self.format {
            JsonFormat::Array { .. } => decode_json(&mut self.reader, &self.options),
            JsonFormat::Ndjson => self.drain_ndjson_into_single_table(),
        }
    }

    /// Consume the entire input and return a `SuperTable` whose batches
    /// each correspond to one call's worth of rows.
    ///
    /// For NDJSON, every chunk-flush produces one batch; the SuperTable
    /// holds them in order. For array-of-objects, the whole payload is
    /// a single batch wrapped in a one-batch SuperTable.
    pub fn load_batched(mut self) -> io::Result<SuperTable> {
        let name = match self.format {
            JsonFormat::Ndjson => "ndjson",
            JsonFormat::Array { .. } => "json",
        };
        match self.format {
            JsonFormat::Array { .. } => {
                let table = decode_json(&mut self.reader, &self.options)?;
                Ok(SuperTable::from_batches(
                    vec![Arc::new(table)],
                    Some(name.into()),
                ))
            }
            JsonFormat::Ndjson => {
                let mut batches: Vec<Arc<Table>> = Vec::new();
                while let Some(table) = self.next_batch()? {
                    batches.push(Arc::new(table));
                }
                Ok(SuperTable::from_batches(batches, Some(name.into())))
            }
        }
    }

    /// NDJSON path of `load_table`: keeps one set of builders alive
    /// across every chunk flush so all decoded rows land in one Table.
    fn drain_ndjson_into_single_table(&mut self) -> io::Result<Table> {
        let schema = self
            .options
            .schema
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "schema is required"))?
            .clone();
        let mut builders: Vec<ColumnBuilder> =
            make_builders(&schema, 0, self.options.string_bytes_per_row)?;
        let field_map = make_field_map(&schema);
        loop {
            let n = self.fill_chunk()?;
            if n == 0 {
                break;
            }
            self.chunk.push(b']');
            self.decoder.decode_rows(
                self.chunk.as_mut_slice(),
                &mut builders,
                &field_map,
                self.options.on_type_mismatch,
            )?;
            self.chunk.clear();
        }
        Ok(finish_table(&schema, builders))
    }

    /// Pull up to `batch_size` rows out of the reader and decode them
    /// into a fresh `Table`. The chunk and line buffers are reused.
    fn next_ndjson_batch(&mut self) -> io::Result<Option<Table>> {
        let schema = self
            .options
            .schema
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "schema is required"))?
            .clone();

        let line_count = self.fill_chunk()?;
        if line_count == 0 {
            return Ok(None);
        }
        self.chunk.push(b']');

        let mut builders = make_builders(&schema, line_count, self.options.string_bytes_per_row)?;
        let field_map = make_field_map(&schema);
        self.decoder.decode_rows(
            self.chunk.as_mut_slice(),
            &mut builders,
            &field_map,
            self.options.on_type_mismatch,
        )?;
        self.chunk.clear();
        Ok(Some(finish_table(&schema, builders)))
    }

    /// Fill `self.chunk` with up to `batch_size` lines wrapped as a
    /// JSON array body (`[obj1,obj2,...`) - the trailing `]` is added
    /// by the caller once the batch is sealed. Stops on batch size,
    /// `max_chunk_bytes`, or EOF; sets `self.finished` only on EOF.
    /// Returns the number of non-blank lines accumulated.
    fn fill_chunk(&mut self) -> io::Result<usize> {
        let mut line_count = 0usize;
        while line_count < self.batch_size && self.chunk.len() < self.options.max_chunk_bytes {
            self.line.clear();
            let n = self.reader.read_until(b'\n', &mut self.line)?;
            if n == 0 {
                self.finished = true;
                break;
            }
            append_ndjson_line(&self.line, &mut self.chunk, &mut line_count);
        }
        Ok(line_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minarrow::{Array, ArrowType, Field, NumericArray};
    use std::io::BufReader;

    #[test]
    fn reader_array_full_table() {
        let json = br#"[{"i":1,"s":"a"},{"i":2,"s":"b"},{"i":3,"s":"c"}]"#;
        let schema = vec![
            Field::new("i", ArrowType::Int32, false, None),
            Field::new("s", ArrowType::String, true, None),
        ];
        let reader = JsonReader::<BufReader<&[u8]>>::from_slice(
            json,
            JsonFormat::default(),
            JsonDecodeOptions {
                schema: Some(schema),
                ..Default::default()
            },
            10,
        );
        let tbl = reader.load_table().unwrap();
        assert_eq!(tbl.n_rows, 3);
        assert_eq!(tbl.cols.len(), 2);
    }

    #[test]
    fn reader_ndjson_batches() {
        let json = b"{\"i\":1}\n{\"i\":2}\n{\"i\":3}\n{\"i\":4}\n";
        let schema = vec![Field::new("i", ArrowType::Int32, false, None)];
        let mut reader = JsonReader::<BufReader<&[u8]>>::from_slice(
            json,
            JsonFormat::Ndjson,
            JsonDecodeOptions {
                schema: Some(schema),
                ..Default::default()
            },
            2,
        );

        let b1 = reader.next_batch().unwrap().unwrap();
        assert_eq!(b1.n_rows, 2);
        let b2 = reader.next_batch().unwrap().unwrap();
        assert_eq!(b2.n_rows, 2);
        let b3 = reader.next_batch().unwrap();
        assert!(b3.is_none());
    }

    #[test]
    fn reader_ndjson_full_table() {
        let json = b"{\"i\":1}\n{\"i\":2}\n{\"i\":3}\n";
        let schema = vec![Field::new("i", ArrowType::Int32, false, None)];
        let reader = JsonReader::<BufReader<&[u8]>>::from_slice(
            json,
            JsonFormat::Ndjson,
            JsonDecodeOptions {
                schema: Some(schema),
                ..Default::default()
            },
            10,
        );
        let tbl = reader.load_table().unwrap();
        assert_eq!(tbl.n_rows, 3);
        match &tbl.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                let vs: Vec<i32> = arr.data.as_ref().iter().copied().collect();
                assert_eq!(vs, vec![1, 2, 3]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn reader_ndjson_short_batch_does_not_finish() {
        // Tail batch has only 1 row but more data could in principle
        // arrive on a streaming source - the reader must not flip
        // `finished` just because rows < batch_size.
        let json = b"{\"i\":1}\n{\"i\":2}\n{\"i\":3}\n";
        let schema = vec![Field::new("i", ArrowType::Int32, false, None)];
        let mut reader = JsonReader::<BufReader<&[u8]>>::from_slice(
            json,
            JsonFormat::Ndjson,
            JsonDecodeOptions {
                schema: Some(schema),
                ..Default::default()
            },
            2,
        );
        let b1 = reader.next_batch().unwrap().unwrap();
        assert_eq!(b1.n_rows, 2);
        let b2 = reader.next_batch().unwrap().unwrap();
        assert_eq!(b2.n_rows, 1);
        // Now actual EOF: read_until returns 0, finished flips, None.
        let b3 = reader.next_batch().unwrap();
        assert!(b3.is_none());
    }
}
