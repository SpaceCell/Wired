// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked Parquet reader
//!
//! Reads a directory of `<base>-NNNNNNNNNN.parquet` files emitted by
//! `ChunkedParquetWriter` and presents them as an ordered iterator of
//! `Table`s. Each chunk file is a complete, independently parseable
//! Parquet file. The reader sorts files by their numeric index so
//! consumers see batches in write order.
//!
//! ## Concurrency model
//!
//! The default [`Iterator`] path is sync and serial. Per-file Parquet
//! decode is CPU-heavy (decompression, page parsing, dictionary
//! resolution), so across files the gain from parallel reads is larger
//! than for raw IPC. [`ChunkedTableReader::par_load_batched`] (inherited from
//! the trait) uses `std::thread::scope` to fan per-chunk work end-to-end
//! across worker threads and returns a `SuperTable` with batches in
//! write order.

use std::path::{Path, PathBuf};

use minarrow::Table;

use crate::error::IoError;
use crate::models::readers::parquet::{load_parquet_table_cols, load_parquet_table};
use crate::models::readers::object_storage::{StorageReader, list_dir};
use crate::traits::chunked_table_reader::ChunkedTableReader;

use std::sync::Arc;

use minarrow::SuperTable;

/// Iterator over chunk files in a directory written by
/// `ChunkedParquetWriter`. Yields one `Table` per chunk file in ascending
/// index order; the previous chunk's file handle is dropped before the
/// next is opened.
pub struct ChunkedParquetReader {
    paths: Vec<PathBuf>,
    cursor: usize,
}

impl ChunkedTableReader for ChunkedParquetReader {
    type Error = IoError;
    type Options = ();

    fn open<P: AsRef<Path>>(dir: P, base: &str, _options: ()) -> Result<Self, IoError> {
        let paths = Self::list_paths(dir, base)?;
        Ok(Self { paths, cursor: 0 })
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn list_paths<P: AsRef<Path>>(dir: P, base: &str) -> Result<Vec<PathBuf>, IoError> {
        let prefix = format!("{base}-");
        let mut indexed: Vec<(u64, PathBuf)> = Vec::new();
        for path in list_dir(dir)? {
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !name.starts_with(&prefix) || !name.ends_with(".parquet") {
                continue;
            }
            let index_str = &name[prefix.len()..name.len() - ".parquet".len()];
            let Ok(index) = index_str.parse::<u64>() else {
                continue;
            };
            indexed.push((index, path));
        }
        indexed.sort_by_key(|(i, _)| *i);
        Ok(indexed.into_iter().map(|(_, p)| p).collect())
    }

    fn read_chunk(&self, path: &Path) -> Result<Table, IoError> {
        load_parquet_table(StorageReader::open(path)?)
    }

    fn read_chunk_cols(&self, path: &Path, columns: &[&str]) -> Result<Table, IoError> {
        load_parquet_table_cols(StorageReader::open(path)?, columns)
    }

    fn load_batched_cols(self, columns: &[&str]) -> Result<SuperTable, IoError> {
        let mut batches: Vec<Arc<Table>> = Vec::new();
        let mut name: Option<String> = None;
        for path in &self.paths[self.cursor..] {
            let table = self.read_chunk_cols(path, columns)?;
            if name.is_none() {
                name = Some(table.name.clone());
            }
            batches.push(Arc::new(table));
        }
        Ok(SuperTable::from_batches(
            batches,
            name.or(Some("chunked".into())),
        ))
    }
}

impl Iterator for ChunkedParquetReader {
    type Item = Result<Table, IoError>;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.paths.get(self.cursor)?.clone();
        self.cursor += 1;
        Some(self.read_chunk(&path))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::models::writers::chunked::parquet::ChunkedParquetWriter;
    use crate::traits::chunked_table_writer::ChunkedTableWriter;
    use minarrow::{Table, fa_i32};

    #[test]
    fn par_load_batched_returns_batches_in_write_order() {
        let dir = std::env::temp_dir().join("lightstream_chunked_parquet_par_reader");
        let _ = fs::remove_dir_all(&dir);

        let mut w = ChunkedParquetWriter::new(&dir, "part", None).unwrap();
        for i in 0..8i32 {
            w.write_chunk(&Table::new(
                "b",
                Some(vec![fa_i32!("n", i, i + 100)]),
            ))
            .unwrap();
        }

        let st = ChunkedParquetReader::par_load_batched(&dir, "part", (), None).unwrap();
        assert_eq!(st.batches.len(), 8);
        for (i, batch) in st.batches.iter().enumerate() {
            assert_eq!(batch.n_rows, 2);
            // First column's first row should be `i`, confirming write order
            // round-trips through parallel decode unchanged.
            let arr = &batch.cols[0].array;
            let s = format!("{arr:?}");
            assert!(
                s.contains(&format!("{i}")),
                "batch {i} first column did not contain {i}: {s}"
            );
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn categorical_column_reads_back_as_utf8_strings() {
        // Regression: the reader's DictionaryPageHeader inner parser used
        // to consume raw i32 values without the Thrift type/id prefixes
        // the writer emits. That left the cursor 3 bytes shy of the
        // payload, so the dictionary bytes were misaligned and the next
        // length prefix was parsed as garbage - causing
        // `parse_dictionary_values` to ask for ~50 MB and fail with
        // `UnexpectedEof`. This test exercises the round-trip end-to-end.
        //
        // Parquet has no dictionary type, so the categorical column is a
        // UTF8 string column with dictionary-encoded pages and reads back
        // as strings.
        use crate::models::readers::parquet::load_parquet_table;
        use crate::models::writers::parquet::write_parquet_table;
        use minarrow::{
            Array, ArrowType, Bitmask, Buffer, CategoricalArray, Field, FieldArray, MaskedArray,
            TextArray, Vec64, ffi::arrow_dtype::CategoricalIndexType,
        };
        use std::sync::Arc;

        let n_rows = 256usize;
        let unique_values = Vec64::from(vec![
            "red".to_string(),
            "green".to_string(),
            "blue".to_string(),
        ]);
        // The u8 categorical form exists only under `default_categorical_8`,
        // so the index width follows the active feature set.
        #[cfg(feature = "default_categorical_8")]
        let (dtype, array) = {
            let indices: Vec64<u8> = (0..n_rows).map(|i| (i % 3) as u8).collect();
            (
                ArrowType::Dictionary(CategoricalIndexType::UInt8),
                Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray::new(
                    Buffer::from(indices),
                    unique_values,
                    Some(Bitmask::new_set_all(n_rows, true)),
                )))),
            )
        };
        #[cfg(not(feature = "default_categorical_8"))]
        let (dtype, array) = {
            let indices: Vec64<u32> = (0..n_rows).map(|i| (i % 3) as u32).collect();
            (
                ArrowType::Dictionary(CategoricalIndexType::UInt32),
                Array::TextArray(TextArray::Categorical32(Arc::new(CategoricalArray::new(
                    Buffer::from(indices),
                    unique_values,
                    Some(Bitmask::new_set_all(n_rows, true)),
                )))),
            )
        };
        let dict_col = FieldArray::new(
            Field {
                name: "category".into(),
                dtype,
                nullable: true,
                metadata: Default::default(),
            },
            array,
        );
        let table = Table::new("t", Some(vec![dict_col]));

        let path = std::env::temp_dir().join("ls_categorical_roundtrip.parquet");
        let _ = std::fs::remove_file(&path);
        write_parquet_table(
            &table,
            std::fs::File::create(&path).unwrap(),
            None,
        )
        .unwrap();

        let got = load_parquet_table(std::io::BufReader::new(std::fs::File::open(&path).unwrap()))
            .expect("categorical column must round-trip via Parquet");
        assert_eq!(got.n_rows, n_rows);
        assert_eq!(got.cols.len(), 1);
        assert_eq!(got.cols[0].field.dtype, ArrowType::String);
        match &got.cols[0].array {
            Array::TextArray(TextArray::String32(strings)) => {
                let expected: Vec<Option<&str>> =
                    (0..n_rows).map(|i| Some(["red", "green", "blue"][i % 3])).collect();
                assert_eq!((0..n_rows).map(|i| strings.get(i)).collect::<Vec<_>>(), expected);
            }
            other => panic!("expected a String column, got {other:?}"),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn par_load_batched_handles_empty_directory() {
        let dir = std::env::temp_dir().join("lightstream_chunked_parquet_par_reader_empty");
        let _ = fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let st = ChunkedParquetReader::par_load_batched(&dir, "part", (), None).unwrap();
        assert!(st.batches.is_empty());
        assert_eq!(st.n_rows, 0);

        fs::remove_dir_all(&dir).ok();
    }
}
