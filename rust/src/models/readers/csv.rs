// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # CsvReader - *Read CSVs into `Table`*
//!
//! High-level API for reading CSV files or streams into Minarrow Tables.
//! - Supports chunked reading (i.e., batch iteration), schema inference or user-specified schema.
//! - Fully customisable options - delimiter, nulls, quoting, batch size, etc.
//! - No external dependencies.
//!
//! See `CsvDecodeOptions` for configuration.

use crate::models::decoders::csv::{CsvDecodeOptions, decode_csv};
use crate::models::readers::object_storage::StorageReader;
use minarrow::{Field, SuperTable, Table};
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

/// Reads CSV files into Minarrow Tables.
/// - Use `from_path`, `from_reader`, or `from_slice`.
/// - Includes `next_batch` for chunked reading.
/// - Supports schema inference and access.
pub struct CsvReader<R: BufRead> {
    reader: R,
    options: CsvDecodeOptions,
    schema: Option<Vec<Field>>,
    batch_size: usize,
    finished: bool,
}

impl CsvReader<StorageReader> {
    /// Open a CSV file at the given path with the given options.
    ///
    /// `path` is a local file path or, with the `object_storage` feature, an
    /// object URL.
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        options: CsvDecodeOptions,
        batch_size: usize,
    ) -> io::Result<Self> {
        Ok(Self::from_reader(StorageReader::open(path)?, options, batch_size))
    }
}

impl<R: BufRead> CsvReader<R> {
    /// Create from any BufRead (file, slice, etc.)
    pub fn from_reader(reader: R, options: CsvDecodeOptions, batch_size: usize) -> Self {
        CsvReader {
            reader,
            options,
            schema: None,
            batch_size,
            finished: false,
        }
    }

    /// Create from a byte slice.
    pub fn from_slice(
        slice: &[u8],
        options: CsvDecodeOptions,
        batch_size: usize,
    ) -> CsvReader<BufReader<&[u8]>> {
        let reader = BufReader::new(slice);
        CsvReader::from_reader(reader, options, batch_size)
    }

    /// Get the inferred or user-provided schema
    ///
    /// Requires reading first batch if not already done
    pub fn schema(&mut self) -> io::Result<&[Field]> {
        if self.schema.is_none()
            && !self.finished
            && let Some(batch) = self.next_batch()?
        {
            self.schema = Some(
                batch
                    .cols
                    .iter()
                    .map(|c| c.field.as_ref().clone())
                    .collect(),
            );
        }
        Ok(self.schema.as_deref().unwrap_or(&[]))
    }

    /// Read the next batch of rows as a Table.
    /// Returns Ok(None) if end of file is reached
    pub fn next_batch(&mut self) -> io::Result<Option<Table>> {
        if self.finished {
            return Ok(None);
        }

        let mut batch_options = self.options.clone();

        // If we have a schema, all further batches should *not* treat any row as header
        if self.schema.is_some() {
            batch_options.has_header = false;
            batch_options.schema = self.schema.clone();
        }

        let mut rows: Vec<Vec<u8>> = Vec::new();
        let mut saw_any = false;

        if self.batch_size == usize::MAX {
            // Unbounded path: read the whole input in one syscall, count
            // newlines via SWAR to size `rows` exactly, then split.
            // Avoids both the integer overflow on `batch_size + 1` and
            // any Vec growth on the row vector.
            let mut whole = Vec::new();
            self.reader.read_to_end(&mut whole)?;
            if !whole.is_empty() {
                // Strip the trailing newline so `split` doesn't emit an
                // empty element for `\n`-terminated input.
                let body: &[u8] = if whole.ends_with(b"\n") {
                    &whole[..whole.len() - 1]
                } else {
                    &whole[..]
                };
                // `lines = separators + 1` for any non-empty `body`.
                let n_lines = count_byte_swar(body, b'\n') + 1;
                rows.reserve_exact(n_lines);
                for line in body.split(|&b| b == b'\n') {
                    let stripped: &[u8] = if line.ends_with(b"\r") {
                        &line[..line.len() - 1]
                    } else {
                        line
                    };
                    if stripped.is_empty() && !saw_any {
                        continue;
                    }
                    saw_any = true;
                    rows.push(stripped.to_vec());
                }
            }
            self.finished = true;
        } else {
            let header_row = if self.schema.is_none() && self.options.has_header {
                1
            } else {
                0
            };
            let row_limit = self.batch_size + header_row;
            rows.reserve_exact(row_limit);
            let mut buf = Vec::new();
            while rows.len() < row_limit {
                buf.clear();
                let n = self.reader.read_until(b'\n', &mut buf)?;
                if n == 0 {
                    break;
                }
                if buf.ends_with(b"\r\n") {
                    buf.truncate(buf.len() - 2);
                } else if buf.ends_with(b"\n") {
                    buf.truncate(buf.len() - 1);
                }
                if buf.is_empty() && !saw_any {
                    continue;
                }
                saw_any = true;
                rows.push(buf.clone());
            }
        }

        if rows.is_empty() {
            self.finished = true;
            return Ok(None);
        }

        // Compose chunk for decode_csv
        let mut chunk = Vec::new();
        for line in &rows {
            chunk.extend_from_slice(line);
            chunk.push(b'\n');
        }

        let table = decode_csv(std::io::Cursor::new(chunk), &batch_options)?;

        // Capture schema after first batch
        if self.schema.is_none() {
            self.schema = Some(
                table
                    .cols
                    .iter()
                    .map(|c| c.field.as_ref().clone())
                    .collect(),
            );
        }

        // If fewer than batch_size data rows, mark as finished
        let effective_n_rows = table.n_rows;
        if effective_n_rows < self.batch_size {
            self.finished = true;
        }

        Ok(Some(table))
    }

    /// Consume the entire input and return a single Table containing
    /// every row decoded in one pass through `decode_csv`.
    pub fn load_table(mut self) -> io::Result<Table> {
        // Always respect has_header on first call
        decode_csv(&mut self.reader, &self.options)
    }

    /// Consume the entire input and return a `SuperTable` whose batches
    /// reflect the reader's `batch_size`. Successive `next_batch` calls
    /// are drained internally; consumers wanting one consolidated Table
    /// use [`Self::load_table`] instead.
    pub fn load_batched(mut self) -> io::Result<SuperTable> {
        let mut batches: Vec<Arc<Table>> = Vec::new();
        let mut name: Option<String> = None;
        while let Some(batch) = self.next_batch()? {
            if name.is_none() {
                name = Some(batch.name.clone());
            }
            batches.push(Arc::new(batch));
        }
        Ok(SuperTable::from_batches(batches, name.or(Some("csv".into()))))
    }
}

/// Count occurrences of `byte` in `haystack` using a SWAR (SIMD within
/// a register) zero-byte detection over 64-bit chunks. Used in the
/// "read whole chunk as one batch" path to size the row vector exactly
/// from a single pass, avoiding any Vec growth.
///
/// The zero-byte-detection pattern is the standard SWAR idiom from
/// Hacker's Delight (Warren, 2002): broadcast the needle across a
/// 64-bit lane, XOR with the chunk, then use
/// `(x - 0x01..01) & !x & 0x80..80` to set the high bit of every
/// matching byte position.
fn count_byte_swar(haystack: &[u8], byte: u8) -> usize {
    // Broadcast the target byte across every lane of a u64.
    let needle = u64::from_ne_bytes([byte; 8]);
    let mut count = 0usize;
    let mut chunks = haystack.chunks_exact(8);
    for chunk in &mut chunks {
        let bytes = u64::from_ne_bytes(<[u8; 8]>::try_from(chunk).unwrap());
        // Zero-byte detection: each lane of `xor` that is zero gets its
        // high bit set in `zeros`. Standard SWAR pattern.
        let xor = bytes ^ needle;
        let zeros = xor.wrapping_sub(0x0101_0101_0101_0101) & !xor & 0x8080_8080_8080_8080;
        count += zeros.count_ones() as usize;
    }
    for &b in chunks.remainder() {
        if b == byte {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decoders::csv::CsvDecodeOptions;
    use std::io::BufReader;

    #[test]
    fn test_csv_reader_full_table() {
        let csv = b"i,s,b\n1,hello,true\n2,,false\n3,world,1\n4,rust,0\n";
        let opts = CsvDecodeOptions::default();
        let reader = CsvReader::<BufReader<&[u8]>>::from_slice(csv, opts, 2);
        let table = reader.load_table().unwrap();
        assert_eq!(table.n_rows, 4);
        assert_eq!(table.cols.len(), 3);
    }

    #[test]
    fn test_csv_reader_batch_iter() {
        let csv = b"i,s,b\n1,hello,true\n2,,false\n3,world,1\n4,rust,0\n";
        let opts = CsvDecodeOptions::default();
        let mut reader = CsvReader::<BufReader<&[u8]>>::from_slice(csv, opts, 2);

        let mut total_rows = 0;
        while let Some(batch) = reader.next_batch().unwrap() {
            total_rows += batch.n_rows;
        }
        assert_eq!(total_rows, 4);
    }

    #[test]
    fn test_csv_reader_schema() {
        let csv = b"i,s\n1,hello\n2,world\n";
        let opts = CsvDecodeOptions::default();
        let mut reader = CsvReader::<BufReader<&[u8]>>::from_slice(csv, opts, 1);
        let fields = reader.schema().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "i");
        assert_eq!(fields[1].name, "s");
    }
}
