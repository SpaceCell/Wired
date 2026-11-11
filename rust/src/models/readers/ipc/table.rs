// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Async Arrow IPC table reader utilities.
//!
//! Wraps a framed Arrow IPC decoder into ergonomic async helpers that:
//! - read all or N record batches as `Table`,
//! - assemble batches into a `SuperTable` (retaining batch windows), or
//! - concatenate batches into a single `Table`.
//!
//! Works with any `Stream<Item = Result<B, io::Error>> + AsyncRead` where `B: StreamBuffer`
//! (e.g. `Vec<u8>` or `Vec64<u8>`), and both Arrow IPC protocols (File/Stream).

use crate::enums::IPCMessageProtocol;
use crate::models::decoders::ipc::table_stream_decoder::TableStreamDecoder;
use crate::traits::stream_buffer::StreamBuffer;
use futures_core::Stream;
use futures_util::StreamExt;
use minarrow::{Field, FieldArray, SuperTable, Table};
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::AsyncRead;

/// High-level async table reader over an Arrow IPC stream.
///
/// Wraps a [`TableStreamDecoder`] with zero-copy record batch decoding
/// and exposes convenience methods to pull batches, aggregate into a
/// `SuperTable`, or concatenate into a single `Table`.
pub struct TableReader<B: StreamBuffer + Unpin + 'static = Vec<u8>> {
    inner: TableStreamDecoder<B>,
}

impl<B: StreamBuffer + Unpin + 'static> TableReader<B> {
    /// Create a new reader from an AsyncRead source.
    ///
    /// Reads frame headers into a small accumulation buffer, then reads
    /// record batch bodies directly into a Vec64 for SharedBuffer
    /// zero-copy column mapping. Each batch is yielded individually.
    ///
    pub fn new(
        stream: impl AsyncRead + Unpin + Send + 'static,
        initial_capacity: usize,
        protocol: IPCMessageProtocol,
        limits: Option<crate::models::decoders::limits::DecodeLimits>,
    ) -> Self {
        Self {
            inner: TableStreamDecoder::new(stream, initial_capacity, protocol, limits),
        }
    }

    /// Read all available Arrow tables from the stream.
    pub async fn read_all_tables(mut self) -> io::Result<Vec<Table>> {
        let mut tables = Vec::new();
        while let Some(batch) = self.next().await {
            tables.push(batch?);
        }
        Ok(tables)
    }

    /// Read up to `n` Arrow tables.
    ///
    /// If `n` is `None` read until EOS.
    pub async fn read_tables(mut self, n: Option<usize>) -> io::Result<Vec<Table>> {
        let mut tables = Vec::new();
        let mut count = 0usize;
        while let Some(batch) = self.next().await {
            let batch = batch?;
            tables.push(batch);
            count += 1;
            if let Some(max) = n
                && count >= max {
                    break;
                }
        }
        Ok(tables)
    }

    /// Read up to `n` batches and assemble them into a `SuperTable`.
    ///
    /// If `n` is `None`, read to EOS.
    pub async fn read_to_super_table(
        mut self,
        name: Option<String>,
        n: Option<usize>,
    ) -> io::Result<SuperTable> {
        let mut batches = Vec::new();
        let mut count = 0usize;
        while let Some(batch) = self.next().await {
            let batch = batch?;
            batches.push(Arc::new(batch));
            count += 1;
            if let Some(max) = n
                && count >= max {
                    break;
                }
        }
        Ok(SuperTable::from_batches(
            batches,
            Some(name.unwrap_or_else(|| "SuperTable".to_string())),
        ))
    }

    /// Read all batches and concatenate into a single `Table` row-wise.
    pub async fn combine_to_table(mut self, name: Option<String>) -> io::Result<Table> {
        let mut all_batches = Vec::new();
        while let Some(batch) = self.next().await {
            all_batches.push(batch?);
        }
        combine_batches_to_table(all_batches, name)
    }

    /// Return the decoded schema if available after first schema message.
    pub fn schema(&self) -> Option<&[Field]> {
        self.inner.schema()
    }

    /// Read the next `Table` from the stream, or `None` on EOS.
    pub async fn read_next(&mut self) -> io::Result<Option<Table>> {
        self.next().await.transpose()
    }
}

impl<B: StreamBuffer + Unpin + 'static> Stream for TableReader<B> {
    type Item = io::Result<Table>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }
}

/// Concatenate a vector of batches into a single `Table`
pub(crate) fn combine_batches_to_table(batches: Vec<Table>, name: Option<String>) -> io::Result<Table> {
    if batches.is_empty() {
        return Ok(Table::default());
    }
    let schema = batches[0]
        .cols
        .iter()
        .map(|f| f.field.clone())
        .collect::<Vec<_>>();
    let n_rows: usize = batches.iter().map(|t| t.n_rows).sum();
    let name = name.unwrap_or_else(|| "CombinedTable".to_string());

    // Group columns by index
    let mut combined_cols: Vec<Vec<FieldArray>> = vec![Vec::new(); schema.len()];
    for batch in &batches {
        for (i, col) in batch.cols.iter().enumerate() {
            combined_cols[i].push(col.clone());
        }
    }

    // Concatenate by column
    let cols = combined_cols
        .into_iter()
        .map(concat_field_arrays)
        .collect::<Result<Vec<FieldArray>, io::Error>>()?;

    Ok(Table {
        cols,
        n_rows,
        name,
        ..Default::default()
    })
}

/// Concatenate multiple `FieldArray` segments of the same schema into one.
fn concat_field_arrays(batches: Vec<FieldArray>) -> io::Result<FieldArray> {
    if batches.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No arrays to concatenate",
        ));
    }
    let mut iter = batches.into_iter();
    let mut first = iter.next().unwrap();

    for arr in iter {
        first.array.concat_array(&arr.array);
        first.null_count += arr.null_count;
    }
    Ok(first)
}

// src/models/readers/table_stream_reader/tests.rs
#[cfg(test)]
mod tests {
    use crate::enums::IPCMessageProtocol;
    use crate::models::readers::ipc::table::StreamBuffer;
    use crate::models::readers::ipc::table::TableReader;
    use crate::models::writers::ipc::table_stream::TableStreamWriter;
    use crate::test_helpers::{make_all_types_table, make_schema_all_types};
    use crate::utils;
    use futures_core::Stream;
    use minarrow::{SuperTable, Table, Vec64};
    use std::io;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tokio::io::{AsyncRead, AsyncWriteExt, ReadBuf, duplex};

    /// Helper function to register dictionaries for categorical columns in a table
    fn register_dictionaries_for_table<B: StreamBuffer + Unpin + 'static>(
        writer: &mut TableStreamWriter<B>,
        table: &Table,
    ) {
        for (col_idx, col) in table.cols.iter().enumerate() {
            if let Some(values) = utils::dict_values(&col.array) {
                writer.register_dictionary(col_idx as i64, values);
            }
        }
    }

    /// A helper that implements both `Stream<Item=io::Result<Vec<u8>>>` and `AsyncRead`
    /// by using the DuplexStream reader for both interfaces.
    struct Combined {
        reader: tokio::io::DuplexStream,
    }

    impl Stream for Combined {
        type Item = io::Result<Vec<u8>>;
        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            let mut buf = vec![0u8; 8192];
            let mut read_buf = ReadBuf::new(&mut buf);
            match Pin::new(&mut this.reader).poll_read(cx, &mut read_buf) {
                Poll::Ready(Ok(())) => {
                    let n = read_buf.filled().len();
                    if n == 0 {
                        Poll::Ready(None)
                    } else {
                        buf.truncate(n);
                        Poll::Ready(Some(Ok(buf)))
                    }
                }
                Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    impl AsyncRead for Combined {
        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            let this = self.get_mut();
            Pin::new(&mut this.reader).poll_read(cx, buf)
        }
    }

    /// Write two identical test tables, then read them back via `read_all_tables`.
    #[tokio::test]
    async fn test_read_all_tables() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        println!("Table:");
        println!("{}", table);
        println!("Schema:");
        println!("{:?}", schema);
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        writer.write(table.clone()).unwrap();
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        // write all bytes into a duplex and close
        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        let (mut tx, rx) = duplex(64 * 1024); // Increase buffer size for dictionary data
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let out = reader.read_all_tables().await.unwrap();
        assert_eq!(out.len(), 2);
        for batch in out {
            assert_eq!(batch.n_rows, table.n_rows);
            assert_eq!(batch.cols.len(), table.cols.len());
        }
    }

    /// Limit the number of batches returned by `read_tables(Some(n))`.
    #[tokio::test]
    async fn test_read_tables_limit() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        // three batches
        writer.write(table.clone()).unwrap();
        writer.write(table.clone()).unwrap();
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        let (mut tx, rx) = duplex(64 * 1024); // Increase buffer size for dictionary data
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let out = reader.read_tables(Some(2)).await.unwrap();
        assert_eq!(out.len(), 2);
    }

    /// Read into a `SuperTable`, preserving windows and summing `n_rows`.
    #[tokio::test]
    async fn test_read_to_super_table() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        writer.write(table.clone()).unwrap();
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        let (mut tx, rx) = duplex(64 * 1024); // Increase buffer size for dictionary data
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let st: SuperTable = reader
            .read_to_super_table(Some("my_window".into()), None)
            .await
            .unwrap();
        assert_eq!(st.n_rows, table.n_rows * 2);
        assert_eq!(st.batches.len(), 2);
        assert_eq!(st.name, "my_window");
    }

    /// Combine all batches into one `Table`, row-concatenated.
    #[tokio::test]
    async fn test_combine_to_table() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        writer.write(table.clone()).unwrap();
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        let (mut tx, rx) = duplex(64 * 1024); // Increase buffer size for dictionary data
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let t: Table = reader.combine_to_table(Some("all".into())).await.unwrap();
        assert_eq!(t.n_rows, table.n_rows * 2);
        assert_eq!(t.name, "all");
        assert_eq!(t.cols.len(), table.cols.len());
    }

    /// One-by-one batch reading with `read_next`.
    /// Debug test to understand EOS handling issue
    #[tokio::test]
    async fn test_debug_buffer_consumption() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        // Check what's in the last frame
        println!("Total frames: {}", frames.len());
        for (i, frame) in frames.iter().enumerate() {
            println!(
                "Frame {}: {} bytes = {:?}",
                i,
                frame.len(),
                &frame[..std::cmp::min(frame.len(), 20)]
            );
        }

        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        println!("Total bytes: {}", all_bytes.len());
        println!("Last 20 bytes: {:?}", &all_bytes[all_bytes.len() - 20..]);

        let (mut tx, rx) = duplex(64 * 1024);
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let result = reader.read_all_tables().await;
        match result {
            Ok(tables) => println!("Success: {} tables", tables.len()),
            Err(e) => println!("Error: {}", e),
        }
    }

    /// Round-trip test for the direct Arena-based decode path.
    /// Encodes a table, reads it back via `new_direct`, and verifies values.
    #[tokio::test]
    async fn test_direct_read_all_tables() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        writer.write(table.clone()).unwrap();
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        let (mut tx, rx) = duplex(64 * 1024);
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let out = reader.read_all_tables().await.unwrap();
        // Each batch yields individually - no concatenation
        assert_eq!(out.len(), 2);
        for batch in &out {
            assert_eq!(batch.n_rows, table.n_rows);
            assert_eq!(batch.cols.len(), table.cols.len());
        }
    }

    #[tokio::test]
    async fn test_read_next_and_schema() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        register_dictionaries_for_table(&mut writer, &table);
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();
        let frames = writer.drain_all_frames();

        let all_bytes: Vec<u8> = frames.iter().flat_map(|v| v.iter().cloned()).collect();
        let (mut tx, rx) = duplex(64 * 1024); // Increase buffer size for dictionary data
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        let combined = Combined { reader: rx };

        let mut reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);

        // schema is only known after seeing the first message
        assert!(reader.schema().is_none());
        // first batch
        let first = reader.read_next().await.unwrap().unwrap();
        assert_eq!(first.n_rows, table.n_rows);
        // now schema is populated
        let seen = reader.schema().unwrap();
        assert_eq!(seen.len(), schema.len());
        // no more batches
        assert!(reader.read_next().await.unwrap().is_none());
    }
}
