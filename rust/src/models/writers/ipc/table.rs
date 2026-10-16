// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Asynchronous Arrow Table Writer
//!
//! High-level API for writing [`minarrow::Table`] values to files or streams
//! in Arrow IPC format using its `File` or `Stream` protocols.
//!
//! ## Features
//! - Wraps [`GTableSink`] for Arrow IPC framing and schema management
//! - Supports optional compression
//! - Automatically registers categorical dictionaries
//! - Provides helpers for writing one or many tables to disk
//!
//! ## Typical usage
//! - Create with [`TableWriter::new`] or `TableWriter::new_with_compression`
//! - Optionally register dictionaries with [`TableWriter::register_dictionary`]  
//! - Write tables using [`TableWriter::write_table`] or [`TableWriter::write_all_tables`]  
//! - Finalise with [`TableWriter::finish`]  

use std::io;

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::sinks::table_sink::GTableSink;
#[cfg(not(target_family = "wasm"))]
use crate::utils::dict_values;
use futures_util::sink::SinkExt;
use minarrow::{Field, Table, TableV};
#[cfg(not(target_family = "wasm"))]
use crate::models::writers::object_storage::AsyncStorageWriter;
use tokio::io::AsyncWrite;

use minarrow::Vec64;
use std::pin::Pin;

/// Main Table Writer
///
/// This struct provides all high-level async methods for writing Arrow tables,
/// managing schema, dictionaries, and protocol negotiation.
/// Internally delegates to a wrapped [`GTableSink`] instance.
pub struct TableWriter<W>
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    sink: GTableSink<W, Vec64<u8>>,
}

impl<W> TableWriter<W>
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    /// Create a new Arrow Table writer. Pass `None` for `compression` to
    /// write uncompressed batches; `Some(codec)` compresses every
    /// record-batch body.
    pub fn new(
        destination: W,
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        Ok(Self {
            sink: GTableSink::new(destination, schema, protocol, compression)?,
        })
    }

    /// Create a new Arrow Table writer from a Table's schema. Extracts
    /// the fields from `Table::schema()` directly and writes uncompressed.
    pub fn from_schema(
        destination: W,
        schema: Vec<std::sync::Arc<Field>>,
        protocol: IPCMessageProtocol,
    ) -> io::Result<Self> {
        let fields: Vec<Field> = schema.into_iter().map(|f| (*f).clone()).collect();
        Ok(Self {
            sink: GTableSink::new(destination, fields, protocol, None)?,
        })
    }

    /// Get the schema used for this writer.
    pub fn schema(&self) -> &[Field] {
        &self.sink.schema
    }

    /// Register a dictionary with the given id and values.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.sink.codec.register_dictionary(dict_id, values.clone());
        if let Some(writer) = &mut self.sink.file_writer {
            writer.register_dictionary(dict_id, values);
        }
    }

    /// Return the protocol in use (Stream or File).
    pub fn protocol(&self) -> IPCMessageProtocol {
        self.sink.protocol
    }

    /// Write a single table to the sink and flush all output.
    pub async fn write_table(&mut self, table: impl Into<TableV> + Send) -> io::Result<()> {
        SinkExt::send(&mut self.sink, table.into()).await?;
        SinkExt::flush(&mut self.sink).await?;
        Ok(())
    }

    /// Write all tables from an iterator to the sink and flush/close.
    pub async fn write_all_tables<I>(&mut self, tables: I) -> io::Result<()>
    where
        I: IntoIterator<Item = Table>,
    {
        let mut sink = Pin::new(&mut self.sink);
        for table in tables {
            SinkExt::send(&mut sink, table.into()).await?;
        }
        SinkExt::close(&mut sink).await?;
        Ok(())
    }

    /// Closes the sink. Must be done after writing all of the tables.
    pub async fn finish(&mut self) -> io::Result<()> {
        SinkExt::close(&mut self.sink).await
    }
}

/// Write a sequence of `Table`s to disk in Arrow File format.
///
/// * `file_path`   - where to create/write the .arrow file  
/// * `tables`      - the batches to write (each a `Table`)  
/// * `schema`      - the common schema (must match each `Table`)  
/// * `protocol`    - usually `IPCMessageProtocol::File`  
#[cfg(not(target_family = "wasm"))]
pub async fn write_tables_to_file(
    file_path: &str,
    tables: &[Table],
    schema: Vec<Field>,
) -> io::Result<()> {
    let file = AsyncStorageWriter::create(file_path).await?;
    let mut writer = TableWriter::new(file, schema, IPCMessageProtocol::File, None)?;
    // Automatically register any Categorical dictionaries found in the tables.
    for table in tables {
        for (col_idx, col) in table.cols.iter().enumerate() {
            if let Some(values) = dict_values(&col.array) {
                // We use the column index as the unique dictionary key
                writer.register_dictionary(col_idx as i64, values);
            }
        }
    }
    writer.write_all_tables(tables.to_vec()).await?;
    Ok(())
}

/// Writes a single table to a file
#[cfg(not(target_family = "wasm"))]
pub async fn write_table_to_file(
    file_path: &str,
    table: &Table,
    schema: Vec<Field>,
) -> io::Result<()> {
    let file = AsyncStorageWriter::create(file_path).await?;
    let mut writer = TableWriter::new(file, schema, IPCMessageProtocol::File, None)?;
    // Automatically register any Categorical dictionaries found in the table.
    for (col_idx, col) in table.cols.iter().enumerate() {
        if let Some(values) = dict_values(&col.array) {
            writer.register_dictionary(col_idx as i64, values);
        }
    }
    writer.write_table(table.clone()).await?;
    // Finishing writes the file footer and completes an object upload.
    writer.finish().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use minarrow::{
        Array, ArrowType, Bitmask, Buffer, CategoricalArray, Field, FieldArray, Table, TextArray,
        Vec64,
    };
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tokio::fs::File;
    use tokio::io::AsyncReadExt;

    fn dict_strs() -> Vec<String> {
        vec![
            "apple".to_string(),
            "banana".to_string(),
            "pear".to_string(),
        ]
    }

    fn make_bitmask(valid: &[bool]) -> Bitmask {
        let mut bits = vec![0u8; (valid.len() + 7) / 8];
        for (i, v) in valid.iter().enumerate() {
            if *v {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        Bitmask::new(Buffer::from(Vec64::from_slice(&bits[..])), valid.len())
    }

    #[cfg(not(feature = "default_categorical_8"))]
    fn make_schema() -> Vec<Field> {
        vec![Field {
            name: "col".to_string(),
            dtype: ArrowType::Dictionary(minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt32),
            nullable: true,
            metadata: Default::default(),
        }]
    }

    #[cfg(feature = "default_categorical_8")]
    fn make_schema() -> Vec<Field> {
        vec![Field {
            name: "col".to_string(),
            dtype: ArrowType::Dictionary(minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt8),
            nullable: true,
            metadata: Default::default(),
        }]
    }

    #[cfg(not(feature = "default_categorical_8"))]
    fn make_table() -> Table {
        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u32, 0, 2, 1])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, false, true, true])),
        );
        Table {
            cols: vec![FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(
                        minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt32,
                    ),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical32(Arc::new(arr))),
            )],
            n_rows: 4,
            name: "tbl".to_string(),
        }
    }

    #[cfg(feature = "default_categorical_8")]
    fn make_table() -> Table {
        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u8, 0, 2, 1])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, false, true, true])),
        );
        Table {
            cols: vec![FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(
                        minarrow::ffi::arrow_dtype::CategoricalIndexType::UInt8,
                    ),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical8(Arc::new(arr))),
            )],
            n_rows: 4,
            name: "tbl".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_table_writer_file_protocol() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(file, schema.clone(), IPCMessageProtocol::File, None).unwrap();
        writer.register_dictionary(0, dict_strs());

        let tbl = make_table();
        // now write *and* finish in one call
        writer.write_all_tables(vec![tbl]).await.unwrap();

        // Validate: read file and check it's not empty and starts/ends with Arrow magic.
        let mut file = File::open(&path).await.unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).await.unwrap();
        assert!(!buf.is_empty());
        // Arrow "ARROW1\0\0" magic
        assert!(buf.starts_with(b"ARROW1\0\0"));
        assert!(buf.ends_with(b"ARROW1"));
    }

    #[tokio::test]
    async fn test_table_writer_stream_protocol() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(file, schema, IPCMessageProtocol::Stream, None).unwrap();
        writer.register_dictionary(0, dict_strs());

        let tbl = make_table();
        writer.write_all_tables(vec![tbl]).await.unwrap();

        let mut file = File::open(&path).await.unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).await.unwrap();
        assert!(!buf.is_empty());
        // Stream protocol starts with 0xFFFF_FFFF
        assert_eq!(&buf[..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[tokio::test]
    async fn test_finish_idempotent() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(file, schema, IPCMessageProtocol::Stream, None).unwrap();
        writer.register_dictionary(0, dict_strs());

        let tbl = make_table();
        // first write+close
        writer.write_all_tables(vec![tbl]).await.unwrap();
        // second close with no new tables
        writer.write_all_tables(Vec::<Table>::new()).await.unwrap();
    }

    #[tokio::test]
    async fn test_simd_aligned_table_writer() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(file, schema, IPCMessageProtocol::File, None).unwrap();
        writer.register_dictionary(0, dict_strs());

        let tbl = make_table();
        writer.write_all_tables(vec![tbl]).await.unwrap();

        let mut file = File::open(&path).await.unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).await.unwrap();
        assert!(!buf.is_empty());
        assert!(buf.starts_with(b"ARROW1\0\0"));
        assert!(buf.ends_with(b"ARROW1"));
    }

    #[tokio::test]
    async fn test_error_handling_invalid_sink() {
        // Use an io::sink, which always returns Ok, so this just checks for panics.
        let schema = make_schema();
        let sink = tokio::io::sink();
        let mut writer = TableWriter::new(sink, schema, IPCMessageProtocol::File, None).unwrap();
        // writing nothing should simply close without error
        writer.write_all_tables(Vec::<Table>::new()).await.unwrap();
    }

    #[tokio::test]
    async fn test_with_compression_none() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(
            file,
            schema.clone(),
            IPCMessageProtocol::File,
            None,
        )
        .unwrap();
        writer.register_dictionary(0, dict_strs());

        let tbl = make_table();
        writer.write_all_tables(vec![tbl]).await.unwrap();

        // Validate: read file and check it's not empty and starts/ends with Arrow magic.
        let mut file = File::open(&path).await.unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).await.unwrap();
        assert!(!buf.is_empty());
        assert!(buf.starts_with(b"ARROW1\0\0"));
        assert!(buf.ends_with(b"ARROW1"));
    }

    #[cfg(feature = "snappy")]
    #[tokio::test]
    async fn test_snappy_rejected_for_ipc() {
        // Arrow IPC BodyCompression permits only LZ4_FRAME and ZSTD, so the
        // IPC writer must reject Snappy rather than emit a non-conformant
        // file.
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(
            file,
            schema.clone(),
            IPCMessageProtocol::File,
            Some(Compression::Snappy),
        )
        .unwrap();
        writer.register_dictionary(0, dict_strs());

        let err = writer
            .write_all_tables(vec![make_table()])
            .await
            .expect_err("Snappy is not a valid Arrow IPC body codec");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[cfg(feature = "zstd")]
    #[tokio::test]
    async fn test_with_compression_zstd() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let file = File::create(&path).await.unwrap();
        let schema = make_schema();
        let mut writer = TableWriter::new(
            file,
            schema.clone(),
            IPCMessageProtocol::File,
            Some(Compression::Zstd),
        )
        .unwrap();
        writer.register_dictionary(0, dict_strs());

        let tbl = make_table();
        writer.write_all_tables(vec![tbl]).await.unwrap();

        // Validate: read file and check it's not empty and has Arrow magic
        let mut file = File::open(&path).await.unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).await.unwrap();
        assert!(!buf.is_empty());
        assert!(buf.starts_with(b"ARROW1\0\0"));
        assert!(buf.ends_with(b"ARROW1"));

        // For compressed files, we expect the content to be different from uncompressed
        // but still valid Arrow format
        println!("Zstd compressed file size: {} bytes", buf.len());
    }

    #[tokio::test]
    async fn test_compression_api_equivalence() {
        // Test that TableWriter::new and TableWriter::new_with_compression(Compression::None)
        // produce equivalent results
        let temp1 = NamedTempFile::new().unwrap();
        let temp2 = NamedTempFile::new().unwrap();
        let path1 = temp1.path().to_path_buf();
        let path2 = temp2.path().to_path_buf();

        let schema = make_schema();
        let tbl = make_table();

        // Write with regular constructor
        {
            let file = File::create(&path1).await.unwrap();
            let mut writer =
                TableWriter::new(file, schema.clone(), IPCMessageProtocol::File, None).unwrap();
            writer.register_dictionary(0, dict_strs());
            writer.write_all_tables(vec![tbl.clone()]).await.unwrap();
        }

        // Write with compression = None
        {
            let file = File::create(&path2).await.unwrap();
            let mut writer = TableWriter::new(
                file,
                schema.clone(),
                IPCMessageProtocol::File,
                None,
            )
            .unwrap();
            writer.register_dictionary(0, dict_strs());
            writer.write_all_tables(vec![tbl]).await.unwrap();
        }

        // Read both files and compare
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        File::open(&path1)
            .await
            .unwrap()
            .read_to_end(&mut buf1)
            .await
            .unwrap();
        File::open(&path2)
            .await
            .unwrap()
            .read_to_end(&mut buf2)
            .await
            .unwrap();

        // Both should be valid Arrow files
        assert!(!buf1.is_empty());
        assert!(!buf2.is_empty());
        assert!(buf1.starts_with(b"ARROW1\0\0"));
        assert!(buf2.starts_with(b"ARROW1\0\0"));

        // Content should be identical since both are uncompressed
        assert_eq!(buf1, buf2);
    }
}
