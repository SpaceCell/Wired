// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Arrow Table Writer
//!
//! Sync analogue of [`TableWriter`]. End-to-end blocking writer over any
//! `std::io::Write` sink.
//!
//! ## Where this fits among the IPC writers
//!
//! - [`TableStreamWriter`] - **frame producer**, sink-agnostic. Encodes a
//!   table into IPC frames and queues them; the caller pulls frames via
//!   `next_frame` and is responsible for getting bytes onto the wire.
//! - [`TableWriter`] - **async end-to-end writer**. Wraps `TableStreamWriter`
//!   and drives the queued frames into a `tokio::io::AsyncWrite` sink.
//! - [`SyncTableWriter`] (this struct) - **sync end-to-end writer**. Wraps
//!   `TableStreamWriter` and drives the queued frames into a
//!   `std::io::Write` sink. Use this for one-shot file writes from a sync
//!   caller (e.g. `ChunkedArrowWriter`'s per-batch path) so the caller does
//!   not have to spin up a tokio runtime to push bytes.
//!
//! All three handle both Stream and File IPC protocols; the protocol choice
//! controls the wire framing (continuation markers / EOS for Stream; magic +
//! footer for File), not whether the writer owns a sink.
//!
//! [`TableWriter`]: crate::models::writers::ipc::table::TableWriter
//! [`TableStreamWriter`]: crate::models::writers::ipc::table_stream::TableStreamWriter

use std::io;

use minarrow::{Field, Table, TableV, Vec64};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::writers::object_storage::StorageWriter;
use crate::models::writers::ipc::table_stream::TableStreamWriter;
use crate::traits::stream_buffer::StreamBuffer;
use crate::utils::dict_values;

/// Sync end-to-end Arrow Table writer over any `std::io::Write` sink.
///
/// Owns a [`TableStreamWriter`] for frame production and writes each emitted
/// frame to the sink before returning. Dictionaries can be registered up
/// front via [`Self::register_dictionary`] or are auto-detected from
/// categorical columns by [`Self::write_table`].
pub struct SyncTableWriter<W, B = Vec64<u8>>
where
    W: io::Write,
    B: StreamBuffer + Unpin + 'static,
{
    inner: TableStreamWriter<B>,
    sink: W,
    finished: bool,
}

impl<W, B> SyncTableWriter<W, B>
where
    W: io::Write,
    B: StreamBuffer + Unpin + 'static,
{
    /// Create a new sync table writer. Pass `None` for `compression` to
    /// write uncompressed batches; `Some(codec)` compresses every
    /// record-batch body.
    pub fn new(
        sink: W,
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
    ) -> Self {
        Self {
            inner: TableStreamWriter::new(schema, protocol, compression),
            sink,
            finished: false,
        }
    }

    /// Get the schema used for this writer.
    pub fn schema(&self) -> &[Field] {
        self.inner.schema()
    }

    /// Register a dictionary for categorical columns.
    ///
    /// Must be called before the first `write_table` for any column that
    /// uses dictionary encoding and is not auto-detected from the table.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.inner.register_dictionary(dict_id, values);
    }

    /// Write a single table or table view, auto-registering dictionaries
    /// from any categorical columns present, then drain encoded frames to
    /// the sink.
    pub fn write_table(&mut self, table: impl Into<TableV>) -> io::Result<()> {
        let view: TableV = table.into();
        for (col_idx, col) in view.cols.iter().enumerate() {
            if let Some(values) = dict_values(col.as_tuple_ref().0) {
                self.inner.register_dictionary(col_idx as i64, values);
            }
        }
        self.inner.write(view)?;
        self.drain_frames()
    }

    /// Write every table from the iterator, then finalise the stream.
    pub fn write_all_tables<I>(&mut self, tables: I) -> io::Result<()>
    where
        I: IntoIterator<Item = Table>,
    {
        for table in tables {
            self.write_table(table)?;
        }
        self.finish()
    }

    /// Finalise the stream. Emits the EOS marker for Stream protocol or the
    /// footer + closing magic for File protocol, drains any remaining
    /// frames, and flushes the sink. Idempotent.
    pub fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        self.inner.finish()?;
        self.drain_frames()?;
        self.sink.flush()?;
        self.finished = true;
        Ok(())
    }

    /// Consume the writer and return the underlying sink. Calls `finish` if
    /// it has not been called.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.finish()?;
        Ok(self.sink)
    }

    /// Pull every queued frame from the encoder and write it to the sink.
    fn drain_frames(&mut self) -> io::Result<()> {
        while let Some(frame) = self.inner.next_frame() {
            let buf = frame?;
            self.sink.write_all(buf.as_ref())?;
        }
        Ok(())
    }
}

/// Write a sequence of `Table`s to a file path in Arrow File format using a
/// blocking sink. Sync analogue of `write_tables_to_file`.
pub fn write_tables_to_file_sync(
    file_path: &std::path::Path,
    tables: &[Table],
    schema: Vec<Field>,
) -> io::Result<()> {
    let file = StorageWriter::create(file_path)?;
    let mut writer = SyncTableWriter::<_, Vec64<u8>>::new(file, schema, IPCMessageProtocol::File, None);
    for table in tables {
        writer.write_table(table.clone())?;
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::SyncTableWriter;
    use super::write_tables_to_file_sync;
    use crate::enums::IPCMessageProtocol;
    use minarrow::{Field, Table, Vec64, fa_i32};
    use tempfile::NamedTempFile;

    #[test]
    fn writes_arrow_file_round_trips_via_file_reader() {
        use crate::models::readers::ipc::file_table::FileTableReader;

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let table = Table::new("t", Some(vec![fa_i32!["n", 1, 2, 3]]));
        let schema: Vec<Field> = table.cols.iter().map(|c| (*c.field).clone()).collect();

        write_tables_to_file_sync(&path, std::slice::from_ref(&table), schema).unwrap();

        let reader = FileTableReader::open(&path).unwrap();
        assert_eq!(reader.num_batches(), 1);
        let got = reader.read_batch(0).unwrap();
        assert_eq!(got.n_rows, 3);
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn timestamp_columns_round_trip_with_unit_and_timezone() {
        use minarrow::enums::time_units::TimeUnit;
        use minarrow::{Array, ArrowType, DatetimeArray, Dt, FieldArray, TemporalArray};

        use crate::models::readers::ipc::file_table::FileTableReader;

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let ts = FieldArray::new(
            Field::new(
                "ts_event",
                ArrowType::DateTime(Dt::Timestamp(TimeUnit::Nanoseconds, Some("UTC".into()))),
                false,
                None,
            ),
            Array::from_datetime_i64(DatetimeArray::from_slice(
                &[1_782_999_000_000_000_000, 1_782_999_000_000_000_001],
                Dt::Timestamp(TimeUnit::Nanoseconds, Some("UTC".into())),
            )),
        );
        let table = Table::new("t", Some(vec![ts]));
        let schema: Vec<Field> = table.cols.iter().map(|c| (*c.field).clone()).collect();

        write_tables_to_file_sync(&path, std::slice::from_ref(&table), schema).unwrap();

        let reader = FileTableReader::open(&path).unwrap();
        let got = reader.read_batch(0).unwrap();
        assert_eq!(got.n_rows, 2);
        assert_eq!(
            got.cols[0].field.dtype,
            ArrowType::DateTime(Dt::Timestamp(TimeUnit::Nanoseconds, Some("UTC".into()))),
        );
        match &got.cols[0].array {
            Array::TemporalArray(TemporalArray::Datetime64(a)) => {
                assert_eq!(a.data.as_slice(), &[1_782_999_000_000_000_000, 1_782_999_000_000_000_001]);
                assert_eq!(a.dtype, Dt::Timestamp(TimeUnit::Nanoseconds, Some("UTC".into())));
            }
            other => panic!("timestamp column decoded as {other:?}"),
        }
    }

    #[test]
    fn writes_multiple_tables_via_file_protocol() {
        use crate::models::readers::ipc::file_table::FileTableReader;

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let t1 = Table::new("t", Some(vec![fa_i32!["n", 1, 2, 3]]));
        let t2 = Table::new("t", Some(vec![fa_i32!["n", 4, 5]]));
        let t3 = Table::new("t", Some(vec![fa_i32!["n", 6, 7, 8, 9]]));
        let schema: Vec<Field> = t1.cols.iter().map(|c| (*c.field).clone()).collect();

        let mut writer = SyncTableWriter::<_, Vec64<u8>>::new(
            std::fs::File::create(&path).unwrap(),
            schema,
            IPCMessageProtocol::File,
            None,
        );
        writer.write_table(t1.clone()).unwrap();
        writer.write_table(t2.clone()).unwrap();
        writer.write_table(t3.clone()).unwrap();
        writer.finish().unwrap();

        let reader = FileTableReader::open(&path).unwrap();
        assert_eq!(reader.num_batches(), 3);
        assert_eq!(reader.read_batch(0).unwrap().n_rows, 3);
        assert_eq!(reader.read_batch(1).unwrap().n_rows, 2);
        assert_eq!(reader.read_batch(2).unwrap().n_rows, 4);
    }

    #[test]
    fn writes_stream_protocol_to_buffer() {
        use crate::models::readers::ipc::table::TableReader;
        use futures_util::StreamExt;

        let t1 = Table::new("t", Some(vec![fa_i32!["n", 10, 20, 30]]));
        let t2 = Table::new("t", Some(vec![fa_i32!["n", 40, 50]]));
        let schema: Vec<Field> = t1.cols.iter().map(|c| (*c.field).clone()).collect();

        let mut buf: Vec<u8> = Vec::new();
        let mut writer =
            SyncTableWriter::<_, Vec64<u8>>::new(&mut buf, schema, IPCMessageProtocol::Stream, None);
        writer.write_table(t1.clone()).unwrap();
        writer.write_table(t2.clone()).unwrap();
        writer.finish().unwrap();
        drop(writer);

        // Round-trip via the async stream reader to confirm the wire bytes
        // are a valid Arrow IPC stream that anything else can decode.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut reader = TableReader::<Vec64<u8>>::new(
            std::io::Cursor::new(buf),
            8 * 1024,
            IPCMessageProtocol::Stream,
            None,
        );
        let batches: Vec<Table> = rt.block_on(async move {
            let mut out = Vec::new();
            while let Some(batch) = reader.next().await {
                out.push(batch.unwrap());
            }
            out
        });
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].n_rows, 3);
        assert_eq!(batches[1].n_rows, 2);
    }

    #[test]
    fn finish_is_idempotent() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = SyncTableWriter::<_, Vec64<u8>>::new(
            file.reopen().unwrap(),
            vec![],
            IPCMessageProtocol::Stream,
            None,
        );
        writer.finish().unwrap();
        writer.finish().unwrap();
    }
}
