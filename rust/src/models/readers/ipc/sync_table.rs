// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Arrow Table Reader
//!
//! Sync analogue of [`TableReader`]. Blocking reader over any
//! `std::io::Read` source, and the read-side counterpart of
//! [`SyncTableWriter`].
//!
//! ## Context
//!
//! Sync readers suit callers that run a stream on a dedicated thread with
//! no async runtime, such as high-volume feeds between processes. The reader
//! decodes through [`TableDecoder`], as [`TableReader`] does, and fills it
//! with blocking reads. Its read path doesn't run any async code. Both
//! readers accept the same input, reject the same malformed or oversized
//! input, and yield tables backed by 64-byte aligned buffers.
//!
//! ## Usage
//!
//! `SyncTableReader` implements `Iterator<Item = io::Result<Table>>`, with
//! one item per record batch until the end-of-stream marker.
//!
//! ```rust,no_run
//! use lightstream::enums::IPCMessageProtocol;
//! use lightstream::models::readers::ipc::sync_table::SyncTableReader;
//! use minarrow::Vec64;
//!
//! # fn run() -> std::io::Result<()> {
//! let file = std::fs::File::open("data.arrows")?;
//! let reader =
//!     SyncTableReader::<_, Vec64<u8>>::new(file, 64 * 1024, IPCMessageProtocol::Stream, None);
//! for table in reader {
//!     let table = table?;
//!     // process each batch as it arrives
//! }
//! # Ok(()) }
//! ```
//!
//! [`TableReader`]: crate::models::readers::ipc::table::TableReader
//! [`TableDecoder`]: crate::models::decoders::ipc::table_decoder::TableDecoder
//! [`SyncTableWriter`]: crate::models::writers::ipc::sync_table::SyncTableWriter

use std::io::{self, Read};
use std::sync::Arc;

use minarrow::{Field, SuperTable, Table};

use crate::enums::IPCMessageProtocol;
use crate::models::decoders::ipc::table_decoder::{TableDecodeResult, TableDecoder};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::readers::ipc::table::combine_batches_to_table;
use crate::traits::stream_buffer::StreamBuffer;

/// Largest read issued to a blocking source in one call.
///
/// `std::io::Read` writes into initialised memory, which requires zeroing
/// each read target first. The cap keeps that region cache-sized, and is
/// above the amount a socket read typically returns in one call.
const MAX_BLOCKING_READ: usize = 256 * 1024;

/// Sync Arrow table reader over any `std::io::Read` source.
///
/// Decodes an Arrow IPC stream into one `Table` per record batch,
/// with zero-copy column mapping onto 64-byte aligned buffers. Reads block
/// the calling thread until bytes arrive.
///
/// The `B` parameter controls frame alignment and must match what the
/// encoder used. Use `Vec64<u8>` for streams written with 64-byte alignment,
/// such as those from [`SyncTableWriter`] and the transport writers.
///
/// [`SyncTableWriter`]: crate::models::writers::ipc::sync_table::SyncTableWriter
pub struct SyncTableReader<R, B = Vec<u8>>
where
    R: Read,
    B: StreamBuffer + Unpin + 'static,
{
    source: R,
    decoder: TableDecoder<B>,
}

impl<R, B> SyncTableReader<R, B>
where
    R: Read,
    B: StreamBuffer + Unpin + 'static,
{
    /// Create a new reader from a `std::io::Read` source.
    ///
    /// Reads frame headers into a small accumulation buffer of
    /// `initial_capacity` bytes, then reads record batch bodies into the
    /// body arena for zero-copy column mapping. Pass `None` for `limits` to
    /// apply the default [`DecodeLimits`].
    pub fn new(
        source: R,
        initial_capacity: usize,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self {
        Self {
            source,
            decoder: TableDecoder::new(initial_capacity, protocol, limits),
        }
    }

    /// Read all available Arrow tables from the source.
    pub fn read_all_tables(self) -> io::Result<Vec<Table>> {
        self.collect()
    }

    /// Read up to `n` Arrow tables.
    ///
    /// If `n` is `None` read until EOS.
    pub fn read_tables(self, n: Option<usize>) -> io::Result<Vec<Table>> {
        self.take(n.unwrap_or(usize::MAX)).collect()
    }

    /// Read up to `n` batches and assemble them into a `SuperTable`.
    ///
    /// If `n` is `None`, read to EOS.
    pub fn read_to_super_table(
        self,
        name: Option<String>,
        n: Option<usize>,
    ) -> io::Result<SuperTable> {
        let batches = self
            .take(n.unwrap_or(usize::MAX))
            .map(|batch| batch.map(Arc::new))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(SuperTable::from_batches(
            batches,
            Some(name.unwrap_or_else(|| "SuperTable".to_string())),
        ))
    }

    /// Read all batches and concatenate into a single `Table` row-wise.
    pub fn combine_to_table(self, name: Option<String>) -> io::Result<Table> {
        combine_batches_to_table(self.read_all_tables()?, name)
    }

    /// Return the decoded schema if available after first schema message.
    pub fn schema(&self) -> Option<&[Field]> {
        self.decoder.schema()
    }

    /// Read the next `Table` from the source, or `None` on EOS.
    pub fn read_next(&mut self) -> io::Result<Option<Table>> {
        loop {
            match self.decoder.decode_next()? {
                TableDecodeResult::Batch(table, _) => return Ok(Some(table)),
                TableDecodeResult::EndOfStream => return Ok(None),
                TableDecodeResult::NeedMore => self.fill()?,
            }
        }
    }

    /// Return a reference to the underlying source.
    pub fn get_ref(&self) -> &R {
        &self.source
    }

    /// Read once from the source into the region the decoder requires next.
    ///
    /// Blocks until the source returns bytes, the end of its data or an
    /// error. At most `MAX_BLOCKING_READ` bytes of the region are zeroed and
    /// offered to the source.
    fn fill(&mut self) -> io::Result<()> {
        let region = self.decoder.spare_uninit();
        let len = region.len().min(MAX_BLOCKING_READ);
        // SAFETY: the first `len` elements are zeroed before the slice is
        // reinterpreted as initialised bytes.
        let target = unsafe {
            std::ptr::write_bytes(region.as_mut_ptr(), 0, len);
            std::slice::from_raw_parts_mut(region.as_mut_ptr().cast::<u8>(), len)
        };
        loop {
            match self.source.read(target) {
                Ok(0) => {
                    self.decoder.end_input();
                    return Ok(());
                }
                Ok(n) => {
                    // SAFETY: the first `len` bytes of the region are
                    // initialised, and the count is clamped to `len`.
                    unsafe { self.decoder.advance(n.min(len)) };
                    return Ok(());
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.decoder.end_input();
                    return Err(e);
                }
            }
        }
    }
}

impl<R, B> Iterator for SyncTableReader<R, B>
where
    R: Read,
    B: StreamBuffer + Unpin + 'static,
{
    type Item = io::Result<Table>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_next().transpose()
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Read};

    use futures_util::StreamExt;
    use minarrow::{Field, Table, Vec64, fa_i32};

    use super::SyncTableReader;
    use crate::enums::IPCMessageProtocol;
    use crate::models::decoders::limits::DecodeLimits;
    use crate::models::readers::ipc::table::TableReader;
    use crate::models::writers::ipc::sync_table::SyncTableWriter;
    use crate::test_helpers::{make_all_types_table, make_schema_all_types};

    fn encode(tables: &[Table], schema: Vec<Field>, protocol: IPCMessageProtocol) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut writer = SyncTableWriter::<_, Vec64<u8>>::new(&mut buf, schema, protocol, None);
        for table in tables {
            writer.write_table(table.clone()).unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        buf
    }

    /// Source that returns at most `step` bytes per read, exercising partial
    /// reads across frame headers and bodies.
    struct Trickle {
        inner: Cursor<Vec<u8>>,
        step: usize,
    }

    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.step);
            self.inner.read(&mut buf[..n])
        }
    }

    #[test]
    fn reads_every_batch_with_equal_contents() {
        let table = make_all_types_table();
        let bytes = encode(
            &[table.clone(), table.clone()],
            make_schema_all_types(),
            IPCMessageProtocol::Stream,
        );

        let reader = SyncTableReader::<_, Vec64<u8>>::new(
            Cursor::new(bytes),
            1024,
            IPCMessageProtocol::Stream,
            None,
        );
        let out = reader.read_all_tables().unwrap();
        assert_eq!(out.len(), 2);
        for batch in &out {
            assert_eq!(batch.n_rows, table.n_rows);
            assert_eq!(batch.cols, table.cols);
        }
    }

    #[test]
    fn decodes_the_same_tables_as_the_async_reader() {
        let table = make_all_types_table();
        let bytes = encode(
            &[table.clone(), table.clone(), table],
            make_schema_all_types(),
            IPCMessageProtocol::Stream,
        );

        let sync_tables = SyncTableReader::<_, Vec64<u8>>::new(
            Cursor::new(bytes.clone()),
            1024,
            IPCMessageProtocol::Stream,
            None,
        )
        .read_all_tables()
        .unwrap();

        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let async_tables = rt
            .block_on(
                TableReader::<Vec64<u8>>::new(
                    Cursor::new(bytes),
                    1024,
                    IPCMessageProtocol::Stream,
                    None,
                )
                .read_all_tables(),
            )
            .unwrap();

        assert_eq!(sync_tables.len(), async_tables.len());
        for (s, a) in sync_tables.iter().zip(&async_tables) {
            assert_eq!(s.n_rows, a.n_rows);
            assert_eq!(s.cols, a.cols);
        }
    }

    #[test]
    fn partial_reads_decode_every_batch() {
        let t1 = Table::new("t", Some(vec![fa_i32!["n", 1, 2, 3]]));
        let t2 = Table::new("t", Some(vec![fa_i32!["n", 4, 5]]));
        let schema: Vec<Field> = t1.cols.iter().map(|c| (*c.field).clone()).collect();
        let bytes = encode(&[t1.clone(), t2.clone()], schema, IPCMessageProtocol::Stream);

        let source = Trickle {
            inner: Cursor::new(bytes),
            step: 7,
        };
        let out = SyncTableReader::<_, Vec64<u8>>::new(source, 16, IPCMessageProtocol::Stream, None)
            .read_all_tables()
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].cols, t1.cols);
        assert_eq!(out[1].cols, t2.cols);
    }

    #[test]
    fn decoded_columns_are_64_byte_aligned() {
        let table = Table::new("t", Some(vec![fa_i32!["n", 1, 2, 3, 4, 5]]));
        let schema: Vec<Field> = table.cols.iter().map(|c| (*c.field).clone()).collect();
        let bytes = encode(&[table.clone(), table], schema, IPCMessageProtocol::Stream);

        let reader = SyncTableReader::<_, Vec64<u8>>::new(
            Cursor::new(bytes),
            1024,
            IPCMessageProtocol::Stream,
            None,
        );
        for batch in reader {
            let batch = batch.unwrap();
            match &batch.cols[0].array {
                minarrow::Array::NumericArray(minarrow::NumericArray::Int32(a)) => {
                    assert_eq!(a.data.as_slice().as_ptr() as usize % 64, 0);
                }
                other => panic!("expected Int32, got {other:?}"),
            }
        }
    }

    #[test]
    fn read_next_and_schema() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let bytes = encode(std::slice::from_ref(&table), schema.clone(), IPCMessageProtocol::Stream);

        let mut reader = SyncTableReader::<_, Vec64<u8>>::new(
            Cursor::new(bytes),
            1024,
            IPCMessageProtocol::Stream,
            None,
        );
        // The schema is known once the first message has been read.
        assert!(reader.schema().is_none());
        let first = reader.read_next().unwrap().unwrap();
        assert_eq!(first.n_rows, table.n_rows);
        assert_eq!(reader.schema().unwrap().len(), schema.len());
        assert!(reader.read_next().unwrap().is_none());
    }

    #[test]
    fn read_tables_super_table_and_combine() {
        let table = make_all_types_table();
        let schema = make_schema_all_types();
        let bytes = encode(
            &[table.clone(), table.clone(), table.clone()],
            schema,
            IPCMessageProtocol::Stream,
        );
        let reader = || {
            SyncTableReader::<_, Vec64<u8>>::new(
                Cursor::new(bytes.clone()),
                1024,
                IPCMessageProtocol::Stream,
                None,
            )
        };

        assert_eq!(reader().read_tables(Some(2)).unwrap().len(), 2);

        let st = reader()
            .read_to_super_table(Some("window".into()), None)
            .unwrap();
        assert_eq!(st.n_rows, table.n_rows * 3);
        assert_eq!(st.batches.len(), 3);
        assert_eq!(st.name, "window");

        let combined = reader().combine_to_table(Some("all".into())).unwrap();
        assert_eq!(combined.n_rows, table.n_rows * 3);
        assert_eq!(combined.name, "all");
    }

    #[test]
    fn truncated_stream_is_an_error() {
        let table = Table::new("t", Some(vec![fa_i32!["n", 1, 2, 3]]));
        let schema: Vec<Field> = table.cols.iter().map(|c| (*c.field).clone()).collect();
        let mut bytes = encode(&[table], schema, IPCMessageProtocol::Stream);
        bytes.truncate(bytes.len() - 16);

        let result = SyncTableReader::<_, Vec64<u8>>::new(
            Cursor::new(bytes),
            1024,
            IPCMessageProtocol::Stream,
            None,
        )
        .read_all_tables();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn rejects_input_beyond_decode_limits_as_the_async_reader_does() {
        let table = Table::new("t", Some(vec![fa_i32!["n", 1, 2, 3, 4]]));
        let schema: Vec<Field> = table.cols.iter().map(|c| (*c.field).clone()).collect();
        let bytes = encode(&[table], schema, IPCMessageProtocol::Stream);
        let limits = DecodeLimits {
            max_n_rows: 3,
            ..DecodeLimits::default()
        };

        let sync_err = SyncTableReader::<_, Vec64<u8>>::new(
            Cursor::new(bytes.clone()),
            1024,
            IPCMessageProtocol::Stream,
            Some(limits),
        )
        .read_all_tables()
        .unwrap_err();

        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let async_err = rt
            .block_on(async {
                let mut reader = TableReader::<Vec64<u8>>::new(
                    Cursor::new(bytes),
                    1024,
                    IPCMessageProtocol::Stream,
                    Some(limits),
                );
                reader.next().await.unwrap()
            })
            .unwrap_err();

        assert_eq!(sync_err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(sync_err.kind(), async_err.kind());
        assert_eq!(sync_err.to_string(), async_err.to_string());
        assert!(sync_err.to_string().contains("record batch rows"));
    }
}
