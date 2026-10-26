// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow IPC File Reader - *Heap-backed version*
//!
//! ## Overview
//! Reads Arrow IPC file format into heap memory. Parses the footer,
//! loads schema, dictionaries, and record batch blocks, and exposes batches as
//! `Table` or a `SuperTable` aggregation.
//!
//! Consistent with the Arrow IPC file specification; expects opening/closing magic,
//! footer length, and block tables.
//!
#[cfg(not(feature = "arena"))]
use minarrow::Vec64;
use minarrow::structs::shared_buffer::SharedBuffer;
/// # Which reader?
/// - **Speed**: Prefer the mmap variant [`MmapTableReader`] when zero-copy performance is required -
///   for e.g., the MMAP version can read millions of rows in microseconds, and very large volumes in milliseconds.
/// - **Flexibility**: this standard reader is more flexible as it is not tied to memory-mapped shared memory.
use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::sync::Arc;

use flatbuffers::Vector;
use minarrow::{Field, SuperTable, Table};

use crate::arrow::file::org::apache::arrow::flatbuf as fbf;
use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
use crate::constants::ARROW_MAGIC_NUMBER;
#[cfg(feature = "arena")]
use crate::models::streams::stream_arena::StreamArena;
use crate::models::decoders::ipc::parser::{
    convert_fb_field_to_arrow, decode_record_batch, handle_dictionary_batch,
};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::readers::ipc::window::window_table;
use crate::models::readers::object_storage::StorageSource;

/// Footer-declared block entry (i.e., offsets/lengths) for a dictionary or record batch.
#[derive(Debug, Clone)]
struct IPCFileBlock {
    /// Absolute byte offset of the block in the file.
    offset: usize,
    /// Length of the FlatBuffers message metadata segment in bytes.
    meta_bytes: usize,
    /// Length of the data body segment in bytes.
    body_bytes: usize,
}

/// Heap-allocated Arrow file reader.
///
/// # Which reader?
/// - **Speed**: Prefer the mmap variant [`MmapTableReader`](crate::models::readers::ipc::mmap_table::MmapTableReader) when zero-copy performance is required -
///   for e.g., the MMAP version can read millions of rows in microseconds, and very large volumes in milliseconds.
/// - **Flexibility**: this standard reader is more flexible as it is not tied to memory-mapped
///   shared memory.
#[derive(Clone)]
pub struct FileTableReader {
    /// Storage handle shared across the reader's lifetime. All block
    /// reads go through positional reads on this handle: `pread`/`seek_read`
    /// for a local file and one ranged request per read for an object.
    /// Opening once amortises across the footer parse, every dictionary
    /// block, and every record batch read.
    source: Arc<StorageSource>,
    /// Arrow schema fields from the file footer
    schema: Vec<Arc<Field>>,
    /// Footer-declared dictionary block table
    dict_blocks: Vec<IPCFileBlock>,
    /// Footer-declared record batch block table
    record_blocks: Vec<IPCFileBlock>,
    /// Loaded dictionaries keyed by dictionary id
    dictionaries: std::collections::HashMap<i64, Vec<String>>,
    /// Single long-lived backing for every block read. Each `read_batch`
    /// reserves a region in the arena and hands the resulting
    /// `SharedBuffer` window to the decoder. While outstanding windows
    /// exist the arena cannot reset; once they drop, `recycle_if_free`
    /// returns the write position to zero without freeing the backing,
    /// so a sequential scan that drops each batch before reading the
    /// next reuses one committed region for the whole file.
    /// Mutex serialises arena writes; reads are independent.
    #[cfg(feature = "arena")]
    block_arena: Arc<std::sync::Mutex<StreamArena>>,
}

impl FileTableReader {
    /// Open an Arrow IPC file into heap memory and parse footer/schema/block tables.
    ///
    /// `path` is a local file path or, with the `object_storage` feature, an
    /// object URL. An object is read with ranged requests for the footer and
    /// for each block as it is read.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let source = StorageSource::open(path)?;
        let file_len = source.len()? as usize;

        if file_len < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for Arrow",
            ));
        }

        // Read only the trailing 10 bytes: footer_len (4) + closing magic (6).
        let mut tail = [0u8; 10];
        source.read_at(&mut tail, (file_len - 10) as u64)?;

        if &tail[4..] != ARROW_MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing closing magic",
            ));
        }

        let footer_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        // The declared footer length is untrusted file data, so the
        // subtraction is checked rather than allowed to wrap.
        let footer_start = (file_len - 10).checked_sub(footer_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "footer length exceeds file size",
            )
        })?;
        if footer_start < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "footer out of bounds",
            ));
        }

        // Read just the footer
        let mut footer_buf = vec![0u8; footer_len];
        source.read_at(&mut footer_buf, footer_start as u64)?;

        let footer_msg = flatbuffers::root::<fbf::Footer>(&footer_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad footer: {e}")))?;

        let fb_schema = footer_msg
            .schema()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "footer missing schema"))?;
        let fb_fields = fb_schema.fields().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "footer schema missing fields")
        })?;
        let mut fields = Vec::with_capacity(fb_fields.len());
        for i in 0..fb_fields.len() {
            let f = convert_fb_field_to_arrow(&fb_fields.get(i))?;
            fields.push(Arc::new(f));
        }

        let dict_blocks = footer_msg
            .dictionaries()
            .unwrap_or_else(|| unsafe { Vector::new(&[], 0) })
            .iter()
            .map(|b| IPCFileBlock {
                offset: b.offset() as usize,
                meta_bytes: b.metaDataLength() as usize,
                body_bytes: b.bodyLength() as usize,
            })
            .collect::<Vec<_>>();

        let record_blocks = footer_msg
            .recordBatches()
            .unwrap_or_else(|| unsafe { Vector::new(&[], 0) })
            .iter()
            .map(|b| IPCFileBlock {
                offset: b.offset() as usize,
                meta_bytes: b.metaDataLength() as usize,
                body_bytes: b.bodyLength() as usize,
            })
            .collect::<Vec<_>>();

        // Footer block entries are untrusted file data. Reject any block
        // whose range falls outside the file before it can drive an
        // out-of-range slice or reservation.
        for blk in dict_blocks.iter().chain(record_blocks.iter()) {
            let end = blk
                .offset
                .checked_add(blk.meta_bytes)
                .and_then(|v| v.checked_add(blk.body_bytes));
            match end {
                Some(end) if blk.offset >= 8 && end <= file_len => {}
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "footer block out of bounds",
                    ));
                }
            }
        }

        // Size the arena to the largest declared block. The recycling in
        // `read_block` means one block is resident at a time on the
        // sequential path, so this bounds the backing to a small
        // allocator-serviced buffer rather than a large reservation.
        #[cfg(feature = "arena")]
        let arena_capacity = dict_blocks
            .iter()
            .chain(record_blocks.iter())
            .map(|b| b.meta_bytes + b.body_bytes)
            .max()
            .unwrap_or(0);

        let mut rdr = Self {
            source: Arc::new(source),
            schema: fields,
            dict_blocks,
            record_blocks,
            dictionaries: std::collections::HashMap::new(),
            #[cfg(feature = "arena")]
            block_arena: Arc::new(std::sync::Mutex::new(StreamArena::with_capacity(
                arena_capacity,
            ))),
        };

        rdr.load_all_dictionaries()?;
        Ok(rdr)
    }

    /// Return the parsed schema fields
    #[inline]
    pub fn schema(&self) -> &[Arc<Field>] {
        &self.schema
    }

    /// Return the number of record batches in the file
    #[inline]
    pub fn num_batches(&self) -> usize {
        self.record_blocks.len()
    }

    /// Read the `idx`th record batch as a `Table`
    pub fn read_batch(&self, idx: usize) -> io::Result<Table> {
        let blk = self
            .record_blocks
            .get(idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "batch idx OOB"))?;
        self.parse_batch_block(blk, None)
    }

    /// Read the `idx`th record batch, materialising only the named columns.
    ///
    /// Column names must match schema field names. Returns an error if any
    /// name is not found. The returned Table contains only the projected
    /// columns, in schema order.
    pub fn read_batch_cols(&self, idx: usize, columns: &[&str]) -> io::Result<Table> {
        let blk = self
            .record_blocks
            .get(idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "batch idx OOB"))?;
        let projection = self.resolve_column_indices(columns)?;
        self.parse_batch_block(blk, Some(&projection))
    }

    /// Read the row window `[row_offset, row_offset + rows)` of record
    /// batch `idx` as a standalone `Table`.
    ///
    /// The block is read from disk once and the window's column buffers
    /// view it zero-copy through its shared backing. String columns
    /// rewrite their small offsets strip against the window base, which
    /// is the only data written. `row_offset` must be a multiple of 512
    /// rows so bit-packed buffers cut on 64-byte boundaries.
    pub fn read_batch_window(
        &self,
        idx: usize,
        row_offset: usize,
        rows: usize,
    ) -> io::Result<Table> {
        let table = self.read_batch(idx)?;
        window_table(&table, row_offset, rows)
    }

    /// Read record batch `idx` as row windows sized towards `target_bytes`
    /// each, returned as a `SuperTable` of standalone (smaller) batch tables.
    ///
    /// See [`Self::batch_windows`] for how the row count is derived and
    /// why a window can exceed the target. A batch that already fits
    /// yields one window sharing the batch's buffers.
    pub fn read_batch_windows(&self, idx: usize, target_bytes: usize) -> io::Result<SuperTable> {
        let mut batches = Vec::new();
        for window in self.batch_windows(idx, target_bytes)? {
            batches.push(Arc::new(window?));
        }
        Ok(SuperTable::from_batches(batches, None))
    }

    /// Iterate record batch `idx` as row windows sized towards
    /// `target_bytes` each. As [`Self::read_batch_windows`] without
    /// collecting, so a streaming consumer holds one window at a time
    /// over the batch's single block buffer.
    ///
    /// The row count derives from the batch's average bytes per row
    /// (`body_bytes / rows`), so `target_bytes` is a target rather than a
    /// hard ceiling. A window exceeds it in two cases:
    ///     1. The window row count is rounded to a 512-row boundary purposely
    ///     so bit-packed and fixed-width buffers cut on 64-byte boundaries,
    ///     with a floor of 512 rows including when a single 512-row block is larger
    ///     than `target_bytes`. The 64-bytes is to uphold SIMD compatiblity for
    ///     any calculations that may need to take place on the Arrow data buffers.
    ///     2. Variable-width columns deviate from the average, so a window denser
    ///     than the batch mean carries more than the average row size implies.
    pub fn batch_windows(
        &self,
        idx: usize,
        target_bytes: usize,
    ) -> io::Result<impl Iterator<Item = io::Result<Table>> + '_> {
        let blk = self
            .record_blocks
            .get(idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "batch idx OOB"))?;
        let body_bytes = blk.body_bytes;
        let table = self.parse_batch_block(blk, None)?;
        let rows = table.n_rows;
        let per_row = (body_bytes / rows.max(1)).max(1);
        let stride = ((target_bytes / per_row) & !511).max(512).min(rows.max(1));
        Ok((0..rows.max(1))
            .step_by(stride)
            .map(move |off| window_table(&table, off, stride.min(rows - off))))
    }

    /// Read every record batch into a `SuperTable` whose batches retain
    /// the file's chunking. Each `Arc<Table>` references the file
    /// reader's owned per-batch buffers; total resident memory is the
    /// sum of every batch's columns.
    ///
    /// For files larger than RAM consider [`MmapTableReader::load_batched`](crate::models::readers::ipc::mmap_table::MmapTableReader::load_batched)
    /// which holds Arc references into the mmap region instead of
    /// owning the underlying bytes.
    pub fn load_batched(&self, name_override: Option<String>) -> io::Result<SuperTable> {
        let mut batches = Vec::with_capacity(self.record_blocks.len());
        for blk in &self.record_blocks {
            batches.push(Arc::new(self.parse_batch_block(blk, None)?));
        }
        Ok(SuperTable::from_batches(batches, name_override))
    }

    /// As [`Self::load_batched`] but materialising only the named
    /// columns from every batch. The resulting `SuperTable` carries
    /// chunks whose `cols` contain just the projection.
    pub fn load_batched_cols(
        &self,
        columns: &[&str],
        name_override: Option<String>,
    ) -> io::Result<SuperTable> {
        let projection = self.resolve_column_indices(columns)?;
        let mut batches = Vec::with_capacity(self.record_blocks.len());
        for blk in &self.record_blocks {
            batches.push(Arc::new(self.parse_batch_block(blk, Some(&projection))?));
        }
        Ok(SuperTable::from_batches(batches, name_override))
    }

    /// Read every record batch and consolidate into a single contiguous
    /// `Table`. Equivalent to `load_batched(None).consolidate()`.
    ///
    /// Consolidation copies every chunk into one contiguous buffer per
    /// column, so this requires that the resulting Table fits in RAM.
    /// For larger-than-memory files use [`Self::load_batched`] or the
    /// per-batch `read_batch[_cols]` family instead.
    pub fn load_table(&self) -> io::Result<Table> {
        use minarrow::Consolidate;
        Ok(self.load_batched(None)?.consolidate())
    }

    /// As [`Self::load_table`] but reading only the named columns. The
    /// consolidated Table contains just the projection.
    pub fn load_table_cols(&self, columns: &[&str]) -> io::Result<Table> {
        use minarrow::Consolidate;
        Ok(self.load_batched_cols(columns, None)?.consolidate())
    }

    /// Read a block from disk and return a `SharedBuffer` over its bytes.
    ///
    /// With the `arena` feature the block is read into the reader's
    /// shared `StreamArena` and a window over the just-written region
    /// is returned. Without the feature a fresh 64-byte aligned `Vec64`
    /// is allocated per call. Both paths use positional I/O on the
    /// reader's shared `Arc<StorageSource>` so concurrent block reads do
    /// not need `&mut self` and the file or object is opened only once per
    /// reader.
    #[cfg(feature = "arena")]
    fn read_block(&self, blk: &IPCFileBlock) -> io::Result<SharedBuffer> {
        let total = blk.meta_bytes + blk.body_bytes;
        let mut arena = self.block_arena.lock().unwrap();
        // Rewind over the previous block once its windows have dropped,
        // so the read lands on already-committed pages instead of
        // faulting fresh ones for every block.
        arena.recycle_if_free();
        arena.ensure_capacity(total);
        let start = arena.write_pos();
        let spare = arena.spare_uninit();
        self.source.read_at_uninit(&mut spare[..total], blk.offset as u64)?;
        // SAFETY: read_at_uninit filled the entire requested region.
        unsafe { arena.advance(total) };
        let shared = arena.window(start, total);
        arena.align();
        Ok(shared)
    }

    #[cfg(not(feature = "arena"))]
    fn read_block(&self, blk: &IPCFileBlock) -> io::Result<SharedBuffer> {
        let total = blk.meta_bytes + blk.body_bytes;
        let mut buf = Vec64::with_capacity(total);
        // SAFETY: `total` equals `buf.capacity()` and `read_at` is the
        // read_exact_at-style wrapper above: it either fills every byte
        // we just exposed via `set_len` or returns Err, in which case
        // `buf` is dropped without anyone observing the uninitialised
        // tail.
        unsafe {
            buf.set_len(total);
        }
        self.source.read_at(&mut buf, blk.offset as u64)?;
        Ok(SharedBuffer::from_vec64(buf))
    }

    /// Parse the IPC frame header from a block buffer, returning the
    /// metadata slice. Validates the continuation marker.
    fn parse_frame_header(buf: &[u8]) -> io::Result<&[u8]> {
        if buf.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "block too short",
            ));
        }
        let cont = u32::from_le_bytes(buf[..4].try_into().unwrap());
        if cont != 0xFFFF_FFFF {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad continuation marker: {cont:#X}"),
            ));
        }
        let meta_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let end = 8 + meta_len;
        if end > buf.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "metadata OOB"));
        }
        Ok(&buf[8..end])
    }

    /// Load and materialise all dictionary batches declared in the footer
    fn load_all_dictionaries(&mut self) -> io::Result<()> {
        let mut new_dicts = std::collections::HashMap::<i64, Vec<String>>::new();
        for blk in &self.dict_blocks {
            let shared = self.read_block(blk)?;
            let meta = Self::parse_frame_header(shared.as_slice())?;
            let fb_msg = flatbuffers::root::<fbm::Message>(meta).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("bad dict msg: {e}"))
            })?;
            let dict_batch = fb_msg.header_as_dictionary_batch().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "expected DictionaryBatch")
            })?;
            let body = &shared.as_slice()[blk.meta_bytes..blk.meta_bytes + blk.body_bytes];
            handle_dictionary_batch(&dict_batch, body, &mut new_dicts, DecodeLimits::default())?;
        }
        self.dictionaries = new_dicts;
        Ok(())
    }

    /// Resolve column names to their schema indices, erroring on unknown names.
    fn resolve_column_indices(&self, columns: &[&str]) -> io::Result<HashSet<usize>> {
        let mut indices = HashSet::with_capacity(columns.len());
        for name in columns {
            let idx = self
                .schema
                .iter()
                .position(|f| f.name == *name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("column '{}' not found in schema", name),
                    )
                })?;
            indices.insert(idx);
        }
        Ok(indices)
    }

    /// Parse a record batch block by reading it from disk on demand.
    /// When `projection` is `Some`, only the specified columns are materialised.
    fn parse_batch_block(
        &self,
        blk: &IPCFileBlock,
        projection: Option<&HashSet<usize>>,
    ) -> io::Result<Table> {
        let shared = self.read_block(blk)?;
        let body_offset = blk.meta_bytes;
        let body_len = blk.body_bytes;
        let fields: Vec<_> = self.schema.iter().map(|a| a.as_ref().clone()).collect();
        let meta = Self::parse_frame_header(shared.as_slice())?;
        let fb_msg = flatbuffers::root::<fbm::Message>(meta).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad record msg: {e}"))
        })?;
        let rec = fb_msg.header_as_record_batch().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "expected RecordBatch header")
        })?;

        // Compressed bodies are handled inside decode_record_batch, which
        // decompresses per buffer and corrects the buffer offsets.
        let (table, _) = decode_record_batch(
            &rec,
            &fields,
            &self.dictionaries,
            shared.clone(),
            body_offset,
            body_len,
            projection,
            DecodeLimits::default(),
        )?;
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use minarrow::{
        Array, Field, FieldArray, NumericArray, Table, TextArray, Vec64, arr_f64, arr_i32,
        arr_str32,
    };
    use tempfile::NamedTempFile;
    use tracing::debug;

    use crate::{
        models::readers::ipc::file_table::FileTableReader,
        models::writers::ipc::table::write_tables_to_file,
        test_helpers::{make_all_types_table, write_test_table_to_file},
    };

    #[tokio::test]
    async fn test_single_batch_roundtrip_heap() {
        let table = make_all_types_table();
        let temp = write_test_table_to_file(&[table.clone()]).await;
        let rdr = FileTableReader::open(&temp.path()).unwrap();
        assert_eq!(rdr.num_batches(), 1);
        let table2 = rdr.read_batch(0).unwrap();

        assert_eq!(table2.n_rows, 4);
        assert_eq!(table2.cols.len(), table.cols.len());

        println!("TABLE {:?}\n", &table2);

        // Int32 col: sum, buffer type
        match &table2.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                let s: i32 = arr.data.as_ref().iter().sum();
                assert_eq!(s, 10);
                // Check if buffer is shared (will be true if data is 64-byte aligned in file)
                // If not aligned, minarrow will clone for safety
                if arr.data.is_shared() {
                    debug!("Int32 buffer is shared (zero-copy)");
                } else {
                    debug!("Int32 buffer was cloned (not 64-byte aligned in file)");
                }
            }
            _ => panic!("wrong type"),
        }
        // Float64 col: value and buffer type
        match &table2.cols[5].array {
            Array::NumericArray(NumericArray::Float64(arr)) => {
                let vals: Vec<_> = arr.data.as_ref().iter().cloned().collect();
                assert_eq!(vals, vec![1.1, 2.2, 3.3, 4.4]);
                // Check if buffer is shared (will be true if data is 64-byte aligned in file)
                // If not aligned, minarrow will clone for safety
                if arr.data.is_shared() {
                    debug!("Float64 buffer is shared (zero-copy)");
                } else {
                    debug!("Float64 buffer was cloned (not 64-byte aligned in file)");
                }
            }
            _ => panic!("wrong type"),
        }
        // Check at least one string, bool, all others present
        let mut seen_string = false;
        let mut seen_bool = false;
        let mut any_shared = false;
        for arr in &table2.cols {
            match &arr.array {
                Array::TextArray(TextArray::String32(a)) => {
                    seen_string = true;
                    if a.data.is_shared() {
                        debug!("String32 data buffer is shared (zero-copy)");
                        any_shared = true;
                    } else {
                        debug!("String32 data buffer was cloned (not 64-byte aligned in file)");
                    }
                }
                Array::BooleanArray(a) => {
                    seen_bool = true;
                    if a.data.bits.is_shared() {
                        debug!("Boolean bits buffer is shared (zero-copy)");
                        any_shared = true;
                    } else {
                        debug!("Boolean bits buffer was cloned (not 64-byte aligned in file)");
                    }
                }
                _ => {}
            }
        }
        assert!(
            seen_string && seen_bool,
            "String32 and Bool must be present"
        );
        debug!("Any buffers shared: {}", any_shared);
        drop(rdr);
        drop(temp);
    }

    #[tokio::test]
    async fn test_shared_buffers_with_aligned_data() {
        // Arrow file structure:
        // 1. Magic "ARROW1\0\0"
        // 2. Schema message (aligned)
        // 3. Record batch message (aligned)
        // 4. Footer
        // 5. Footer length (4 bytes)
        // 6. Magic "ARROW1\0\0"

        // For now, just test that our reader works with the regular file
        // and report on sharing status
        let table = make_all_types_table();
        let tables = vec![table.clone()];
        let temp = write_test_table_to_file(&tables).await;

        let rdr = FileTableReader::open(&temp.path()).unwrap();
        assert_eq!(rdr.num_batches(), 1);
        let table2 = rdr.read_batch(0).unwrap();

        // Count how many buffers are shared vs cloned
        let mut shared_count = 0;
        let mut cloned_count = 0;

        for col in &table2.cols {
            match &col.array {
                Array::NumericArray(na) => match na {
                    NumericArray::Int32(arr) if arr.data.is_shared() => shared_count += 1,
                    NumericArray::Int64(arr) if arr.data.is_shared() => shared_count += 1,
                    NumericArray::UInt32(arr) if arr.data.is_shared() => shared_count += 1,
                    NumericArray::UInt64(arr) if arr.data.is_shared() => shared_count += 1,
                    NumericArray::Float32(arr) if arr.data.is_shared() => shared_count += 1,
                    NumericArray::Float64(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(arr) if arr.data.is_shared() => shared_count += 1,
                    _ => cloned_count += 1,
                },
                Array::BooleanArray(arr) => {
                    if arr.data.bits.is_shared() {
                        shared_count += 1;
                    } else {
                        cloned_count += 1;
                    }
                }
                Array::TextArray(ta) => match ta {
                    TextArray::String32(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "large_string")]
                    TextArray::String64(arr) if arr.data.is_shared() => shared_count += 1,
                    TextArray::Categorical32(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "default_categorical_8")]
                    TextArray::Categorical8(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical16(arr) if arr.data.is_shared() => shared_count += 1,
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical64(arr) if arr.data.is_shared() => shared_count += 1,
                    _ => cloned_count += 1,
                },
                _ => {}
            }
        }

        debug!(
            "Shared buffers: {}, Cloned buffers: {}",
            shared_count, cloned_count
        );
        debug!("Note: Cloning is expected when file data is not 64-byte aligned.");
        debug!("The writer currently doesn't guarantee 64-byte alignment.");

        // We don't assert on specific counts because alignment depends on the writer
        // Just verify the file was read correctly
        assert_eq!(table2.n_rows, 4);
        assert_eq!(table2.cols.len(), table.cols.len());
    }

    #[tokio::test]
    async fn test_read_batch_window_matches_slice_clone() {
        let n: usize = 2048;
        let ids: Vec64<i32> = (0..n as i32).collect();
        let vals: Vec64<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let labels: Vec<String> = (0..n).map(|i| format!("row_{i}")).collect();
        let label_refs: Vec64<&str> = labels.iter().map(String::as_str).collect();
        let table = Table::new(
            "windowed".to_string(),
            Some(vec![
                FieldArray::from_arr("ids", arr_i32!(ids)),
                FieldArray::from_arr("vals", arr_f64!(vals)),
                FieldArray::from_arr("labels", arr_str32!(label_refs)),
            ]),
        );

        let schema: Vec<Field> = table.schema().iter().map(|f| (**f).clone()).collect();
        let temp = NamedTempFile::new().unwrap();
        write_tables_to_file(temp.path().to_str().unwrap(), &[table.clone()], schema)
            .await
            .unwrap();
        let rdr = FileTableReader::open(&temp.path()).unwrap();

        // Windows at 512-row starts, including the ragged tail.
        for (off, len) in [(0usize, 512usize), (512, 512), (1024, 1024), (1536, 512)] {
            let window = rdr.read_batch_window(0, off, len).unwrap();
            let expected = table.slice_clone(off, len);
            assert_eq!(window.n_rows, len);
            for (w, e) in window.cols.iter().zip(expected.cols.iter()) {
                assert_eq!(w.array.to_string(), e.array.to_string(), "col {}", w.field.name);
            }
        }

        // Batch iteration covers every row once, in order.
        let mut rows = 0usize;
        for subbatch in rdr.batch_windows(0, 4096).unwrap() {
            let subbatch = subbatch.unwrap();
            let expected = table.slice_clone(rows, subbatch.n_rows);
            for (w, e) in subbatch.cols.iter().zip(expected.cols.iter()) {
                assert_eq!(w.array.to_string(), e.array.to_string(), "col {}", w.field.name);
            }
            rows += subbatch.n_rows;
        }
        assert_eq!(rows, n);

        // SuperTable variant carries the same total.
        let st = rdr.read_batch_windows(0, 4096).unwrap();
        assert_eq!(st.n_rows(), n);

        // Misaligned window start raises an error.
        assert!(rdr.read_batch_window(0, 100, 100).is_err());
    }
}
