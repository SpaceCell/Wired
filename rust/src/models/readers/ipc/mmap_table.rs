// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Memory-mapped Arrow IPC file reader.
//!
//! # Overview
//! Zero-copy reader for Arrow IPC files. Parses the footer/schema, loads dictionaries, and
//! exposes batches as `Table` or aggregates as `SuperTable`.
//!
//! # Zero-Copy
//! Buffers are read directly when 64-byte aligned (as produced by this
//! crate’s writers); otherwise they are copied via SIMD-friendly allocations.
//! This ensures data is in the optimal format for downstream calculations upfront,
//! though is a notable limitation or those who aren't planning for that use case
//! at the present time.
//!
//! # Platform:
//! Uses POSIX `mmap(2)`, supported on Unix-like systems only.
//!
//! # Reader spec
//! Follows the Arrow IPC specification
//! <https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format>.

use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use flatbuffers::Vector;
use minarrow::{Field, SuperTable, Table};

use crate::arrow::file::org::apache::arrow::flatbuf as fbf;
use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
use crate::constants::ARROW_MAGIC_NUMBER;
use crate::debug_println;
use minarrow::structs::shared_buffer::SharedBuffer;

use crate::models::decoders::ipc::parser::{
    convert_fb_field_to_arrow, decode_record_batch, handle_dictionary_batch,
};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::mmap::MemMap;
use crate::models::readers::ipc::window::window_table;

/// Footer-declared block entry offsets/lengths for a dictionary or record batch.
#[derive(Debug, Clone)]
struct IPCFileBlock {
    /// Absolute byte offset of the block in the file
    offset: usize,
    /// Length of the FlatBuffers message metadata segment in bytes
    meta_bytes: usize,
    /// Length of the data body segment in bytes
    body_bytes: usize,
}

/// Keeps file handle and mmap region alive together; dereferences to file bytes.
struct MmapBytes {
    /// File handle - kept alive for the lifetime of the mapping
    _file: File,
    /// Memory-mapped region - 64-byte aligned mapping wrapper
    mmap: Arc<MemMap<64>>,
}

impl std::ops::Deref for MmapBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.mmap
    }
}

impl AsRef<[u8]> for MmapBytes {
    fn as_ref(&self) -> &[u8] {
        &self.mmap
    }
}

/// Wrapper enabling `Arc<MmapBytes>` to be used with `SharedBuffer::from_owner`.
///
/// `SharedBuffer::from_owner` requires `AsRef<[u8]> + Send + Sync + 'static`.
/// `Arc<T>` only implements `AsRef<T>`, not `AsRef<[u8]>`, so we delegate
/// through this wrapper which keeps the mmap alive via the inner Arc.
struct MmapRegionOwner(Arc<MmapBytes>);

impl AsRef<[u8]> for MmapRegionOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

// Safety: the inner File is held only to keep the mmap alive. No data is
// read through it after construction, so sharing across threads is safe.
unsafe impl Send for MmapRegionOwner {}
unsafe impl Sync for MmapRegionOwner {}

/// Zero-copy Arrow IPC file reader backed by a memory map
///
/// It uses a custom `mmap` implementation to avoid bloating dependencies.
///
/// ## Zero-Copy behaviour
/// - Currently zero-copy for 64-byte aligned writers, otherwise it copied into SIMD-friendly buffers.
/// - The current method to guarantee zero-copy is to use the `TableWriter` from this crate, and the resulting
///   file can be zero-copy read.
/// - `.arrow` files written with other implementations e.g., `pyarrow`, `arrow-rs` are usually 8-byte aligned,
///   and thus will copy at the current time. Though, this means their buffers are often not 64-byte aligned and
///   thus require re-allocations before processing with SIMD-kernels and related scenarios.
/// - Hence, this library has initially prioritised this high-performance scenario, though in future we may
///   add support for the general case, and invite community contributions.
///
/// ## Platform
/// -  This implementation uses POSIX `mmap(2)` for zero-copy access, and is therefore
///    supported only on Unix-like operating systems (Linux, macOS, BSDs, Solaris, etc.).
/// - There are no plans to support Windows, however PR's will be accepted.
///
/// ## Overview
/// - Parses footer/schema, loads dictionaries, and exposes batches as `Table` or composites as `SuperTable`.
#[derive(Clone)]
pub struct MmapTableReader {
    /// Backing mmap region and owning file, shared across clones
    region: Arc<MmapBytes>,
    /// Arrow schema fields from the file footer
    schema: Vec<Arc<Field>>,
    /// Footer-declared dictionary block table
    dict_blocks: Vec<IPCFileBlock>,
    /// Footer-declared record batch block table
    record_blocks: Vec<IPCFileBlock>,
    /// Loaded dictionaries keyed by dictionary id
    dictionaries: std::collections::HashMap<i64, Vec<String>>,
    /// Offset from original file start to the chosen 64-byte aligned data start
    #[allow(dead_code)]
    aligned_offset: usize,
}

impl MmapTableReader {
    /// Open and mmap an Arrow IPC file
    ///
    /// Parses footer/schema and block tables.
    ///
    /// The mapping reads through the OS page cache, so the file should
    /// fit in RAM. For datasets larger than RAM use
    /// [`FileTableReader`](crate::models::readers::ipc::file_table::FileTableReader),
    /// whose buffered reads leave the page cache cheaply reclaimable
    /// instead of stalling page faults in reclaim when memory fills.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(&path)?;
        let meta = file.metadata()?;
        let file_len = meta.len() as usize;

        debug_println!("MMAP File len: {}", file_len);

        // Opening magic (6) + footer length (4) + closing magic (6) is the
        // smallest well-formed file, so anything shorter is malformed.
        if file_len < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for Arrow",
            ));
        }

        let path_str = path.as_ref().to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "path is not valid UTF-8",
            )
        })?;

        // MMAP entire file and find 64-byte aligned data region
        let mmap = Arc::new(MemMap::<64>::open(path_str, 0, file_len)?);
        let region = Arc::new(MmapBytes { _file: file, mmap });

        let data = region.as_ref();

        // Find the first 64-byte aligned offset after the 6-byte Arrow magic
        let magic_end = 6;
        let base_ptr = data.as_ptr() as usize;
        let aligned_data_offset = {
            let desired_ptr = base_ptr + magic_end;
            let aligned_ptr = (desired_ptr + 63) & !63; // Round up to next 64-byte boundary
            aligned_ptr - base_ptr
        };

        debug_println!(
            "Base ptr: 0x{:x}, Magic end: {}, Aligned data offset: {}, Is 64-byte aligned: {}",
            base_ptr,
            magic_end,
            aligned_data_offset,
            (base_ptr + aligned_data_offset).is_multiple_of(64)
        );

        if &data[..6] != ARROW_MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing opening magic",
            ));
        }
        if &data[file_len - 6..] != ARROW_MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing closing magic",
            ));
        }

        let footer_len_offset = file_len - 6 - 4;
        let footer_len = u32::from_le_bytes(
            data[footer_len_offset..footer_len_offset + 4]
                .try_into()
                .unwrap(),
        ) as usize;

        // The declared footer length is untrusted file data, so the
        // subtraction is checked rather than allowed to wrap.
        let footer_start = footer_len_offset.checked_sub(footer_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "footer length exceeds file size",
            )
        })?;
        let footer_end = footer_start + footer_len;
        if footer_start < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "footer out of bounds",
            ));
        }

        let footer_msg: &fbf::Footer = {
            &flatbuffers::root::<fbf::Footer>(&data[footer_start..footer_end]).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("bad footer: {e}"))
            })?
        };

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

        let mut rdr = Self {
            region,
            schema: fields,
            dict_blocks,
            record_blocks,
            dictionaries: std::collections::HashMap::new(),
            aligned_offset: aligned_data_offset,
        };

        rdr.load_all_dictionaries()?;
        Ok(rdr)
    }

    /// Return the parsed schema fields.
    #[inline]
    pub fn schema(&self) -> &[Arc<Field>] {
        &self.schema
    }

    /// Return the number of record batches in the file.
    #[inline]
    pub fn num_batches(&self) -> usize {
        self.record_blocks.len()
    }

    /// Read the `idx`th record batch as a `Table`.
    pub fn read_batch(&self, idx: usize) -> io::Result<Table> {
        self.parse_batch_block(idx, None)
    }

    /// Read the `idx`th record batch, materialising only the named columns.
    ///
    /// Column names must match schema field names. Returns an error if any
    /// name is not found. The returned Table contains only the projected
    /// columns, in schema order.
    ///
    /// For the mmap reader, this means pages backing skipped columns are
    /// never faulted in - a significant win for wide tables on cold reads.
    pub fn read_batch_cols(&self, idx: usize, columns: &[&str]) -> io::Result<Table> {
        let projection = self.resolve_column_indices(columns)?;
        self.parse_batch_block(idx, Some(&projection))
    }

    /// Read the row window `[row_offset, row_offset + rows)` of record
    /// batch `idx` as a standalone `Table`.
    ///
    /// Column buffers window zero-copy through the map's shared owner,
    /// so only the pages the window touches fault in. String columns
    /// rewrite their small offsets strip against the window base, which
    /// is the only data written. `row_offset` must be a multiple of 512
    /// rows so bit-packed buffers cut on 64-byte boundaries. Whole-batch
    /// and whole-file reads are unchanged: `read_batch` and the load
    /// methods still fault in their full range.
    pub fn read_batch_window(
        &self,
        idx: usize,
        row_offset: usize,
        rows: usize,
    ) -> io::Result<Table> {
        let table = self.parse_batch_block(idx, None)?;
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
    /// collecting, so a streaming consumer holds one window at a time and
    /// the map's pages fault in window by window.
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
        let table = self.parse_batch_block(idx, None)?;
        let rows = table.n_rows;
        let per_row = (body_bytes / rows.max(1)).max(1);
        let stride = ((target_bytes / per_row) & !511).max(512).min(rows.max(1));
        Ok((0..rows.max(1))
            .step_by(stride)
            .map(move |off| window_table(&table, off, stride.min(rows - off))))
    }

    /// Read every record batch into a `SuperTable` whose batches are
    /// zero-copy views over the mmap region. Each `Arc<Table>` holds
    /// `SharedBuffer` references into the mapping; total resident
    /// memory is bounded by the OS page cache, not by the file size,
    /// so a multi-GiB file can be wrapped on a smaller-RAM host.
    pub fn load_batched(&self, name_override: Option<String>) -> io::Result<SuperTable> {
        let mut batches = Vec::with_capacity(self.record_blocks.len());
        for idx in 0..self.record_blocks.len() {
            batches.push(Arc::new(self.parse_batch_block(idx, None)?));
        }
        Ok(SuperTable::from_batches(batches, name_override))
    }

    /// As [`Self::load_batched`] but materialising only the named
    /// columns from every batch. The mmap pages backing non-projected
    /// columns are never faulted in for any batch.
    pub fn load_batched_cols(
        &self,
        columns: &[&str],
        name_override: Option<String>,
    ) -> io::Result<SuperTable> {
        let projection = self.resolve_column_indices(columns)?;
        let mut batches = Vec::with_capacity(self.record_blocks.len());
        for idx in 0..self.record_blocks.len() {
            batches.push(Arc::new(self.parse_batch_block(idx, Some(&projection))?));
        }
        Ok(SuperTable::from_batches(batches, name_override))
    }

    /// Read every record batch and consolidate into a single contiguous
    /// `Table`. Equivalent to `load_batched(None).consolidate()`.
    ///
    /// Consolidation copies every batch's columns into one contiguous
    /// buffer per column, so this requires the resulting Table fits in
    /// RAM. For larger-than-memory files use [`Self::load_batched`] or
    /// the per-batch `read_batch[_cols]` family instead.
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

    /// Load and materialise all dictionary batches declared in the footer.
    fn load_all_dictionaries(&mut self) -> io::Result<()> {
        let mut new_dicts = std::collections::HashMap::<i64, Vec<String>>::new();
        let data = self.region.as_ref();

        for blk in &self.dict_blocks {
            let msg = self.slice_message(data, blk)?;
            let fb_msg: &fbm::Message = &flatbuffers::root::<fbm::Message>(msg).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("bad dict msg: {e}"))
            })?;

            let dict_batch = fb_msg.header_as_dictionary_batch().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "expected DictionaryBatch")
            })?;

            let body =
                &data[blk.offset + blk.meta_bytes..blk.offset + blk.meta_bytes + blk.body_bytes];

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

    /// Parse the `idx`th record batch block into a `Table` - zero-copy over
    /// the mmap region. When `projection` is `Some`, only the specified
    /// columns are materialised, so mmap pages backing skipped columns are
    /// never faulted in.
    fn parse_batch_block(
        &self,
        idx: usize,
        projection: Option<&HashSet<usize>>,
    ) -> io::Result<Table> {
        let blk = self
            .record_blocks
            .get(idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "batch idx OOB"))?;
        let data = self.region.as_ref();
        let meta_slice = self.slice_message(data, blk)?;

        // The body begins at the block offset plus the footer-declared
        // metadata length, which includes the length prefix and padding per
        // the Arrow file format. Files written by this crate pad the
        // metadata so the body lands on a 64-byte boundary and buffers map
        // zero-copy; other writers pad to 8 bytes and their buffers copy
        // into aligned allocations during decode.
        let body_offset = blk.offset + blk.meta_bytes;
        let body_len = blk.body_bytes;

        debug_println!("Body offset: {}, body len: {}", body_offset, body_len);

        let fb_msg: &fbm::Message =
            &flatbuffers::root::<fbm::Message>(meta_slice).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("bad record msg: {e}"))
            })?;

        let rec = fb_msg.header_as_record_batch().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "expected RecordBatch header")
        })?;

        // Wrap the mmap region as a SharedBuffer for zero-copy column decoding
        let shared = SharedBuffer::from_owner(MmapRegionOwner(self.region.clone()));
        let fields: Vec<_> = self.schema.iter().map(|a| a.as_ref().clone()).collect();

        let (table, _shared) = decode_record_batch(
            &rec,
            &fields,
            &self.dictionaries,
            shared,
            body_offset,
            body_len,
            projection,
            DecodeLimits::default(),
        )?;
        Ok(table)
    }

    /// Slice and validate the FlatBuffers message at the given block - checks continuation + size.
    fn slice_message<'a>(&self, data: &'a [u8], blk: &IPCFileBlock) -> io::Result<&'a [u8]> {
        if blk.offset + 8 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "block header OOB",
            ));
        }
        // Continuation marker
        let cont = u32::from_le_bytes(data[blk.offset..blk.offset + 4].try_into().unwrap());
        if cont != 0xFFFF_FFFF {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad continuation marker: {cont:#X}"),
            ));
        }
        // Metadata length
        let meta_len =
            u32::from_le_bytes(data[blk.offset + 4..blk.offset + 8].try_into().unwrap()) as usize;
        let start = blk.offset + 8;
        let end = start + meta_len;
        if end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "msg slice OOB",
            ));
        }
        Ok(&data[start..end])
    }
}

#[cfg(test)]
mod tests {

    use tracing::debug;

    use tempfile::NamedTempFile;

    use crate::{
        models::readers::ipc::mmap_table::MmapTableReader,
        models::writers::ipc::table::write_tables_to_file,
        test_helpers::{make_all_types_table, write_test_table_to_file},
    };
    use minarrow::{
        Array, Field, FieldArray, NumericArray, Table, TextArray, Vec64, arr_f64, arr_i32,
        arr_str32,
    };

    // -------------------- Tests -------------------- //

    #[tokio::test]
    async fn test_single_batch_roundtrip_mmap() {
        let table = make_all_types_table();
        let temp = write_test_table_to_file(&[table.clone()]).await;
        let rdr = MmapTableReader::open(&temp.path()).unwrap();
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
                // Note: Currently mmap requires copying data to create Arc<[u8]>
                // so buffers won't be shared. True zero-copy would require
                // modifying minarrow to accept mmap memory directly.
                if arr.data.is_shared() {
                    debug!("Int32 buffer is shared (64-byte aligned in copied Arc)");
                } else {
                    debug!("Int32 buffer was cloned (not 64-byte aligned in copied Arc)");
                }
            }
            _ => panic!("wrong type"),
        }
        // Float64 col: value and buffer type
        match &table2.cols[5].array {
            Array::NumericArray(NumericArray::Float64(arr)) => {
                let vals: Vec<_> = arr.data.as_ref().iter().cloned().collect();
                assert_eq!(vals, vec![1.1, 2.2, 3.3, 4.4]);
                // Note: Currently mmap requires copying data to create Arc<[u8]>
                // so buffers won't be shared. True zero-copy would require
                // modifying minarrow to accept mmap memory directly.
                if arr.data.is_shared() {
                    debug!("Float64 buffer is shared (64-byte aligned in copied Arc)");
                } else {
                    debug!("Float64 buffer was cloned (not 64-byte aligned in copied Arc)");
                }
            }
            _ => panic!("wrong type"),
        }
        // // Dictionary col: unique values and zero-copy buffer
        // match &table2.cols[8].array {
        //     Array::TextArray(TextArray::Categorical32(arr)) => {
        //         assert_eq!(&arr.unique_values[..], &["apple", "banana", "pear"]);
        //         assert!(arr.data.is_shared());
        //     }
        //     _ => panic!("wrong type")
        // }
        // Check at least one string, bool, all others present
        let mut seen_string = false;
        let mut seen_bool = false;
        let mut any_shared = false;
        for arr in &table2.cols {
            match &arr.array {
                Array::TextArray(TextArray::String32(a)) => {
                    seen_string = true;
                    if a.data.is_shared() {
                        debug!("String32 data buffer is shared");
                        any_shared = true;
                    } else {
                        debug!("String32 data buffer was cloned");
                    }
                }
                Array::BooleanArray(a) => {
                    seen_bool = true;
                    if a.data.bits.is_shared() {
                        debug!("Boolean bits buffer is shared");
                        any_shared = true;
                    } else {
                        debug!("Boolean bits buffer was cloned");
                    }
                }
                _ => {}
            }
        }
        assert!(
            seen_string && seen_bool,
            "String32 and Bool must be present"
        );
        debug!("Any buffers shared in mmap: {}", any_shared);
        drop(rdr);
        drop(temp);
    }

    #[tokio::test]
    async fn test_read_batch_and_sharedness() {
        let table = make_all_types_table();
        let temp = write_test_table_to_file(&[table.clone()]).await;
        let rdr = MmapTableReader::open(&temp.path()).unwrap();

        let t2 = rdr.read_batch(0).unwrap();
        // Note: Currently mmap requires copying data to create Arc<[u8]>
        // so we check for shared OR owned buffers. True zero-copy would require
        // modifying minarrow to accept mmap memory directly.
        let mut shared_count = 0;
        let mut owned_count = 0;
        for arr in t2.cols.iter().map(|fa| &fa.array) {
            match arr {
                Array::NumericArray(NumericArray::Int32(a)) => {
                    if a.data.is_shared() {
                        shared_count += 1;
                    } else {
                        owned_count += 1;
                    }
                }
                Array::NumericArray(NumericArray::Float64(a)) => {
                    if a.data.is_shared() {
                        shared_count += 1;
                    } else {
                        owned_count += 1;
                    }
                }
                Array::TextArray(TextArray::String32(a)) => {
                    if a.data.is_shared() {
                        shared_count += 1;
                    } else {
                        owned_count += 1;
                    }
                }
                Array::BooleanArray(a) => {
                    if a.data.bits.is_shared() {
                        shared_count += 1;
                    } else {
                        owned_count += 1;
                    }
                }
                Array::TextArray(TextArray::Categorical32(a)) => {
                    if a.data.is_shared() {
                        shared_count += 1;
                    } else {
                        owned_count += 1;
                    }
                }
                #[cfg(feature = "default_categorical_8")]
                Array::TextArray(TextArray::Categorical8(a)) => {
                    if a.data.is_shared() {
                        shared_count += 1;
                    } else {
                        owned_count += 1;
                    }
                }
                _ => {}
            }
        }
        debug!(
            "Mmap read_batch: {} shared, {} owned buffers",
            shared_count, owned_count
        );
        drop(temp)
    }

    #[tokio::test]
    async fn test_multiple_batches_and_supertable() {
        let t1 = make_all_types_table();
        let t2 = make_all_types_table();
        let temp = write_test_table_to_file(&[t1.clone(), t2.clone()]).await;

        let rdr = MmapTableReader::open(temp.path()).unwrap();
        assert_eq!(rdr.num_batches(), 2);

        let supertbl = rdr
            .load_batched(Some("my_supertable".to_string()))
            .unwrap();
        assert_eq!(supertbl.n_rows, 8);
        assert_eq!(supertbl.batches.len(), 2);

        // Check one batch/col
        assert_eq!(supertbl.batches[0].cols[0].field.name, "int32");
        match &supertbl.batches[1].cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                let values: Vec<i32> = arr.data.as_ref().iter().copied().collect();
                assert_eq!(values, vec![1, 2, 3, 4]);
            }
            _ => panic!("expected int32 col"),
        }
    }

    #[tokio::test]
    async fn test_big_super_table_iteration_and_owned_conversion() {
        let tables: Vec<Table> = (0..10).map(|_| make_all_types_table()).collect();
        let temp = write_test_table_to_file(&tables).await;
        let rdr = MmapTableReader::open(temp.path()).unwrap();
        let supertbl = rdr.load_batched(None).unwrap();
        assert_eq!(supertbl.batches.len(), 10);

        for batch in &supertbl.batches {
            for col in &batch.cols {
                match &col.array {
                    Array::NumericArray(NumericArray::Int32(arr)) => {
                        // Note: May be owned or shared depending on alignment
                        if arr.data.is_shared() {
                            debug!("Int32 is shared");
                        }
                        let owned = arr.data.to_owned_copy();
                        assert!(!owned.is_shared());
                    }
                    Array::NumericArray(NumericArray::Float64(arr)) => {
                        // Note: May be owned or shared depending on alignment
                        if arr.data.is_shared() {
                            debug!("Float64 is shared");
                        }
                        let owned = arr.data.to_owned_copy();
                        assert!(!owned.is_shared());
                    }
                    Array::TextArray(TextArray::String32(arr)) => {
                        // Note: May be owned or shared depending on alignment
                        if arr.data.is_shared() {
                            debug!("String32 is shared");
                        }
                        let owned = arr.data.to_owned_copy();
                        assert!(!owned.is_shared());
                    }
                    Array::BooleanArray(arr) => {
                        // Note: May be owned or shared depending on alignment
                        if arr.data.bits.is_shared() {
                            debug!("Boolean is shared");
                        }
                        let owned = arr.data.to_owned_copy();
                        assert!(!owned.bits.is_shared());
                    }
                    Array::TextArray(TextArray::Categorical32(arr)) => {
                        if arr.data.is_shared() {
                            debug!("Categorical32 is shared");
                        }
                        let owned = arr.data.to_owned_copy();
                        assert!(!owned.is_shared());
                    }
                    #[cfg(feature = "default_categorical_8")]
                    Array::TextArray(TextArray::Categorical8(arr)) => {
                        if arr.data.is_shared() {
                            debug!("Categorical8 is shared");
                        }
                        let owned = arr.data.to_owned_copy();
                        assert!(!owned.is_shared());
                    }
                    _ => {}
                }
            }
        }
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
        let rdr = MmapTableReader::open(&temp.path()).unwrap();

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

    #[tokio::test]
    async fn test_error_on_invalid_batch_index() {
        let table = make_all_types_table();
        let temp = write_test_table_to_file(&[table]).await;
        let rdr = MmapTableReader::open(temp.path()).unwrap();
        let err = rdr.read_batch(1000).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    /// Malformed inputs must come back as errors, never panics: too-short
    /// files, a footer length larger than the file, and a corrupt footer.
    #[test]
    fn malformed_files_error_instead_of_panicking() {
        use std::os::unix::fs::FileExt;

        // Too short for magic + footer length + magic.
        let temp = NamedTempFile::new().unwrap();
        temp.as_file().write_all_at(b"ARROW1", 0).unwrap();
        assert!(MmapTableReader::open(temp.path()).is_err());

        // Valid magics but a footer length that exceeds the file.
        let temp = NamedTempFile::new().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"ARROW1\0\0");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(b"ARROW1");
        temp.as_file().write_all_at(&bytes, 0).unwrap();
        assert!(MmapTableReader::open(temp.path()).is_err());

        // Well-sized but garbage footer bytes.
        let temp = NamedTempFile::new().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"ARROW1\0\0");
        bytes.extend_from_slice(&[0xFFu8; 32]);
        bytes.extend_from_slice(&32u32.to_le_bytes());
        bytes.extend_from_slice(b"ARROW1");
        temp.as_file().write_all_at(&bytes, 0).unwrap();
        assert!(MmapTableReader::open(temp.path()).is_err());
    }
}
