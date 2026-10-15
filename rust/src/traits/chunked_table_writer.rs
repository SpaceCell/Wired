// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked-file writer trait
//!
//! Common shape for the per-format chunked-file writers
//! (`ChunkedCsvWriter`, `ChunkedParquetWriter`, `ChunkedArrowWriter`):
//! a directory of `<base>-NNNNNNNNNN.<ext>` files produced one chunk per
//! call to [`Self::write_chunk`](crate::traits::chunked_table_writer::ChunkedTableWriter::write_chunk). Each emitted file is a complete,
//! independently readable file of the matching format, paired with the
//! matching `ChunkedXReader`.
//!
//! Per-format options are stored on the writer itself, so the trait
//! methods do not carry an `Options` parameter. Constructors stay
//! inherent on each writer, free to take exactly the args they need.
//!
//! `write_all` is the writer-side counterpart to
//! `crate::traits::chunked_table_reader::ChunkedTableReader::read_all`:
//! given a `SuperTable`, it writes every batch as its own chunk file in
//! order. `par_write_all` is the sync `std::thread::scope` fan-out where
//! workers reserve indices off the writer's shared atomic counter and
//! write each batch end-to-end via the writer's `write_chunk_at`. Index
//! reservation goes through the same counter as `write_chunk`, so the
//! two paths compose without producing duplicate filenames.
//!
//! ## Partitioning
//!
//! A writer configured with a [`PartitionType`] places each chunk in a
//! partition directory beneath its root, for example
//! `<dir>/date=2026-10-14/hour=13/<base>-NNNNNNNNNN.<ext>`. The partition
//! path comes from the chunk's time, given through
//! [`ChunkedTableWriter::write_chunk_for_time`]. `write_chunk`, `write_all`
//! and `par_write_all` use the current system time, and the two `_all`
//! methods use one time for every batch in the call. Partition directories
//! are created as they are first written to. Chunk indices come from the
//! writer's single counter whatever the partition.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use minarrow::{SuperTable, Table};
use tokio::runtime::Handle;

use crate::models::writers::chunked::partition::PartitionType;
use crate::models::writers::object_storage::create_dir_all;

/// extension / dir / base / write_chunk / write_chunk_at / par_write_all
/// shape shared by the per-format chunked writers.
///
/// Each implementor stores its own per-format options on the struct
/// (compression for Parquet, encode options for CSV, etc.) and accepts
/// them through its own inherent constructor. The trait covers the
/// shared writing flow only.
///
/// Implementors must hold the chunk counter as an [`AtomicU64`] and
/// expose it through [`Self::counter`] so that both `write_chunk` and
/// `par_write_all` reserve indices off the same atom.
pub trait ChunkedTableWriter: Sized {
    /// Per-format error type surfaced by the writer methods.
    type Error: From<io::Error>;

    /// File extension used for the per-chunk files, without the leading
    /// dot (e.g. `"csv"`, `"parquet"`, `"arrow"`). Static so the
    /// `par_write_all` path agrees with the instance writer on naming.
    fn extension() -> &'static str;

    /// Directory the chunks are being written into.
    fn dir(&self) -> &Path;

    /// Filename stem used for chunk files; the full filename for chunk
    /// `index` is `<base()>-NNNNNNNNNN.<extension()>`.
    fn base(&self) -> &str;

    /// Shared chunk counter. Both `write_chunk` and `par_write_all`
    /// reserve indices off this atom so the two paths compose.
    fn counter(&self) -> &AtomicU64;

    /// Partitioning applied to chunk paths, or `None` for a single flat
    /// directory.
    fn partition(&self) -> Option<&PartitionType>;

    /// Total number of chunks written so far.
    fn batches_written(&self) -> u64 {
        self.counter().load(Ordering::Relaxed)
    }

    /// Build the path for chunk `index` under the writer's directory and
    /// base, matching the layout used by both `write_chunk` and
    /// `par_write_all`.
    fn chunk_path_for(&self, index: u64) -> PathBuf {
        self.dir().join(format!(
            "{}-{:010}.{}",
            self.base(),
            index,
            Self::extension()
        ))
    }

    /// Build the path for chunk `index` of `table` at `time_ns` epoch
    /// nanoseconds, creating its partition directory if needed.
    ///
    /// Without partitioning this is [`Self::chunk_path_for`].
    fn chunk_path_for_time(
        &self,
        table: &Table,
        time_ns: u64,
        index: u64,
    ) -> io::Result<PathBuf> {
        let Some(partition) = self.partition() else {
            return Ok(self.chunk_path_for(index));
        };
        let dir = self.dir().join(partition.partition(table, time_ns));
        create_dir_all(&dir)?;
        Ok(dir.join(format!(
            "{}-{:010}.{}",
            self.base(),
            index,
            Self::extension()
        )))
    }

    /// Encode `table` into a complete chunk file at `path`, using any
    /// per-format options stored on the writer. Shared per-format encode
    /// path used both by `write_chunk` (one chunk at a time) and by
    /// `par_write_all` (one chunk per worker step).
    fn write_chunk_at(&self, path: &Path, table: &Table) -> Result<(), Self::Error>;

    /// Reserve the next chunk index off the shared counter and write
    /// `table` to that chunk file. Returns the path of the written file.
    ///
    /// A partitioned writer takes the partition from the current system
    /// time. See [`Self::write_chunk_for_time`] to supply the time.
    fn write_chunk(&mut self, table: &Table) -> Result<PathBuf, Self::Error> {
        self.write_chunk_for_time(table, current_time_ns())
    }

    /// Write `table` as the next chunk, with `time_ns` epoch nanoseconds as
    /// its time for partitioning. Returns the path of the written file.
    ///
    /// The time is typically the start of the event-time window that
    /// produced `table`, which places replayed or late data in the same
    /// partition as the original. Writers without partitioning ignore it.
    fn write_chunk_for_time(&mut self, table: &Table, time_ns: u64) -> Result<PathBuf, Self::Error> {
        let idx = self.counter().fetch_add(1, Ordering::Relaxed);
        let path = self.chunk_path_for_time(table, time_ns, idx)?;
        self.write_chunk_at(&path, table)?;
        Ok(path)
    }

    /// Write every batch in `supertable` sequentially as its own chunk
    /// file, in order. Returns the paths of the written files. Advances
    /// the shared counter by `supertable.batches.len()`.
    fn write_all(&mut self, supertable: &SuperTable) -> Result<Vec<PathBuf>, Self::Error> {
        let time_ns = current_time_ns();
        let mut paths = Vec::with_capacity(supertable.batches.len());
        for batch in supertable.batches.iter() {
            paths.push(self.write_chunk_for_time(batch.as_ref(), time_ns)?);
        }
        Ok(paths)
    }

    /// Write every batch in `tables` in parallel, one file per batch.
    ///
    /// Reserves a contiguous range of chunk indices off the shared
    /// counter, then fans the encode + file write across worker threads
    /// via [`Self::write_chunk_at`]. Workers pull indices from a local
    /// `AtomicU64` so a single oversized batch doesn't stall a whole
    /// partition. `threads` defaults to
    /// [`std::thread::available_parallelism`] when `None`.
    ///
    /// Returns the written paths in batch order. The shared counter
    /// advances by `tables.len()`, so subsequent `write_chunk` calls
    /// continue past the parallel write without filename collisions.
    ///
    /// Sync parallelism via `std::thread::scope`. Do not call it from
    /// inside a tokio task without wrapping in
    /// `tokio::task::spawn_blocking` - worker threads block on syscalls
    /// and CPU-heavy encode and will compete with the runtime executor
    /// for cores.
    ///
    /// If a worker thread panics, the panic is propagated to the caller.
    fn par_write_all(
        &self,
        tables: &[&Table],
        threads: Option<usize>,
    ) -> Result<Vec<PathBuf>, Self::Error>
    where
        Self: Sync,
        Self::Error: Send,
    {
        if tables.is_empty() {
            return Ok(Vec::new());
        }

        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_threads = threads.unwrap_or(default_threads).max(1).min(tables.len());
        let n_files = tables.len();

        // Reserve a contiguous index range off the writer's shared
        // counter so the indices we hand to workers don't collide with
        // any concurrent `write_chunk` callers or with a later resumed
        // streaming write.
        let start = self.counter().fetch_add(n_files as u64, Ordering::Relaxed);
        let time_ns = current_time_ns();

        let cursor = AtomicU64::new(0);
        // Per-worker capacity hint: ceil(n_files / n_threads). Workers
        // pull a roughly even share via fetch_add, so this is exact for
        // perfectly balanced work and a one-slot overshoot worst-case.
        // Avoids the Vec doubling growth as each worker pushes results.
        let per_worker_cap = n_files.div_ceil(n_threads);
        // Worker threads enter the caller's Tokio runtime, when there is
        // one, which lets object storage paths make their requests on it.
        let runtime = Handle::try_current().ok();
        let collected: Result<Vec<(u64, PathBuf)>, Self::Error> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                let cursor = &cursor;
                let tables = &tables;
                let me = &*self;
                let runtime = &runtime;
                handles.push(
                    s.spawn(move || -> Result<Vec<(u64, PathBuf)>, Self::Error> {
                        let _runtime = runtime.as_ref().map(Handle::enter);
                        let mut local: Vec<(u64, PathBuf)> = Vec::with_capacity(per_worker_cap);
                        loop {
                            let i = cursor.fetch_add(1, Ordering::Relaxed);
                            if i as usize >= n_files {
                                break;
                            }
                            let idx = start + i;
                            let path = me.chunk_path_for_time(tables[i as usize], time_ns, idx)?;
                            me.write_chunk_at(&path, tables[i as usize])?;
                            local.push((i, path));
                        }
                        Ok(local)
                    }),
                );
            }
            let mut all: Vec<(u64, PathBuf)> = Vec::with_capacity(n_files);
            for h in handles {
                // Worker panic re-propagates to the caller.
                let part = h.join().expect("chunked writer worker panicked")?;
                all.extend(part);
            }
            Ok(all)
        });

        let mut all = collected?;
        all.sort_by_key(|(i, _)| *i);
        Ok(all.into_iter().map(|(_, p)| p).collect())
    }
}

/// Current system time in epoch nanoseconds.
fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64)
}
