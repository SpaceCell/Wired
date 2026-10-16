// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked-file reader trait
//!
//! Common shape for the per-format chunked-file readers
//! (`ChunkedCsvReader`, `ChunkedParquetReader`, `ChunkedArrowReader`):
//! a directory of `<base>-NNNNNNNNNN.<ext>` files emitted by the matching
//! chunked writer, presented as an ordered iterator of `Table`s plus a
//! `load_batched` that materialises every chunk into a `SuperTable`,
//! and a `load_table` that further consolidates the chunks into a
//! single `Table`. The `*_cols` variants project to the named columns.
//!
//! `load_batched` deliberately does not consolidate; the per-chunk batches
//! are preserved in the returned `SuperTable`. Callers wanting a single
//! `Table` use `load_table` (which itself calls
//! `SuperTable::consolidate`) or invoke `.consolidate()` on the returned
//! `SuperTable` themselves.
//!
//! `par_load_batched` is the parallel counterpart: a sync
//! `std::thread::scope` fan-out where each worker does open + decode
//! end-to-end on chunks pulled from a shared atomic counter. The trait
//! provides it as a default method built on the per-format `list_paths`
//! and `read_chunk`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use minarrow::{Consolidate, SuperTable, Table};
use tokio::runtime::Handle;

/// Open / load_batched / iterate / par_load_batched shape shared by the
/// per-format chunked readers.
///
/// Implementors yield one `Table` per chunk file in ascending index order.
/// Per-format extras (CSV decode options, batch size, etc.) flow through
/// the associated `Options` type; formats with no extras use `()`.
pub trait ChunkedTableReader: Iterator<Item = Result<Table, Self::Error>> + Sized {
    /// Per-format error type surfaced by the constructor and the iterator.
    type Error;
    /// Per-format constructor options. Use `()` if no extras are needed.
    type Options;

    /// Open every `<base>-*.<ext>` file inside `dir`, sorted by numeric
    /// suffix so chunks come out in write order. Files whose names don't
    /// match the pattern are ignored.
    fn open<P: AsRef<Path>>(
        dir: P,
        base: &str,
        options: Self::Options,
    ) -> Result<Self, Self::Error>;

    /// List chunk file paths in `dir` matching this format's naming
    /// pattern, sorted by ascending numeric index. Used by `open` to
    /// populate the reader's path list.
    fn list_paths<P: AsRef<Path>>(dir: P, base: &str) -> Result<Vec<PathBuf>, Self::Error>;

    /// Return the chunk paths held by this reader, in write order.
    /// `par_load_batched` and `par_load_batched_cols` consume this to
    /// dispatch parallel worker decode without re-listing the
    /// directory.
    fn paths(&self) -> &[PathBuf];

    /// Decode one chunk file into a `Table`. The same per-format decode
    /// path used by the iterator (one chunk at a time) and by
    /// `par_load_batched` (one chunk per worker step). Format-specific
    /// configuration (CSV decode options, etc.) lives on `self`.
    fn read_chunk(&self, path: &Path) -> Result<Table, Self::Error>;

    /// Decode one chunk file into a `Table` containing only the named
    /// columns. Implementors must push the projection into the format's
    /// own decoder so non-selected columns are not read from disk -
    /// post-decode filtering would defeat the point of the API.
    /// For Arrow IPC chunks that means
    /// [`FileTableReader::read_batch_cols`](crate::models::readers::ipc::file_table::FileTableReader::read_batch_cols); for Parquet it means the
    /// `parquet::load_parquet_table_cols` path; for CSV it
    /// means configuring the decoder's projection up front.
    fn read_chunk_cols(&self, path: &Path, columns: &[&str]) -> Result<Table, Self::Error>;

    /// Drain the iterator into a `SuperTable` whose `batches` field holds
    /// the per-chunk Tables in order. Does not consolidate; consumers call
    /// `.consolidate()` themselves if they want a single Table.
    fn load_batched(self) -> Result<SuperTable, Self::Error> {
        let mut batches: Vec<Arc<Table>> = Vec::new();
        let mut name: Option<String> = None;
        for chunk in self {
            let chunk = chunk?;
            if name.is_none() {
                name = Some(chunk.name.clone());
            }
            batches.push(Arc::new(chunk));
        }
        Ok(SuperTable::from_batches(
            batches,
            name.or(Some("chunked".into())),
        ))
    }

    /// As [`Self::load_batched`] but materialising only the named
    /// columns from every chunk via [`Self::read_chunk_cols`]. Each
    /// per-format impl provides this so projection is pushed into the
    /// chunk's decoder; non-selected columns are not read from disk.
    fn load_batched_cols(self, columns: &[&str]) -> Result<SuperTable, Self::Error>;

    /// Drain and consolidate every chunk into a single contiguous
    /// `Table`. Equivalent to `load_batched().consolidate()`.
    ///
    /// Consolidation copies each chunk's columns into one contiguous
    /// buffer per column, so this requires the resulting Table fits in
    /// RAM. For larger-than-memory directories use
    /// [`Self::load_batched`] and iterate the SuperTable's batches.
    fn load_table(self) -> Result<Table, Self::Error> {
        Ok(self.load_batched()?.consolidate())
    }

    /// As [`Self::load_table`] but materialising only the named columns.
    /// Equivalent to `load_batched_cols(columns).consolidate()`.
    fn load_table_cols(self, columns: &[&str]) -> Result<Table, Self::Error> {
        Ok(self.load_batched_cols(columns)?.consolidate())
    }

    /// Read every chunk file in `dir` in parallel, returning a SuperTable
    /// whose batches are in write order.
    ///
    /// Each worker thread does open + decode end-to-end via
    /// [`Self::read_chunk`]. Workers pull paths from a shared atomic
    /// counter so a single oversized chunk doesn't stall a whole
    /// partition. `threads` defaults to
    /// [`std::thread::available_parallelism`] when `None`; pass `Some(n)`
    /// to override (e.g. for HDD storage where high concurrency causes
    /// seek thrashing, or to dedicate fewer cores when running alongside
    /// other CPU-heavy work).
    ///
    /// This is sync parallelism using `std::thread::scope`. Do not call
    /// it from inside a tokio task without wrapping in
    /// `tokio::task::spawn_blocking` - worker threads block on syscalls
    /// and CPU-heavy decode and will compete with the runtime executor
    /// for cores.
    ///
    /// If a worker thread panics, the panic is propagated to the caller.
    fn par_load_batched<P: AsRef<Path>>(
        dir: P,
        base: &str,
        options: Self::Options,
        threads: Option<usize>,
    ) -> Result<SuperTable, Self::Error>
    where
        Self: Sync,
        Self::Error: Send,
    {
        let reader = Self::open(dir, base, options)?;
        let paths = reader.paths();
        if paths.is_empty() {
            return Ok(SuperTable::from_batches(Vec::new(), Some("chunked".into())));
        }

        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_threads = threads.unwrap_or(default_threads).max(1).min(paths.len());
        let n_files = paths.len();
        let next_index = AtomicUsize::new(0);

        // Per-worker capacity hint: ceil(n_files / n_threads). Workers
        // pull a roughly even share via fetch_add, so this is exact for
        // perfectly balanced work and a one-slot overshoot worst-case.
        // Avoids the Vec doubling growth as each worker pushes results.
        let per_worker_cap = n_files.div_ceil(n_threads);
        // Worker threads enter the caller's Tokio runtime, when there is
        // one, which lets object storage paths make their requests on it.
        let runtime = Handle::try_current().ok();
        let collected: Result<Vec<(usize, Table)>, Self::Error> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                let next_index = &next_index;
                let this = &reader;
                let runtime = &runtime;
                handles.push(
                    s.spawn(move || -> Result<Vec<(usize, Table)>, Self::Error> {
                        let _runtime = runtime.as_ref().map(Handle::enter);
                        let mut local: Vec<(usize, Table)> = Vec::with_capacity(per_worker_cap);
                        let paths = this.paths();
                        loop {
                            let idx = next_index.fetch_add(1, Ordering::Relaxed);
                            if idx >= n_files {
                                break;
                            }
                            let table = this.read_chunk(&paths[idx])?;
                            local.push((idx, table));
                        }
                        Ok(local)
                    }),
                );
            }
            let mut all: Vec<(usize, Table)> = Vec::with_capacity(n_files);
            for h in handles {
                // Worker panic re-propagates to the caller.
                let part = h.join().expect("chunked reader worker panicked")?;
                all.extend(part);
            }
            Ok(all)
        });

        let mut all = collected?;
        all.sort_by_key(|(i, _)| *i);
        let batches: Vec<Arc<Table>> = all.into_iter().map(|(_, t)| Arc::new(t)).collect();
        Ok(SuperTable::from_batches(batches, Some("chunked".into())))
    }

    /// Parallel counterpart of [`Self::load_batched_cols`]. Each worker
    /// decodes one chunk via [`Self::read_chunk_cols`] so projection
    /// pushes down into the format's decoder on every worker.
    fn par_load_batched_cols<P: AsRef<Path>>(
        dir: P,
        base: &str,
        options: Self::Options,
        threads: Option<usize>,
        columns: &[&str],
    ) -> Result<SuperTable, Self::Error>
    where
        Self: Sync,
        Self::Error: Send,
    {
        let reader = Self::open(dir, base, options)?;
        let paths = reader.paths();
        if paths.is_empty() {
            return Ok(SuperTable::from_batches(Vec::new(), Some("chunked".into())));
        }

        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_threads = threads.unwrap_or(default_threads).max(1).min(paths.len());
        let n_files = paths.len();
        let next_index = AtomicUsize::new(0);

        let per_worker_cap = n_files.div_ceil(n_threads);
        // Worker threads enter the caller's Tokio runtime, when there is
        // one, which lets object storage paths make their requests on it.
        let runtime = Handle::try_current().ok();
        let collected: Result<Vec<(usize, Table)>, Self::Error> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                let next_index = &next_index;
                let this = &reader;
                let runtime = &runtime;
                handles.push(
                    s.spawn(move || -> Result<Vec<(usize, Table)>, Self::Error> {
                        let _runtime = runtime.as_ref().map(Handle::enter);
                        let mut local: Vec<(usize, Table)> = Vec::with_capacity(per_worker_cap);
                        let paths = this.paths();
                        loop {
                            let idx = next_index.fetch_add(1, Ordering::Relaxed);
                            if idx >= n_files {
                                break;
                            }
                            let table = this.read_chunk_cols(&paths[idx], columns)?;
                            local.push((idx, table));
                        }
                        Ok(local)
                    }),
                );
            }
            let mut all: Vec<(usize, Table)> = Vec::with_capacity(n_files);
            for h in handles {
                let part = h.join().expect("chunked reader worker panicked")?;
                all.extend(part);
            }
            Ok(all)
        });

        let mut all = collected?;
        all.sort_by_key(|(i, _)| *i);
        let batches: Vec<Arc<Table>> = all.into_iter().map(|(_, t)| Arc::new(t)).collect();
        Ok(SuperTable::from_batches(batches, Some("chunked".into())))
    }
}

