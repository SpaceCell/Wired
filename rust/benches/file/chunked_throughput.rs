// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Chunked-format throughput benchmarks. Linux only.
//!
//! Writes approximately 100 MiB of data as numbered chunk files, then measures
//! serial and parallel reads before removing the temporary directory. Arrow IPC
//! and CSV are always included; Parquet requires the `parquet` feature.
//!
//! Each chunk contains 100,000 rows, or approximately 3 MiB for the benchmark
//! table. A complete iteration writes 32 chunks using distinct in-memory tables
//! so the encoder processes data beyond typical CPU cache capacity.
//!
//! ## Write benchmarks
//!
//! Each format has two write benchmarks:
//!
//! - `*_logical` measures encoding and buffered writes. It excludes `fsync` and
//!   reports throughput using the logical size of the source column buffers.
//! - `*_physical` measures encoding, writes and durability. Each chunk file and
//!   the parent directory are synchronised before the timed region ends.
//!   Throughput uses the resulting file size.
//!
//! Directory cleanup runs outside the timed region. Logical and physical results
//! use different denominators and should only be compared to assess the cost of
//! durable writes.
//!
//! Read benchmarks use the physical file size as their throughput denominator.
//!
//! ## Linux requirement
//!
//! Before each read iteration, every chunk file is removed from the page cache
//! with `posix_fadvise(POSIX_FADV_DONTNEED)`. Cache eviction runs outside the
//! timed region, and each iteration begins with cold pages.
//!
//! The benchmark is Linux-only because other supported platforms do not provide
//! an equivalent cache-eviction interface with the same semantics.


use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lightstream::traits::chunked_table_reader::ChunkedTableReader;
use lightstream::traits::chunked_table_writer::ChunkedTableWriter;

#[path = "../common/bench_helpers.rs"]
mod bench_helpers;
use bench_helpers::{BENCH_ROWS, logical_payload_bytes, make_bench_table};

const N_CHUNKS: usize = 32;
const BASE: &str = "chunk";

// `/var/tmp` rather than `std::env::temp_dir()`, because `/tmp` is commonly
// mounted as `tmpfs` where cache eviction produces no cold storage read and
// `fsync` costs almost nothing, which would distort the cold-read and
// physical-write measurements.
const BENCH_ROOT: &str = "/var/tmp";

fn fresh_dir(suffix: &str) -> PathBuf {
    let dir = PathBuf::from(BENCH_ROOT).join(format!("lightstream_chunked_bench_{suffix}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Returns the total size of regular files directly under `dir`.
fn physical_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

fn total_payload_bytes() -> u64 {
    logical_payload_bytes(BENCH_ROWS) * N_CHUNKS as u64
}

/// Requests eviction of each chunk file from the page cache.
///
/// Errors are ignored because `POSIX_FADV_DONTNEED` is advisory. This function
/// runs outside the timed region.
fn evict_pages(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        if let Ok(f) = std::fs::File::open(entry.path()) {
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }
}

fn bench_chunked_arrow(c: &mut Criterion) {
    use lightstream::models::readers::chunked::arrow::ChunkedArrowReader;
    use lightstream::models::writers::chunked::arrow::ChunkedArrowWriter;
    use minarrow::Table;

    // Use distinct tables so each iteration reads separate source buffers.
    let tables: Vec<Table> = (0..N_CHUNKS)
        .map(|_| make_bench_table(BENCH_ROWS))
        .collect();
    let table_refs: Vec<&Table> = tables.iter().collect();

    // Write one dataset to determine the encoded size used by physical
    // throughput measurements.
    let read_dir = fresh_dir("arrow_read");
    {
        let mut w = ChunkedArrowWriter::new(&read_dir, BASE, None).unwrap();
        for t in &tables {
            w.write_chunk(t).unwrap();
        }
    }

    let logical_bytes = total_payload_bytes();
    let physical_bytes = physical_bytes(&read_dir);

    let mut group = c.benchmark_group("chunked_arrow");
    group.sample_size(10);

    // Logical writes measure encoding and buffered file writes using the source
    // payload size. Physical writes also synchronise each file and the parent
    // directory, using the encoded file size.
    //
    // Directory cleanup runs in the next iteration's setup and is not timed.
    let write_serial_logical = |dir: PathBuf| {
        let mut w = ChunkedArrowWriter::new(&dir, BASE, None).unwrap();
        for t in &tables {
            w.write_chunk(t).unwrap();
        }
    };

    let write_serial_physical = |dir: PathBuf| {
        let mut w = ChunkedArrowWriter::new(&dir, BASE, None).unwrap();
        let mut paths: Vec<PathBuf> = Vec::with_capacity(N_CHUNKS);

        for t in &tables {
            paths.push(w.write_chunk(t).unwrap());
        }

        for p in &paths {
            std::fs::File::open(p).unwrap().sync_all().unwrap();
        }

        // Synchronise directory metadata for the newly created chunk files.
        std::fs::File::open(&dir).unwrap().sync_all().unwrap();
    };


    group.throughput(Throughput::Bytes(logical_bytes));
    group.bench_function("write_logical", |b| {
        b.iter_batched(
            || fresh_dir("arrow_write"),
            &write_serial_logical,
            criterion::BatchSize::PerIteration,
        );
    });
    group.throughput(Throughput::Bytes(physical_bytes));
    group.bench_function("write_physical", |b| {
        b.iter_batched(
            || fresh_dir("arrow_write"),
            &write_serial_physical,
            criterion::BatchSize::PerIteration,
        );
    });

    let par_write_logical = |dir: PathBuf| {
        let w = ChunkedArrowWriter::new(&dir, BASE, None).unwrap();
        let paths = w.par_write_all(&table_refs, None).unwrap();
        assert_eq!(paths.len(), N_CHUNKS);
    };
    let par_write_physical = |dir: PathBuf| {
        let w = ChunkedArrowWriter::new(&dir, BASE, None).unwrap();
        let paths = w.par_write_all(&table_refs, None).unwrap();
        assert_eq!(paths.len(), N_CHUNKS);
        for p in &paths {
            std::fs::File::open(p).unwrap().sync_all().unwrap();
        }
        std::fs::File::open(&dir).unwrap().sync_all().unwrap();
    };

    group.throughput(Throughput::Bytes(logical_bytes));
    group.bench_function("par_write_logical", |b| {
        b.iter_batched(
            || fresh_dir("arrow_par_write"),
            &par_write_logical,
            criterion::BatchSize::PerIteration,
        );
    });
    group.throughput(Throughput::Bytes(physical_bytes));
    group.bench_function("par_write_physical", |b| {
        b.iter_batched(
            || fresh_dir("arrow_par_write"),
            &par_write_physical,
            criterion::BatchSize::PerIteration,
        );
    });

    // Read benches use the physical denominator since that's what
    // actually flows from disk into memory per second.
    group.throughput(Throughput::Bytes(physical_bytes));

    group.bench_function("serial_read_all", |b| {
        b.iter_batched(
            || evict_pages(&read_dir),
            |_| {
                let r = ChunkedArrowReader::open(&read_dir, BASE, ()).unwrap();
                let st = r.load_batched().unwrap();
                assert_eq!(st.batches.len(), N_CHUNKS);
                assert_eq!(st.n_rows, N_CHUNKS * BENCH_ROWS);
                std::hint::black_box(st);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function("par_load_batched", |b| {
        b.iter_batched(
            || evict_pages(&read_dir),
            |_| {
                let st = ChunkedArrowReader::par_load_batched(&read_dir, BASE, (), None).unwrap();
                assert_eq!(st.batches.len(), N_CHUNKS);
                assert_eq!(st.n_rows, N_CHUNKS * BENCH_ROWS);
                std::hint::black_box(st);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // Reap the write benches' final-iteration directories alongside
    // the read benches' directory now that all timed work is done.
    cleanup(&read_dir);
    cleanup(&PathBuf::from(BENCH_ROOT).join("lightstream_chunked_bench_arrow_write"));
    cleanup(&PathBuf::from(BENCH_ROOT).join("lightstream_chunked_bench_arrow_par_write"));
    group.finish();
}

#[cfg(feature = "parquet")]
fn bench_chunked_parquet(c: &mut Criterion) {
    use lightstream::models::readers::chunked::parquet::ChunkedParquetReader;
    use lightstream::models::writers::chunked::parquet::ChunkedParquetWriter;
    use minarrow::Table;

    // 32 distinct tables in distinct memory so the encoder cannot
    // unrealistically benefit from L2/L3 cache residency.
    let tables: Vec<Table> = (0..N_CHUNKS)
        .map(|_| make_bench_table(BENCH_ROWS))
        .collect();
    let table_refs: Vec<&Table> = tables.iter().collect();

    // Pre-write the dataset once to learn the real physical byte count.
    let read_dir = fresh_dir("parquet_read");
    {
        let mut w = ChunkedParquetWriter::new(&read_dir, BASE, None, None).unwrap();
        for t in &tables {
            w.write_chunk(t).unwrap();
        }
    }
    let logical_bytes = total_payload_bytes();
    let physical_bytes = physical_bytes(&read_dir);

    let mut group = c.benchmark_group("chunked_parquet");
    group.sample_size(10);

    // `_logical`: encode + write, no fsync. Denominator = source bytes.
    // `_physical`: encode + write + fsync per chunk + fsync the parent
    // dir. Denominator = output file bytes. Cleanup of the chunk dir
    // is NOT in the timed region.
    let write_serial_logical = |dir: PathBuf| {
        let mut w = ChunkedParquetWriter::new(&dir, BASE, None, None).unwrap();
        for t in &tables {
            w.write_chunk(t).unwrap();
        }
    };
    let write_serial_physical = |dir: PathBuf| {
        let mut w = ChunkedParquetWriter::new(&dir, BASE, None, None).unwrap();
        let mut paths: Vec<PathBuf> = Vec::with_capacity(N_CHUNKS);
        for t in &tables {
            paths.push(w.write_chunk(t).unwrap());
        }
        for p in &paths {
            std::fs::File::open(p).unwrap().sync_all().unwrap();
        }
        // Fsync the parent dir so the chunk dir entries themselves are durable.
        std::fs::File::open(&dir).unwrap().sync_all().unwrap();
    };

    group.throughput(Throughput::Bytes(logical_bytes));
    group.bench_function("write_logical", |b| {
        b.iter_batched(
            || fresh_dir("parquet_write"),
            &write_serial_logical,
            criterion::BatchSize::PerIteration,
        );
    });
    group.throughput(Throughput::Bytes(physical_bytes));
    group.bench_function("write_physical", |b| {
        b.iter_batched(
            || fresh_dir("parquet_write"),
            &write_serial_physical,
            criterion::BatchSize::PerIteration,
        );
    });

    let par_write_logical = |dir: PathBuf| {
        let w = ChunkedParquetWriter::new(&dir, BASE, None, None).unwrap();
        let paths = w.par_write_all(&table_refs, None).unwrap();
        assert_eq!(paths.len(), N_CHUNKS);
    };
    let par_write_physical = |dir: PathBuf| {
        let w = ChunkedParquetWriter::new(&dir, BASE, None, None).unwrap();
        let paths = w.par_write_all(&table_refs, None).unwrap();
        assert_eq!(paths.len(), N_CHUNKS);
        for p in &paths {
            std::fs::File::open(p).unwrap().sync_all().unwrap();
        }
        std::fs::File::open(&dir).unwrap().sync_all().unwrap();
    };

    group.throughput(Throughput::Bytes(logical_bytes));
    group.bench_function("par_write_logical", |b| {
        b.iter_batched(
            || fresh_dir("parquet_par_write"),
            &par_write_logical,
            criterion::BatchSize::PerIteration,
        );
    });
    group.throughput(Throughput::Bytes(physical_bytes));
    group.bench_function("par_write_physical", |b| {
        b.iter_batched(
            || fresh_dir("parquet_par_write"),
            &par_write_physical,
            criterion::BatchSize::PerIteration,
        );
    });

    group.throughput(Throughput::Bytes(physical_bytes));

    group.bench_function("serial_read_all", |b| {
        b.iter_batched(
            || evict_pages(&read_dir),
            |_| {
                let r = ChunkedParquetReader::open(&read_dir, BASE, ()).unwrap();
                let st = r.load_batched().unwrap();
                assert_eq!(st.batches.len(), N_CHUNKS);
                assert_eq!(st.n_rows, N_CHUNKS * BENCH_ROWS);
                std::hint::black_box(st);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function("par_load_batched", |b| {
        b.iter_batched(
            || evict_pages(&read_dir),
            |_| {
                let st = ChunkedParquetReader::par_load_batched(&read_dir, BASE, (), None).unwrap();
                assert_eq!(st.batches.len(), N_CHUNKS);
                assert_eq!(st.n_rows, N_CHUNKS * BENCH_ROWS);
                std::hint::black_box(st);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    cleanup(&read_dir);
    cleanup(&PathBuf::from(BENCH_ROOT).join("lightstream_chunked_bench_parquet_write"));
    cleanup(&PathBuf::from(BENCH_ROOT).join("lightstream_chunked_bench_parquet_par_write"));
    group.finish();
}

#[cfg(feature = "csv")]
fn bench_chunked_csv(c: &mut Criterion) {
    use lightstream::models::decoders::csv::CsvDecodeOptions;
    use lightstream::models::encoders::csv::CsvEncodeOptions;
    use lightstream::models::readers::chunked::csv::{ChunkedCsvReadOptions, ChunkedCsvReader};
    use lightstream::models::writers::chunked::csv::ChunkedCsvWriter;
    use minarrow::Table;

    // 32 distinct tables in distinct memory so the encoder cannot
    // unrealistically benefit from L2/L3 cache residency.
    let tables: Vec<Table> = (0..N_CHUNKS)
        .map(|_| make_bench_table(BENCH_ROWS))
        .collect();
    let table_refs: Vec<&Table> = tables.iter().collect();

    // Pre-write the dataset once to learn the real physical byte count.
    let read_dir = fresh_dir("csv_read");
    {
        let mut w = ChunkedCsvWriter::new(&read_dir, BASE, CsvEncodeOptions::default(), None).unwrap();
        for t in &tables {
            w.write_chunk(t).unwrap();
        }
    }
    let logical_bytes = total_payload_bytes();
    let physical_bytes = physical_bytes(&read_dir);

    let mut group = c.benchmark_group("chunked_csv");
    group.sample_size(10);

    // Logical writes exclude synchronisation and use the source payload size.
    // Physical writes synchronise each chunk and the parent directory, and use the
    // encoded file size. Directory cleanup runs outside the timed region.
    let write_serial_logical = |dir: PathBuf| {
        let mut w = ChunkedCsvWriter::new(&dir, BASE, CsvEncodeOptions::default(), None).unwrap();
        for t in &tables {
            w.write_chunk(t).unwrap();
        }
    };
    
    let write_serial_physical = |dir: PathBuf| {
        let mut w = ChunkedCsvWriter::new(&dir, BASE, CsvEncodeOptions::default(), None).unwrap();
        let mut paths: Vec<PathBuf> = Vec::with_capacity(N_CHUNKS);
        for t in &tables {
            paths.push(w.write_chunk(t).unwrap());
        }
        for p in &paths {
            std::fs::File::open(p).unwrap().sync_all().unwrap();
        }
        // Fsync the parent dir so the chunk dir entries themselves are durable.
        std::fs::File::open(&dir).unwrap().sync_all().unwrap();
    };

    group.throughput(Throughput::Bytes(logical_bytes));
    group.bench_function("write_logical", |b| {
        b.iter_batched(
            || fresh_dir("csv_write"),
            &write_serial_logical,
            criterion::BatchSize::PerIteration,
        );
    });
    group.throughput(Throughput::Bytes(physical_bytes));
    group.bench_function("write_physical", |b| {
        b.iter_batched(
            || fresh_dir("csv_write"),
            &write_serial_physical,
            criterion::BatchSize::PerIteration,
        );
    });

    let par_write_logical = |dir: PathBuf| {
        let w = ChunkedCsvWriter::new(&dir, BASE, CsvEncodeOptions::default(), None).unwrap();
        let paths = w.par_write_all(&table_refs, None).unwrap();
        assert_eq!(paths.len(), N_CHUNKS);
    };
    let par_write_physical = |dir: PathBuf| {
        let w = ChunkedCsvWriter::new(&dir, BASE, CsvEncodeOptions::default(), None).unwrap();
        let paths = w.par_write_all(&table_refs, None).unwrap();
        assert_eq!(paths.len(), N_CHUNKS);
        for p in &paths {
            std::fs::File::open(p).unwrap().sync_all().unwrap();
        }
        std::fs::File::open(&dir).unwrap().sync_all().unwrap();
    };

    group.throughput(Throughput::Bytes(logical_bytes));
    group.bench_function("par_write_logical", |b| {
        b.iter_batched(
            || fresh_dir("csv_par_write"),
            &par_write_logical,
            criterion::BatchSize::PerIteration,
        );
    });
    group.throughput(Throughput::Bytes(physical_bytes));
    group.bench_function("par_write_physical", |b| {
        b.iter_batched(
            || fresh_dir("csv_par_write"),
            &par_write_physical,
            criterion::BatchSize::PerIteration,
        );
    });

    group.throughput(Throughput::Bytes(physical_bytes));

    let read_opts = || ChunkedCsvReadOptions {
        decode: CsvDecodeOptions::default(),
        // Match the chunk size so each file resolves into one Table on
        // the reader side without further internal splitting.
        batch_size: BENCH_ROWS,
    };

    group.bench_function("serial_read_all", |b| {
        b.iter_batched(
            || evict_pages(&read_dir),
            |_| {
                let r = ChunkedCsvReader::open(&read_dir, BASE, read_opts()).unwrap();
                let st = r.load_batched().unwrap();
                assert_eq!(st.batches.len(), N_CHUNKS);
                assert_eq!(st.n_rows, N_CHUNKS * BENCH_ROWS);
                std::hint::black_box(st);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function("par_load_batched", |b| {
        b.iter_batched(
            || evict_pages(&read_dir),
            |_| {
                let st =
                    ChunkedCsvReader::par_load_batched(&read_dir, BASE, read_opts(), None).unwrap();
                assert_eq!(st.batches.len(), N_CHUNKS);
                assert_eq!(st.n_rows, N_CHUNKS * BENCH_ROWS);
                std::hint::black_box(st);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    cleanup(&read_dir);
    cleanup(&PathBuf::from(BENCH_ROOT).join("lightstream_chunked_bench_csv_write"));
    cleanup(&PathBuf::from(BENCH_ROOT).join("lightstream_chunked_bench_csv_par_write"));
    group.finish();
}

#[cfg(all(feature = "parquet", feature = "csv"))]
criterion_group!(
    benches,
    bench_chunked_arrow,
    bench_chunked_parquet,
    bench_chunked_csv
);
#[cfg(all(feature = "parquet", not(feature = "csv")))]
criterion_group!(benches, bench_chunked_arrow, bench_chunked_parquet);
#[cfg(all(not(feature = "parquet"), feature = "csv"))]
criterion_group!(benches, bench_chunked_arrow, bench_chunked_csv);
#[cfg(all(not(feature = "parquet"), not(feature = "csv")))]
criterion_group!(benches, bench_chunked_arrow);
criterion_main!(benches);
