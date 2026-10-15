// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunk partitioning
//!
//! Partition paths for the chunked writers. A partitioned writer places each
//! chunk in a subdirectory of its root, for example
//! `<dir>/date=2026-10-14/hour=13/<base>-NNNNNNNNNN.parquet`.
//!
//! ## Context
//! Readers locate chunks by listing a directory or object prefix. The
//! partition path determines how much a reader needs to list to find the
//! data for one period of time. This matters most on object storage, where
//! each page of a listing is a separate request.
//! The chunked readers include chunks in nested partition directories,
//! and opening a reader on a partition path such as `<dir>/date=2026-10-14`
//! reads that partition alone.
//!
//! ## Partition types
//! - [`PartitionType::Auto`] derives the path from the time of each write in
//!   the `key=value` directory convention. Query engines such as DuckDB,
//!   Polars and Spark recognise this convention, using it to prune by path
//!   and exposing the values as columns.
//! - [`PartitionType::Custom`] takes the path from a caller-supplied
//!   function, for schemes keyed on other columns such as venue or symbol.
//!
//! The time is supplied by the caller through
//! [`ChunkedTableWriter::write_chunk_for_time`](crate::traits::chunked_table_writer::ChunkedTableWriter::write_chunk_for_time),
//! for example from the event-time window that produced the chunk. The
//! other write methods use the current system time.

use std::fmt;
use std::sync::Arc;

use minarrow::Table;

const NANOS_PER_SECOND: u64 = 1_000_000_000;
const SECONDS_PER_DAY: u64 = 86_400;
const SECONDS_PER_HOUR: u64 = 3_600;

/// How a chunked writer groups its chunks into partition directories.
#[derive(Clone)]
pub enum PartitionType {
    /// `date=YYYY-MM-DD/hour=HH` in UTC, from the time of each write.
    ///
    /// Hourly partitions bound the listing a reader needs for any period.
    Auto,
    /// Partition path returned by the function for each write, given the
    /// Table and its time in epoch nanoseconds.
    ///
    /// The path is `/`-separated and relative to the writer's directory,
    /// for example `venue=XNAS/date=2026-10-14`.
    Custom(Arc<dyn Fn(&Table, u64) -> String + Send + Sync>),
}

impl PartitionType {
    /// Partition path for a write of `table` at `time_ns` epoch nanoseconds.
    pub fn partition(&self, table: &Table, time_ns: u64) -> String {
        match self {
            Self::Auto => auto_partition(time_ns),
            Self::Custom(partition) => partition(table, time_ns),
        }
    }
}

impl fmt::Debug for PartitionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("Auto"),
            Self::Custom(_) => f.write_str("Custom"),
        }
    }
}

/// Format an epoch-nanosecond time as `date=YYYY-MM-DD/hour=HH` in UTC.
fn auto_partition(time_ns: u64) -> String {
    let epoch_secs = time_ns / NANOS_PER_SECOND;
    let (year, month, day) = civil_from_days((epoch_secs / SECONDS_PER_DAY) as i64);
    let hour = epoch_secs % SECONDS_PER_DAY / SECONDS_PER_HOUR;
    format!("date={year:04}-{month:02}-{day:02}/hour={hour:02}")
}

/// The civil date `(year, month, day)` for a day count from the Unix epoch,
/// using Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-02-09T07:48:36.123456789Z
    const SAMPLE: u64 = 1_770_623_316_123_456_789;

    #[test]
    fn auto_partition_formats_date_and_hour_in_utc() {
        assert_eq!(auto_partition(SAMPLE), "date=2026-02-09/hour=07");
    }

    #[test]
    fn auto_partition_at_the_epoch_is_the_first_hour_of_1970() {
        assert_eq!(auto_partition(0), "date=1970-01-01/hour=00");
    }

    #[test]
    fn auto_partition_handles_the_leap_day() {
        // 2024-02-29T23:59:59Z
        let time = 1_709_251_199 * NANOS_PER_SECOND;
        assert_eq!(auto_partition(time), "date=2024-02-29/hour=23");
        assert_eq!(
            auto_partition(time + NANOS_PER_SECOND),
            "date=2024-03-01/hour=00"
        );
    }

    #[test]
    fn custom_partition_receives_the_table_and_time() {
        let partition = PartitionType::Custom(Arc::new(|table: &Table, time: u64| {
            format!("table={}/time={time}", table.name)
        }));
        let table = Table::new("trades".to_string(), None);
        assert_eq!(partition.partition(&table, 42), "table=trades/time=42");
    }
}
