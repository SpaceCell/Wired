// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # JSON interface
//!
//! Maps a vendor's JSON wire shape onto a declared schema and executes
//! that mapping frame by frame, resolving each column to a
//! [`JsonValueRef`](crate::models::decoders::json::value::JsonValueRef).
//! The destination performs the typed conversion through
//! [`push_value_into`](crate::models::decoders::json::push::push_value_into).
//!
//! ## JSON decoder comparison
//! The bulk JSON decoder in [`decoders::json`](crate::models::decoders::json)
//! takes a wire shape that already matches the table - JSON keys as column names,
//! and one flat object per row. In contrast, this interface takes a connector's shape
//! (that may differ by vendor, etc.) and a [`JsonSchema`]'s per-column
//! [`ValueSource`] to map it onto the same columns.
//!
//! ## Behaviour
//!
//! Flow: The mapping compiles once at construction. Each frame is parsed through
//! simd-json's tape with reusable [`Buffers`]. The record envelope is
//! then resolved, and every record yields one value per column.
//!
//! Timing: A wall-clock column fills from the caller's receive `now`,
//! passed in epoch nanoseconds and scaled to the column's time unit when
//! the source compiles.
//!
//! Size Limits: [`DecodeLimits`] caps frame bytes, records per frame, and string value
//! length before the corresponding work happens. Downstream connector input is considered
//! untrusted.

pub mod path;
pub mod schema;

use std::io;

use simd_json::Buffers;
use simd_json::value::tape::Node;

use crate::models::decoders::limits::DecodeLimits;
use crate::models::frames::json::JsonFrame;
use minarrow::Field;

pub use crate::models::interfaces::json::path::JsonPath;
pub use crate::models::interfaces::json::schema::{JsonSchema, ReadAs};

/// Where a column reads its value on each record.
#[derive(Debug, Clone)]
pub enum ValueSource {
    /// Reads the value under `key` on the record object, the common
    /// single-key lookup.
    RecordKey(String),

    /// Walks a nested path through the record's arrays and objects.
    JsonPath(JsonPath),

    /// Fills the column from the caller's receive clock, scaled to the
    /// column's time unit.
    WallClock,

    /// Walks a path through the frame's envelope rather than the record,
    /// resolved once and shared by every record the frame yields. A frame
    /// missing the path is an error.
    FramePath(JsonPath),
}

/// Nanoseconds per tick of `field`'s time unit, dividing the receive clock
/// down to the column's unit. A non-temporal column takes the clock as raw
/// nanoseconds.
#[cfg(feature = "datetime")]
fn clock_nanos_per_tick(field: &Field) -> i64 {
    use minarrow::enums::time_units::TimeUnit;
    use minarrow::{ArrowType, Dt};
    let unit = match &field.dtype {
        ArrowType::DateTime(
            Dt::Timestamp(unit, _)
            | Dt::Time32(unit)
            | Dt::Time64(unit)
            | Dt::Duration32(unit)
            | Dt::Duration64(unit),
        ) => *unit,
        ArrowType::DateTime(Dt::Date64) => TimeUnit::Milliseconds,
        ArrowType::DateTime(Dt::Date32) => TimeUnit::Days,
        _ => return 1,
    };
    match unit {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Milliseconds => 1_000_000,
        TimeUnit::Seconds => 1_000_000_000,
        TimeUnit::Days => 86_400_000_000_000,
    }
}

/// Without the `datetime` feature there are no temporal columns, so the
/// receive clock is the raw nanosecond value.
#[cfg(not(feature = "datetime"))]
fn clock_nanos_per_tick(_field: &Field) -> i64 {
    1
}

/// Executes a [`JsonSchema`] over wire frames.
///
/// ## Behaviour
/// - The per-column mapping compiles at construction.
/// - Each [`parse_frame`](Self::parse_frame) call parses the frame with reusable
///   [`Buffers`] and returns a [`JsonFrame`] positioned on the frame's
///   records.
/// - `Ok(None)` means the frame carries no records for this schema
///   i.e., could be a heartbeat, subscribe ack, or another channel's frame on a
///   multiplexed socket - which is normal traffic rather than an error.
pub struct JsonInterface {
    sources: Vec<ValueSource>,
    wall_clock_scale: Vec<i64>,
    record_path: Option<JsonPath>,
    limits: DecodeLimits,
    buffers: Buffers,
}

impl JsonInterface {
    /// Compile `schema` into an executable mapping. Fails on an empty path
    /// or record path, so endpoint construction surfaces schema errors
    /// before any frame arrives.
    pub fn new(schema: &JsonSchema, limits: DecodeLimits) -> io::Result<Self> {
        for (field, source) in schema.fields().iter().zip(schema.sources()) {
            if let ValueSource::JsonPath(path) | ValueSource::FramePath(path) = source
                && path.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("column '{}' has an empty path", field.name),
                ));
            }
        }
        let sources = schema.sources().to_vec();
        let wall_clock_scale = schema.fields().iter().map(clock_nanos_per_tick).collect();
        let record_path = match schema.record_path() {
            Some(spec) => {
                let path = JsonPath::parse(spec);
                if path.is_empty() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "record_path is empty"));
                }
                Some(path)
            }
            None => None,
        };
        Ok(Self { sources, wall_clock_scale, record_path, limits, buffers: Buffers::default() })
    }

    /// Number of columns in the compiled mapping.
    pub fn n_columns(&self) -> usize {
        self.sources.len()
    }

    /// Parse one frame and position on its records.
    ///
    /// ## Behaviour
    /// - `frame` is parsed in place - simd-json unescapes strings directly
    ///   into the buffer, overwriting the original JSON text.
    /// - `now` is the caller's receive clock in epoch nanoseconds, captured once for the frame,
    ///   which gets scaled to the column's time unit when `WallClock` is used.
    ///
    /// Returns `Ok(None)` when the configured record path is absent from
    /// the frame. Returns an error for:
    ///     - malformed JSON
    ///     - record paths resolving to a scalar
    ///     - frames exceeding the limits.
    pub fn parse_frame<'f>(
        &'f mut self,
        frame: &'f mut [u8],
        now: i64,
    ) -> io::Result<Option<JsonFrame<'f>>> {
        self.limits
            .check(frame.len(), self.limits.max_frame_bytes, "json frame bytes")?;
        let tape = simd_json::to_tape_with_buffers(frame, &mut self.buffers)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid JSON: {e}")))?;
        let nodes = tape.0;
        if nodes.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty JSON frame"));
        }

        let (next_record, records_remaining) = match &self.record_path {
            // Without a record path, the frame root is the single record.
            None => (0, 1),
            Some(path) => {
                let Some(at) = path.resolve(&nodes, 0) else {
                    return Ok(None);
                };
                match nodes[at] {
                    Node::Array { len, .. } => (at + 1, len),
                    Node::Object { .. } => (at, 1),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "record path is neither an array nor an object",
                        ));
                    }
                }
            }
        };
        self.limits
            .check(records_remaining, self.limits.max_n_rows, "records per frame")?;

        Ok(Some(JsonFrame {
            nodes,
            sources: &self.sources,
            wall_clock_scale: &self.wall_clock_scale,
            max_string_bytes: self.limits.max_string_bytes,
            next_record,
            records_remaining,
            now,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use minarrow::ArrowType;

    use crate::models::decoders::json::value::JsonValueRef;

    /// Kraken-book-shaped schema where rows are the levels under
    /// `data.0.bids`. `event_time` and `checksum` read the frame envelope.
    fn book_schema() -> JsonSchema {
        // Int64 stands in for Timestamp here, since the interface only maps
        // and yields wire values. The dtype matters to the destination,
        // where Timestamp is feature-gated in minarrow.
        JsonSchema::new()
            .column(
                Field::new("event_time", ArrowType::Int64, false, None),
                ValueSource::FramePath(JsonPath::parse("data.0.timestamp")),
                ReadAs::Datetime,
            )
            .column(
                Field::new("price", ArrowType::Float64, false, None),
                ValueSource::RecordKey("price".into()),
                ReadAs::Verbatim,
            )
            .column(
                Field::new("qty", ArrowType::Float64, false, None),
                ValueSource::RecordKey("qty".into()),
                ReadAs::Verbatim,
            )
            .column(
                Field::new("checksum", ArrowType::Int64, false, None),
                ValueSource::FramePath(JsonPath::parse("data.0.checksum")),
                ReadAs::Verbatim,
            )
            .with_record_path("data.0.bids")
    }

    #[test]
    fn envelope_records_resolve_with_frame_paths() {
        let mut interface = JsonInterface::new(&book_schema(), DecodeLimits::default()).unwrap();
        let mut frame = br#"{"channel":"book","type":"update","data":[{"symbol":"BTC/USD","bids":[{"price":63245.1,"qty":0.5},{"price":63245.0,"qty":0.0}],"asks":[],"checksum":222,"timestamp":"2026-06-09T07:48:36.123456Z"}]}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 1_000).unwrap().unwrap();
        assert_eq!(parsed.records_remaining(), 2);

        let first = parsed.next_record().unwrap();
        assert!(matches!(
            first.value(0).unwrap(),
            JsonValueRef::Str("2026-06-09T07:48:36.123456Z"),
        ));
        assert!(matches!(first.value(1).unwrap(), JsonValueRef::F64(v) if (v - 63245.1).abs() < 1e-9));
        assert!(matches!(first.value(3).unwrap(), JsonValueRef::U64(222)));
        drop(first);

        let second = parsed.next_record().unwrap();
        assert!(matches!(second.value(2).unwrap(), JsonValueRef::F64(v) if v == 0.0));
        assert!(matches!(second.value(3).unwrap(), JsonValueRef::U64(222)));
        drop(second);

        assert!(parsed.next_record().is_none());
    }

    #[test]
    fn frame_missing_its_frame_path_is_an_error() {
        let mut interface = JsonInterface::new(&book_schema(), DecodeLimits::default()).unwrap();
        // Snapshot frame with no envelope timestamp for event_time's frame path.
        let mut frame = br#"{"channel":"book","type":"snapshot","data":[{"symbol":"BTC/USD","bids":[{"price":63245.1,"qty":0.5}],"asks":[],"checksum":111}]}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 0).unwrap().unwrap();
        let record = parsed.next_record().unwrap();
        assert!(record.value(0).is_err());
        assert!(matches!(record.value(3).unwrap(), JsonValueRef::U64(111)));
    }

    #[test]
    fn missing_record_path_is_not_an_error() {
        let mut interface = JsonInterface::new(&book_schema(), DecodeLimits::default()).unwrap();
        let mut heartbeat = br#"{"channel":"heartbeat"}"#.to_vec();
        assert!(interface.parse_frame(&mut heartbeat, 0).unwrap().is_none());
    }

    #[test]
    fn scalar_at_record_path_is_an_error() {
        let schema = JsonSchema::new()
            .column(
                Field::new("x", ArrowType::Int64, false, None),
                ValueSource::RecordKey("x".into()),
                ReadAs::Verbatim,
            )
            .with_record_path("data");
        let mut interface = JsonInterface::new(&schema, DecodeLimits::default()).unwrap();
        let mut frame = br#"{"data":42}"#.to_vec();
        assert!(interface.parse_frame(&mut frame, 0).is_err());
    }

    #[test]
    fn object_envelope_decodes_as_a_single_record() {
        // Combined-stream shape with one object under `data`.
        let schema = JsonSchema::new()
            .column(
                Field::new("trade_id", ArrowType::Int64, false, None),
                ValueSource::RecordKey("t".into()),
                ReadAs::Verbatim,
            )
            .with_record_path("data");
        let mut interface = JsonInterface::new(&schema, DecodeLimits::default()).unwrap();
        let mut frame = br#"{"stream":"btcusdt@trade","data":{"e":"trade","t":12345}}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 0).unwrap().unwrap();
        let record = parsed.next_record().unwrap();
        assert!(matches!(record.value(0).unwrap(), JsonValueRef::U64(12345)));
        drop(record);
        assert!(parsed.next_record().is_none());
    }

    #[test]
    fn no_record_path_reads_the_frame_root_as_one_record() {
        let schema = JsonSchema::new()
            .column(
                Field::new("event_time", ArrowType::Int64, false, None),
                ValueSource::RecordKey("E".into()),
                ReadAs::Verbatim,
            )
            .column(
                Field::new("bid_px_0", ArrowType::Float64, false, None),
                ValueSource::JsonPath(JsonPath::parse("bids.0.0")),
                ReadAs::Verbatim,
            );
        let mut interface = JsonInterface::new(&schema, DecodeLimits::default()).unwrap();
        let mut frame = br#"{"E":1700000000000,"bids":[["63245.10","0.5"]]}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 0).unwrap().unwrap();
        let record = parsed.next_record().unwrap();
        assert!(matches!(record.value(0).unwrap(), JsonValueRef::U64(1_700_000_000_000)));
        // The nested path lands on the quoted string. Conversion is the
        // destination's job.
        assert!(matches!(record.value(1).unwrap(), JsonValueRef::Str("63245.10")));
    }

    #[test]
    fn wall_clock_yields_the_caller_clock() {
        let schema = JsonSchema::new().column(
            Field::new("recv_time", ArrowType::Int64, false, None),
            ValueSource::WallClock,
            ReadAs::Verbatim,
        );
        let mut interface = JsonInterface::new(&schema, DecodeLimits::default()).unwrap();
        let mut frame = br#"{}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 777).unwrap().unwrap();
        let record = parsed.next_record().unwrap();
        assert!(matches!(record.value(0).unwrap(), JsonValueRef::I64(777)));
    }

    #[test]
    fn missing_key_is_an_error() {
        let schema = JsonSchema::new().column(
            Field::new("price", ArrowType::Float64, false, None),
            ValueSource::RecordKey("price".into()),
            ReadAs::Verbatim,
        );
        let mut interface = JsonInterface::new(&schema, DecodeLimits::default()).unwrap();
        let mut frame = br#"{"qty":1.0}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 0).unwrap().unwrap();
        let record = parsed.next_record().unwrap();
        assert!(record.value(0).is_err());
    }

    fn one_int_column(name: &str) -> JsonSchema {
        JsonSchema::new().column(
            Field::new(name, ArrowType::Int64, false, None),
            ValueSource::RecordKey(name.into()),
            ReadAs::Verbatim,
        )
    }

    #[test]
    fn limits_cap_frame_bytes_records_and_string_length() {
        let limits = DecodeLimits { max_frame_bytes: 8, ..DecodeLimits::default() };
        let schema = one_int_column("x");
        let mut interface = JsonInterface::new(&schema, limits).unwrap();
        let mut frame = br#"{"x":12345678}"#.to_vec();
        assert!(interface.parse_frame(&mut frame, 0).is_err());

        let limits = DecodeLimits { max_n_rows: 1, ..DecodeLimits::default() };
        let schema = one_int_column("x").with_record_path("data");
        let mut interface = JsonInterface::new(&schema, limits).unwrap();
        let mut frame = br#"{"data":[{"x":1},{"x":2}]}"#.to_vec();
        assert!(interface.parse_frame(&mut frame, 0).is_err());

        let limits = DecodeLimits { max_string_bytes: 4, ..DecodeLimits::default() };
        let schema = JsonSchema::new().column(
            Field::new("s", ArrowType::String, false, None),
            ValueSource::RecordKey("s".into()),
            ReadAs::Verbatim,
        );
        let mut interface = JsonInterface::new(&schema, limits).unwrap();
        let mut frame = br#"{"s":"abcdefgh"}"#.to_vec();
        let mut parsed = interface.parse_frame(&mut frame, 0).unwrap().unwrap();
        let record = parsed.next_record().unwrap();
        assert!(record.value(0).is_err());
    }

    #[test]
    fn an_empty_path_surfaces_at_construction() {
        let schema = JsonSchema::new().column(
            Field::new("x", ArrowType::Int64, false, None),
            ValueSource::JsonPath(JsonPath::parse("")),
            ReadAs::Verbatim,
        );
        assert!(JsonInterface::new(&schema, DecodeLimits::default()).is_err());
    }
}
