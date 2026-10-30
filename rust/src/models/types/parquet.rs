// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow <-> Parquet Type Mapping
//!
//! This module includes conversions between Arrow/Minarrow types and Parquet physical/logical
//! types as defined in the Parquet specification.  
//! It's used during encoding and decoding of Parquet files to support type-safe interoperability.

use crate::error::IoError;
#[cfg(feature = "datetime")]
use minarrow::TimeUnit;
use minarrow::ArrowType;
#[cfg(feature = "datetime")]
use minarrow::Dt;

/// Parquet physical types as defined in `parquet.thrift`.
///
/// These represent the low-level storage format for values in Parquet
/// files, independent of higher-level logical annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParquetPhysicalType {
    /// Bitpacked Boolean
    Boolean = 0,
    /// 32-bit signed integer.
    Int32 = 1,
    /// 64-bit signed integer.
    Int64 = 2,
    /// 32-bit IEEE floating point.
    Float = 4,
    /// 64-bit IEEE floating point.
    Double = 5,
    /// Variable-length byte array (used for strings and binary data).
    ByteArray = 6,
    /// Fixed-length byte array (used for Decimal128 and other fixed-width types).
    FixedLenByteArray = 7,
}

impl ParquetPhysicalType {
    /// Return the Parquet `i32` type ID corresponding to this physical type.
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Convert from a Parquet `i32` type ID into a [`ParquetPhysicalType`].
    ///
    /// Returns `None` if the ID does not match a known type.
    pub fn from_i32(val: i32) -> Option<Self> {
        match val {
            0 => Some(Self::Boolean),
            1 => Some(Self::Int32),
            2 => Some(Self::Int64),
            4 => Some(Self::Float),
            5 => Some(Self::Double),
            6 => Some(Self::ByteArray),
            7 => Some(Self::FixedLenByteArray),
            _ => None,
        }
    }
}

/// Parquet logical (or "converted") types as defined in `parquet.thrift`.
///
/// These annotate a physical type with higher-level semantics,
/// e.g. `Utf8` over a `ByteArray` or `Date32` over an `Int32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParquetLogicalType {
    /// No logical annotation.
    NoneType,
    /// UTF-8 encoded string.
    Utf8,
    /// DATE - days since the Unix epoch, stored as INT32.
    #[cfg(feature = "datetime")]
    Date32,
    /// 64-bit timestamp - milliseconds since epoch. `utc` is Parquet's
    /// `isAdjustedToUTC`: true for instants in UTC, false for local
    /// wall-clock time with no zone.
    #[cfg(feature = "datetime")]
    TimestampMillis { utc: bool },
    /// 64-bit timestamp - microseconds since epoch, with the same `utc`
    /// meaning as `TimestampMillis`.
    #[cfg(feature = "datetime")]
    TimestampMicros { utc: bool },
    /// 64-bit timestamp - nanoseconds since epoch, with the same `utc`
    /// meaning as `TimestampMillis`. Exists only as a `LogicalType`.
    #[cfg(feature = "datetime")]
    TimestampNanos { utc: bool },
    /// 32-bit time - milliseconds since midnight
    #[cfg(feature = "datetime")]
    TimeMillis,
    /// 32-bit time - microseconds since midnight
    #[cfg(feature = "datetime")]
    TimeMicros,
    /// 64-bit time - nanoseconds since midnight
    #[cfg(feature = "datetime")]
    TimeNanos,
    /// Fixed-point decimal with precision and scale.
    #[cfg(feature = "decimal")]
    Decimal {
        /// Total number of significant digits.
        precision: u8,
        /// Digits after the decimal point.
        scale: i8,
    },
    /// Integer type with specified bit width and sign.
    IntType {
        /// Number of bits (8, 16, 32, 64).
        bit_width: u8,
        /// Whether the type is signed (`true`) or unsigned (`false`).
        is_signed: bool,
    },
}

impl ParquetLogicalType {
    /// Convert from Parquet `ConvertedType` (legacy logical type IDs).
    ///
    /// Returns `None` if the ID is unsupported or maps to a type that is not
    /// handled (e.g. `MAP`, `DECIMAL`, `LIST`).
    pub fn from_converted_type(id: Option<i32>) -> Option<Self> {
        // ConvertedType ids follow parquet.thrift: UTF8=0, MAP=1,
        // MAP_KEY_VALUE=2, LIST=3, ENUM=4, DECIMAL=5, DATE=6,
        // TIME_MILLIS=7, TIME_MICROS=8, TIMESTAMP_MILLIS=9,
        // TIMESTAMP_MICROS=10, UINT_8..UINT_64=11..14, INT_8..INT_64=15..18,
        // JSON=19, BSON=20, INTERVAL=21.
        match id {
            None => None,
            Some(0) => Some(ParquetLogicalType::Utf8),
            Some(1) => None, // MAP (unsupported)
            Some(2) => None, // MAP_KEY_VALUE (unsupported)
            Some(3) => None, // LIST (unsupported)
            Some(4) => None, // ENUM (unsupported)
            #[cfg(feature = "decimal")]
            Some(5) => None, // DECIMAL - precision/scale come from the schema element, not the converted type
            #[cfg(not(feature = "decimal"))]
            Some(5) => None, // DECIMAL (unsupported)
            #[cfg(feature = "datetime")]
            Some(6) => Some(ParquetLogicalType::Date32),
            #[cfg(feature = "datetime")]
            Some(7) => Some(ParquetLogicalType::TimeMillis),
            #[cfg(feature = "datetime")]
            Some(8) => Some(ParquetLogicalType::TimeMicros),
            // The legacy TIMESTAMP converted types are defined as UTC instants.
            #[cfg(feature = "datetime")]
            Some(9) => Some(ParquetLogicalType::TimestampMillis { utc: true }),
            #[cfg(feature = "datetime")]
            Some(10) => Some(ParquetLogicalType::TimestampMicros { utc: true }),
            Some(11) => Some(ParquetLogicalType::IntType {
                bit_width: 8,
                is_signed: false,
            }),
            Some(12) => Some(ParquetLogicalType::IntType {
                bit_width: 16,
                is_signed: false,
            }),
            Some(13) => Some(ParquetLogicalType::IntType {
                bit_width: 32,
                is_signed: false,
            }),
            Some(14) => Some(ParquetLogicalType::IntType {
                bit_width: 64,
                is_signed: false,
            }),
            Some(15) => Some(ParquetLogicalType::IntType {
                bit_width: 8,
                is_signed: true,
            }),
            Some(16) => Some(ParquetLogicalType::IntType {
                bit_width: 16,
                is_signed: true,
            }),
            Some(17) => Some(ParquetLogicalType::IntType {
                bit_width: 32,
                is_signed: true,
            }),
            Some(18) => Some(ParquetLogicalType::IntType {
                bit_width: 64,
                is_signed: true,
            }),
            Some(19) => None, // JSON
            Some(20) => None, // BSON
            Some(21) => None, // INTERVAL (unsupported)
            _ => None,
        }
    }
}

/// Parquet page encoding as per the official Parquet specification.
///
/// We only implement a small subset of these, but they are here for completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParquetEncoding {
    /// Basic uncompressed binary encoding.
    Plain = 0,
    /// Deprecated in favour of RLE_DICTIONARY.
    PlainDictionary = 2,
    /// Run-Length Encoding (used for levels and dictionary indices).
    Rle = 3,
    /// Packs values together into minimal bits (mainly for levels, rarely used directly now).
    BitPacked = 4,
    /// Delta encoding for integer values.
    DeltaBinaryPacked = 5,
    /// Delta encoding for byte array lengths.
    DeltaLengthByteArray = 6,
    /// Delta encoding for prefixes in byte arrays (e.g., sorted strings).
    DeltaByteArray = 7,
    /// Current standard for dictionary encoding.
    RleDictionary = 8,
    /// Can lead to better ratio and speed when a compression algorithm is used afterwards.
    ByteStreamSplit = 9,
}

impl ParquetEncoding {
    /// Convert from Parquet i32 encoding ID.
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Plain),
            2 => Some(Self::PlainDictionary),
            3 => Some(Self::Rle),
            4 => Some(Self::BitPacked),
            5 => Some(Self::DeltaBinaryPacked),
            6 => Some(Self::DeltaLengthByteArray),
            7 => Some(Self::DeltaByteArray),
            8 => Some(Self::RleDictionary),
            9 => Some(Self::ByteStreamSplit),
            _ => None,
        }
    }

    /// Returns the canonical Parquet i32 encoding ID.
    pub fn to_i32(self) -> i32 {
        self as i32
    }
}

/// Authoritative mapping from ArrowType to Parquet physical/logical.
/// Returns IoError for any unsupported/disabled variant.
pub(crate) fn arrow_type_to_parquet(
    ty: &ArrowType,
) -> Result<(ParquetPhysicalType, ParquetLogicalType), IoError> {
    match ty {
        ArrowType::Boolean => Ok((ParquetPhysicalType::Boolean, ParquetLogicalType::NoneType)),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::IntType {
                bit_width: 8,
                is_signed: true,
            },
        )),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::IntType {
                bit_width: 16,
                is_signed: true,
            },
        )),
        ArrowType::Int32 => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::IntType {
                bit_width: 32,
                is_signed: true,
            },
        )),
        ArrowType::Int64 => Ok((
            ParquetPhysicalType::Int64,
            ParquetLogicalType::IntType {
                bit_width: 64,
                is_signed: true,
            },
        )),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::IntType {
                bit_width: 8,
                is_signed: false,
            },
        )),
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::IntType {
                bit_width: 16,
                is_signed: false,
            },
        )),
        ArrowType::UInt32 => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::IntType {
                bit_width: 32,
                is_signed: false,
            },
        )),
        ArrowType::UInt64 => Ok((
            ParquetPhysicalType::Int64,
            ParquetLogicalType::IntType {
                bit_width: 64,
                is_signed: false,
            },
        )),
        // Parquet has no dictionary type. A categorical column is a UTF8
        // string column whose pages the writer dictionary-encodes.
        ArrowType::Dictionary(_) => Ok((ParquetPhysicalType::ByteArray, ParquetLogicalType::Utf8)),
        #[cfg(feature = "decimal")]
        ArrowType::Decimal32(p, s) => Ok((
            ParquetPhysicalType::Int32,
            ParquetLogicalType::Decimal { precision: *p, scale: *s },
        )),
        #[cfg(feature = "decimal")]
        ArrowType::Decimal64(p, s) => Ok((
            ParquetPhysicalType::Int64,
            ParquetLogicalType::Decimal { precision: *p, scale: *s },
        )),
        #[cfg(feature = "decimal")]
        ArrowType::Decimal128(p, s) => Ok((
            ParquetPhysicalType::FixedLenByteArray,
            ParquetLogicalType::Decimal { precision: *p, scale: *s },
        )),
        ArrowType::Float32 => Ok((ParquetPhysicalType::Float, ParquetLogicalType::NoneType)),
        ArrowType::Float64 => Ok((ParquetPhysicalType::Double, ParquetLogicalType::NoneType)),
        ArrowType::String => Ok((ParquetPhysicalType::ByteArray, ParquetLogicalType::Utf8)),
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => Ok((ParquetPhysicalType::ByteArray, ParquetLogicalType::Utf8)),
        ArrowType::Utf8View => Ok((ParquetPhysicalType::ByteArray, ParquetLogicalType::Utf8)),
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Date32) => Ok((ParquetPhysicalType::Int32, ParquetLogicalType::Date32)),
        // Parquet DATE is a 32-bit day count, so Date64 milliseconds are
        // carried into days on write. See `temporal_unit_scale`.
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Date64) => Ok((ParquetPhysicalType::Int32, ParquetLogicalType::Date32)),
        // A timestamp with a timezone is an instant, so it is stored as
        // adjusted to UTC. A timestamp without one is local wall-clock time.
        // Parquet keeps only that flag, not the zone name.
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Timestamp(unit, tz)) => match unit {
            // Parquet has no seconds unit, so seconds are scaled to
            // milliseconds on write.
            TimeUnit::Seconds | TimeUnit::Milliseconds => Ok((
                ParquetPhysicalType::Int64,
                ParquetLogicalType::TimestampMillis { utc: tz.is_some() },
            )),
            TimeUnit::Microseconds => Ok((
                ParquetPhysicalType::Int64,
                ParquetLogicalType::TimestampMicros { utc: tz.is_some() },
            )),
            TimeUnit::Nanoseconds => Ok((
                ParquetPhysicalType::Int64,
                ParquetLogicalType::TimestampNanos { utc: tz.is_some() },
            )),
            // A timestamp counted in days is a date.
            TimeUnit::Days => Ok((ParquetPhysicalType::Int32, ParquetLogicalType::Date32)),
        },
        // Parquet fixes the storage width by unit: TIME(MILLIS) is INT32,
        // TIME(MICROS) and TIME(NANOS) are INT64, whatever width the Arrow
        // column uses. Seconds are scaled to milliseconds on write.
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Time32(unit) | Dt::Time64(unit)) => match unit {
            TimeUnit::Seconds | TimeUnit::Milliseconds => {
                Ok((ParquetPhysicalType::Int32, ParquetLogicalType::TimeMillis))
            }
            TimeUnit::Microseconds => {
                Ok((ParquetPhysicalType::Int64, ParquetLogicalType::TimeMicros))
            }
            TimeUnit::Nanoseconds => {
                Ok((ParquetPhysicalType::Int64, ParquetLogicalType::TimeNanos))
            }
            TimeUnit::Days => Err(IoError::UnsupportedType(
                "time of day counted in days has no Parquet type".into(),
            )),
        },
        ArrowType::Null => Err(IoError::UnsupportedType(
            "Null type is not supported".into(),
        )),
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Duration32(_) | Dt::Duration64(_)) => Err(IoError::UnsupportedType(
            "Duration has no Parquet logical type".into(),
        )),
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Interval(_)) => Err(IoError::UnsupportedType(
            "Interval is not supported".into(),
        )),
    }
}

/// Multiplier and divisor that carry an Arrow temporal value into the unit
/// of its Parquet annotation from [`arrow_type_to_parquet`].
///
/// Most units match and scale by one. Seconds become milliseconds because
/// Parquet has no seconds unit, and Date64 milliseconds become the day
/// count that Parquet DATE stores.
pub(crate) fn temporal_unit_scale(ty: &ArrowType) -> (i64, i64) {
    match ty {
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Date64) => (1, 86_400_000),
        #[cfg(feature = "datetime")]
        ArrowType::DateTime(Dt::Timestamp(TimeUnit::Seconds, _))
        | ArrowType::DateTime(Dt::Time32(TimeUnit::Seconds))
        | ArrowType::DateTime(Dt::Time64(TimeUnit::Seconds)) => (1000, 1),
        _ => (1, 1),
    }
}

/// Parquet -> Arrow mapping
/// Bit width and sign comes from IntType logical type.
pub(crate) fn parquet_to_arrow_type(
    physical: ParquetPhysicalType,
    logical: Option<ParquetLogicalType>,
) -> Result<ArrowType, IoError> {
    match (physical, logical.clone()) {
        (ParquetPhysicalType::Boolean, _) => Ok(ArrowType::Boolean),

        // Signed/unsigned INT32 and INT64 mappings
        #[cfg(feature = "extended_numeric_types")]
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::IntType {
                bit_width: 8,
                is_signed: true,
            }),
        ) => Ok(ArrowType::Int8),
        #[cfg(feature = "extended_numeric_types")]
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::IntType {
                bit_width: 16,
                is_signed: true,
            }),
        ) => Ok(ArrowType::Int16),
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::IntType {
                bit_width: 32,
                is_signed: true,
            }),
        ) => Ok(ArrowType::Int32),
        #[cfg(feature = "extended_numeric_types")]
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::IntType {
                bit_width: 8,
                is_signed: false,
            }),
        ) => Ok(ArrowType::UInt8),
        #[cfg(feature = "extended_numeric_types")]
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::IntType {
                bit_width: 16,
                is_signed: false,
            }),
        ) => Ok(ArrowType::UInt16),
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::IntType {
                bit_width: 32,
                is_signed: false,
            }),
        ) => Ok(ArrowType::UInt32),
        (
            ParquetPhysicalType::Int64,
            Some(ParquetLogicalType::IntType {
                bit_width: 64,
                is_signed: true,
            }),
        ) => Ok(ArrowType::Int64),
        (
            ParquetPhysicalType::Int64,
            Some(ParquetLogicalType::IntType {
                bit_width: 64,
                is_signed: false,
            }),
        ) => Ok(ArrowType::UInt64),

        // Fallback for non-annotated int types
        (ParquetPhysicalType::Int32, None) => Ok(ArrowType::Int32),
        (ParquetPhysicalType::Int64, None) => Ok(ArrowType::Int64),

        // Dates, times, timestamps
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int32, Some(ParquetLogicalType::Date32)) => Ok(ArrowType::DateTime(Dt::Date32)),
        // A UTC-adjusted timestamp reads back with the "UTC" zone. Parquet
        // records no zone name, so that is the only zone a file can carry.
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int64, Some(ParquetLogicalType::TimestampMillis { utc })) => Ok(
            ArrowType::DateTime(Dt::Timestamp(TimeUnit::Milliseconds, utc.then(|| "UTC".to_string()))),
        ),
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int64, Some(ParquetLogicalType::TimestampMicros { utc })) => Ok(
            ArrowType::DateTime(Dt::Timestamp(TimeUnit::Microseconds, utc.then(|| "UTC".to_string()))),
        ),
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int64, Some(ParquetLogicalType::TimestampNanos { utc })) => Ok(
            ArrowType::DateTime(Dt::Timestamp(TimeUnit::Nanoseconds, utc.then(|| "UTC".to_string()))),
        ),
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int32, Some(ParquetLogicalType::TimeMillis)) => {
            Ok(ArrowType::DateTime(Dt::Time32(TimeUnit::Milliseconds)))
        }
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int32, Some(ParquetLogicalType::TimeMicros)) => {
            Ok(ArrowType::DateTime(Dt::Time32(TimeUnit::Microseconds)))
        }
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int32, Some(ParquetLogicalType::TimeNanos)) => {
            Ok(ArrowType::DateTime(Dt::Time32(TimeUnit::Nanoseconds)))
        }
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int64, Some(ParquetLogicalType::TimeMillis)) => {
            Ok(ArrowType::DateTime(Dt::Time64(TimeUnit::Milliseconds)))
        }
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int64, Some(ParquetLogicalType::TimeMicros)) => {
            Ok(ArrowType::DateTime(Dt::Time64(TimeUnit::Microseconds)))
        }
        #[cfg(feature = "datetime")]
        (ParquetPhysicalType::Int64, Some(ParquetLogicalType::TimeNanos)) => {
            Ok(ArrowType::DateTime(Dt::Time64(TimeUnit::Nanoseconds)))
        }

        // Decimals
        #[cfg(feature = "decimal")]
        (
            ParquetPhysicalType::Int32,
            Some(ParquetLogicalType::Decimal { precision, scale }),
        ) => Ok(ArrowType::Decimal32(precision, scale)),
        #[cfg(feature = "decimal")]
        (
            ParquetPhysicalType::Int64,
            Some(ParquetLogicalType::Decimal { precision, scale }),
        ) => Ok(ArrowType::Decimal64(precision, scale)),
        // A fixed-length decimal may hold any precision. The narrowest
        // Arrow decimal that fits the precision is used, matching the
        // INT32 and INT64 forms above.
        #[cfg(feature = "decimal")]
        (
            ParquetPhysicalType::FixedLenByteArray,
            Some(ParquetLogicalType::Decimal { precision, scale }),
        ) => Ok(match precision {
            0..=9 => ArrowType::Decimal32(precision, scale),
            10..=18 => ArrowType::Decimal64(precision, scale),
            _ => ArrowType::Decimal128(precision, scale),
        }),

        // Floats
        (ParquetPhysicalType::Float, _) => Ok(ArrowType::Float32),
        (ParquetPhysicalType::Double, _) => Ok(ArrowType::Float64),

        // Strings - always logical UTF8/Utf8
        (ParquetPhysicalType::ByteArray, Some(ParquetLogicalType::Utf8)) => Ok(ArrowType::String),

        // Fallback -- treat byte array without logical utf8 as unsupported
        (ParquetPhysicalType::ByteArray, None) => {
            Err(IoError::UnsupportedType("Binary not supported".into()))
        }

        _ => Err(IoError::UnsupportedType(format!(
            "Parquet type {:?} + logical {:?} not supported",
            physical, logical
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "decimal")]
    #[test]
    fn decimal32_maps_to_int32_physical() {
        let (phys, logical) = arrow_type_to_parquet(&ArrowType::Decimal32(7, 2)).unwrap();
        assert_eq!(phys, ParquetPhysicalType::Int32);
        assert_eq!(
            logical,
            ParquetLogicalType::Decimal { precision: 7, scale: 2 }
        );
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn decimal64_maps_to_int64_physical() {
        let (phys, logical) = arrow_type_to_parquet(&ArrowType::Decimal64(18, 4)).unwrap();
        assert_eq!(phys, ParquetPhysicalType::Int64);
        assert_eq!(
            logical,
            ParquetLogicalType::Decimal { precision: 18, scale: 4 }
        );
    }

    #[cfg(feature = "decimal")]
    #[test]
    fn decimal128_maps_to_fixed_len_byte_array_physical() {
        let (phys, logical) = arrow_type_to_parquet(&ArrowType::Decimal128(38, 6)).unwrap();
        assert_eq!(phys, ParquetPhysicalType::FixedLenByteArray);
        assert_eq!(
            logical,
            ParquetLogicalType::Decimal { precision: 38, scale: 6 }
        );
    }
}
