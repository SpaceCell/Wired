// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Parquet table reader - *reads into `minarrow::Table`*
//!
//! Reads flat-schema Parquet files written by lightstream or by other
//! writers such as parquet-cpp (pyarrow) into a fully materialised Table.
//!
//! ## Features
//! - DataPageV1 and DataPageV2 page layouts, across any number of row groups
//! - Hybrid RLE/bit-packed definition levels and dictionary indices
//! - `PLAIN`, `RLE_DICTIONARY` and `PLAIN_DICTIONARY` value encodings, plus
//!   `RLE` booleans. Dictionary-encoded columns of any physical type are
//!   expanded on read.
//! - Optional feature-gated Snappy / Zstd compression
//! - Type maps to Arrow/Minarrow - {i32, i64, u32, u64, f32, f64, bool, utf8
//!   dictionary<u32/u64, and date32/date64 via `datetime` feature}
//! - No nested type support
//! - Works with any `Read + Seek`
//! - Reads into memory - no mmap zero-copy like IPC at the present time.
//!
//! ## Types
//! The schema element decides the column type: its physical type plus the
//! `LogicalType` annotation, or the legacy converted type when that is all
//! the writer recorded. Page encodings do not take part, so a
//! dictionary-encoded column reads as the type its schema element names.
//! Categorical columns written by lightstream therefore come back as UTF8
//! string columns. UTF8 columns use 32-bit offsets and widen to
//! `LargeString` only when the column's data outgrows them.
//!
//! ## Outputs
//! On success returns a fully materialised `Table`; otherwise yields an `IOError`
//! for malformed footers/headers, unsupported encodings, or truncated pages.

use std::collections::{BTreeMap, HashSet};
use std::convert::TryInto;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::compression::{Compression, decompress};
use crate::constants::PARQUET_MAGIC;
use crate::error::IoError;
#[cfg(feature = "datetime")]
use crate::models::decoders::parquet::{decode_datetime32_plain, decode_datetime64_plain};
use crate::models::decoders::parquet::{
    decode_dictionary_indices_rle, decode_float32_plain, decode_float64_plain, decode_int32_plain,
    decode_int64_plain, decode_string_plain, decode_uint32_as_int32_plain,
    decode_uint64_as_int64_plain,
};
use crate::models::encoders::parquet::metadata::{
    ColumnChunkMeta, ColumnMetadata, DataPageHeader, DataPageHeaderV2, DictionaryPageHeader,
    FileMetaData, PageHeader, PageType, RowGroupMeta, SchemaElement, Statistics,
};
use crate::models::types::parquet::{
    ParquetEncoding, ParquetLogicalType, ParquetPhysicalType, parquet_to_arrow_type,
};
use minarrow::{
    Array, ArrowType, Bitmask, BooleanArray, CategoricalArray, Field, FieldArray, FloatArray, IntegerArray,
    NumericArray, StringArray, Table, TextArray, Vec64, vec64,
};
#[cfg(feature = "decimal")]
use minarrow::DecimalArray;
#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, TemporalArray, TimeUnit};

/// Build the logical type for a schema element.
///
/// The `LogicalType` annotation wins when the writer recorded one, since it
/// carries units the converted type cannot, such as nanoseconds. Otherwise
/// the converted type applies, with DECIMAL taking its precision and scale
/// from the schema element's own fields.
fn logical_type_from_schema(se: &SchemaElement) -> Option<ParquetLogicalType> {
    if let Some(logical) = &se.logical_type {
        return Some(logical.clone());
    }
    #[cfg(feature = "decimal")]
    if se.converted_type == Some(5) {
        let precision = se.precision.unwrap_or(0) as u8;
        let scale = se.scale.unwrap_or(0) as i8;
        return Some(ParquetLogicalType::Decimal { precision, scale });
    }
    ParquetLogicalType::from_converted_type(se.converted_type)
}

/// Decode a FIXED_LEN_BYTE_ARRAY decimal buffer into unscaled values.
///
/// Each value is `width` bytes of big-endian two's complement, up to 16
/// bytes. Values are sign-extended into `i128` and then narrowed to the
/// decimal width the schema's precision selects.
#[cfg(feature = "decimal")]
fn decode_decimal_fixed_plain<T: TryFrom<i128>>(
    buf: &[u8],
    width: usize,
) -> Result<Vec64<T>, IoError> {
    if width == 0 || width > 16 {
        return Err(IoError::Format(format!(
            "decimal FIXED_LEN_BYTE_ARRAY width {width} is outside 1..=16"
        )));
    }
    if buf.len() % width != 0 {
        return Err(IoError::Format(format!(
            "decimal value section of {} bytes is not a multiple of width {width}",
            buf.len()
        )));
    }
    buf.chunks_exact(width)
        .map(|chunk| {
            let fill = if chunk[0] & 0x80 != 0 { 0xff } else { 0x00 };
            let mut bytes = [fill; 16];
            bytes[16 - width..].copy_from_slice(chunk);
            T::try_from(i128::from_be_bytes(bytes)).map_err(|_| {
                IoError::Format("decimal value exceeds the width its precision allows".into())
            })
        })
        .collect()
}

/// Read an entire in-memory Table from a Parquet v2 file.
///
// TODO: Serial read throughput on the chunked bench is ~390 MiB/s
// while the parallel path scales to ~1.2 GiB/s, so the serial decode
// is the bottleneck. Candidates worth profiling: the hybrid-RLE
// decode loop in `decode_hybrid`, per-page Read-based allocations in
// `read_data_page_v1`/`read_data_page_v2`, and the page-header
// thrift decode.
pub fn load_parquet_table<R: Read + Seek>(r: R) -> Result<Table, IoError> {
    read_parquet_impl(r, None)
}

/// Read only the named columns from a Parquet v2 file.
///
/// Column names must match the schema's `path_in_schema` entries. Returns
/// an error if any name is not found. The returned Table contains only the
/// projected columns, in schema order.
///
/// Because Parquet stores each column at a separate file offset, skipped
/// columns are never read from disk at all.
pub fn load_parquet_table_cols<R: Read + Seek>(
    r: R,
    columns: &[&str],
) -> Result<Table, IoError> {
    let projection: HashSet<String> =
        columns.iter().map(|s| s.to_string()).collect();
    read_parquet_impl(r, Some(projection))
}

/// Shared implementation for full and projected Parquet reads.
///
/// Every row group contributes its column chunks to the same output
/// columns, so a file with several row groups reads as one Table. When
/// `projection` is `Some`, only columns whose names appear in the set are
/// read. Skipped columns never hit disk.
///
/// ## Behaviour
/// - Values arrive in the Parquet layout, where a page's value section
///   holds non-null entries only. Each page is decoded to PLAIN entries and
///   scattered against its definition levels, so [`decode_column`] always
///   receives one entry per row.
/// - Dictionary-encoded pages of any physical type are expanded through
///   the row group's dictionary, so dictionary encoding stays a storage
///   detail of the writer and the column keeps its schema type.
/// - Repeated (nested) columns and unknown compression codecs are
///   reported as errors rather than decoded.
fn read_parquet_impl<R: Read + Seek>(
    mut r: R,
    projection: Option<HashSet<String>>,
) -> Result<Table, IoError> {
    // read the 8-byte footer
    r.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    r.read_exact(&mut tail)?;
    if &tail[4..] != PARQUET_MAGIC {
        return Err(IoError::Format("missing PAR1 footer".into()));
    }
    let footer_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as u64;

    // pull in the FileMetaData block
    r.seek(SeekFrom::End(-8 - footer_len as i64))?;
    let mut footer = vec![0u8; footer_len as usize];
    r.read_exact(&mut footer)?;
    let mut cur = Cursor::new(&footer);
    let meta = parse_file_metadata(&mut cur)?;

    // Leaf schema elements carry a physical type and the root group element
    // does not. Leaves sit in schema order, which is also the order of the
    // column chunks inside every row group.
    let leaves: Vec<(&SchemaElement, ParquetPhysicalType)> = meta
        .schema
        .iter()
        .filter_map(|se| se.type_.map(|ty| (se, ty)))
        .collect();

    // Validate projection names against schema before reading any data
    if let Some(ref proj) = projection {
        for name in proj {
            if !leaves.iter().any(|(se, _)| se.name == *name) {
                return Err(IoError::Format(format!(
                    "column '{}' not found in schema",
                    name
                )));
            }
        }
    }

    if meta.row_groups.is_empty() {
        return Err(IoError::Format("file has no row groups".into()));
    }
    if let Some(rg) = meta
        .row_groups
        .iter()
        .find(|rg| rg.columns.len() != leaves.len())
    {
        return Err(IoError::Format(format!(
            "row group has {} column chunks for {} schema leaves",
            rg.columns.len(),
            leaves.len()
        )));
    }

    let mut columns = Vec::with_capacity(leaves.len());

    for (col_idx, &(leaf, physical)) in leaves.iter().enumerate() {
        // Skip columns not in the projection
        if let Some(ref proj) = projection
            && !proj.contains(&leaf.name)
        {
            continue;
        }

        // OPTIONAL leaves carry one definition level per row. REQUIRED
        // leaves carry none. REPEATED leaves describe nested data.
        let max_def_level: u8 = match leaf.repetition_type {
            0 => 0,
            1 => 1,
            other => {
                return Err(IoError::UnsupportedType(format!(
                    "repeated column '{}' (repetition_type {other})",
                    leaf.name
                )));
            }
        };

        let ty = parquet_to_arrow_type(physical, logical_type_from_schema(leaf))?;
        let layout = ValueLayout::of(physical, leaf.type_length)?;

        let mut def_levels: Vec<bool> = Vec::new();
        let mut values: Vec<u8> = Vec::new();

        for rg in &meta.row_groups {
            let cmeta = &rg.columns[col_idx].meta_data;
            let codec = map_codec(cmeta.codec)?;

            // The row group's dictionary entries without their PLAIN length
            // prefix.
            let mut dict_entries: Vec<Vec<u8>> = Vec::new();
            if let Some(dict_off) = cmeta.dictionary_page_offset {
                r.seek(SeekFrom::Start(dict_off as u64))?;
                let ph = parse_page_header(&mut r)?;
                if ph.type_ != PageType::DictionaryPage {
                    return Err(IoError::Format("expected DICTIONARY_PAGE".into()));
                }
                let mut body = vec![0u8; ph.compressed_page_size as usize];
                r.read_exact(&mut body)?;
                let body = match codec {
                    Some(c) => decompress(&body, c)?,
                    None => body,
                };
                {
                    dict_entries = match layout {
                        ValueLayout::LengthPrefixed => parse_dictionary_values(&body)?,
                        ValueLayout::Fixed(width) => {
                            if body.len() % width != 0 {
                                return Err(IoError::Format(format!(
                                    "dictionary page of {} bytes is not a multiple of the {width}-byte value width",
                                    body.len()
                                )));
                            }
                            body.chunks_exact(width).map(<[u8]>::to_vec).collect()
                        }
                    };
                }
            }

            // Walk the chunk's data pages. num_values counts every row of
            // the chunk, nulls included, which is the definition level
            // count.
            r.seek(SeekFrom::Start(cmeta.data_page_offset as u64))?;
            let chunk_levels = cmeta.num_values as usize;
            let mut levels_read = 0usize;

            while levels_read < chunk_levels {
                let ph = parse_page_header(&mut r)?;
                let (page_defs, encoding, page_values) = match ph.type_ {
                    PageType::DataPage => read_data_page_v1(&mut r, &ph, codec, max_def_level)?,
                    PageType::DataPageV2 => {
                        read_data_page_v2(&mut r, &ph, codec, max_def_level)?
                    }
                    t => return Err(IoError::Format(format!("unsupported page type {:?}", t))),
                };
                let n_valid = page_defs.iter().filter(|&&v| v).count();

                // PLAIN entries for the page's non-null values. Boolean pages
                // unpack to one byte per value here so every physical type
                // scatters through the same fixed or length-prefixed layout.
                let plain: Vec<u8> = match (encoding, physical) {
                    (ParquetEncoding::Plain, ParquetPhysicalType::Boolean) => {
                        if page_values.len() * 8 < n_valid {
                            return Err(IoError::Format("truncated boolean page".into()));
                        }
                        (0..n_valid)
                            .map(|i| (page_values[i / 8] >> (i % 8)) & 1)
                            .collect()
                    }
                    (ParquetEncoding::Rle, ParquetPhysicalType::Boolean) => {
                        // RLE booleans carry a 4-byte run length ahead of
                        // the hybrid run, like V1 levels.
                        let run = length_prefixed_run(&page_values)?;
                        decode_hybrid(run, 1, n_valid)?
                            .iter()
                            .map(|&v| v as u8)
                            .collect()
                    }
                    (ParquetEncoding::Plain, _) => page_values,
                    (ParquetEncoding::RleDictionary | ParquetEncoding::PlainDictionary, _) => {
                        let indices = if n_valid == 0 {
                            Vec64::new()
                        } else {
                            decode_dictionary_indices_rle(&page_values, n_valid)?
                        };
                        let mut plain = Vec::new();
                        for &idx in indices.iter() {
                            let entry = dict_entries.get(idx as usize).ok_or_else(|| {
                                IoError::Format(format!("dictionary index {idx} out of range"))
                            })?;
                            if layout == ValueLayout::LengthPrefixed {
                                plain.extend_from_slice(&(entry.len() as u32).to_le_bytes());
                            }
                            plain.extend_from_slice(entry);
                        }
                        plain
                    }
                    (enc, _) => {
                        return Err(IoError::UnsupportedEncoding(format!(
                            "{enc:?} for column '{}'",
                            leaf.name
                        )));
                    }
                };

                scatter_by_definition_levels(&mut values, &plain, &page_defs, layout)?;
                levels_read += page_defs.len();
                def_levels.extend(page_defs);
            }
        }

        if def_levels.len() != meta.num_rows as usize {
            return Err(IoError::Format(format!(
                "column '{}' holds {} rows, file metadata says {}",
                leaf.name,
                def_levels.len(),
                meta.num_rows
            )));
        }

        // UTF8 columns use 32-bit offsets unless the data outgrows them.
        #[cfg(feature = "large_string")]
        let ty = if ty == ArrowType::String && values.len() > u32::MAX as usize {
            ArrowType::LargeString
        } else {
            ty
        };
        #[cfg(not(feature = "large_string"))]
        if ty == ArrowType::String && values.len() > u32::MAX as usize {
            return Err(IoError::UnsupportedType(format!(
                "column '{}' holds more than 4 GiB of string data, which needs the large_string feature",
                leaf.name
            )));
        }

        let null_count = def_levels.iter().filter(|&&b| !b).count();
        let array = decode_column(&ty, physical, layout, &values, def_levels.len(), def_levels)?;

        columns.push(FieldArray {
            field: Field {
                name: leaf.name.clone(),
                dtype: ty,
                nullable: max_def_level > 0,
                metadata: Default::default(),
            }
            .into(),
            array,
            null_count,
        });
    }

    Ok(Table {
        cols: columns,
        n_rows: meta.num_rows as usize,
        name: String::new(),
        ..Default::default()
    })
}

/// Error for a DECIMAL column stored in a physical type the reader does
/// not decode, which is BYTE_ARRAY.
#[cfg(feature = "decimal")]
fn decimal_storage_error(physical: ParquetPhysicalType) -> IoError {
    IoError::UnsupportedType(format!("DECIMAL stored as {physical:?}"))
}

/// Byte layout of one value inside a PLAIN value section.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ValueLayout {
    /// Every value occupies the given number of bytes.
    Fixed(usize),
    /// A 4-byte little-endian length precedes each value's bytes.
    LengthPrefixed,
}

impl ValueLayout {
    /// Layout of the physical type once a page has been decoded to PLAIN
    /// entries. Booleans count as one byte per value because the page
    /// readers unpack them before scattering. `type_length` is the schema
    /// element's width for FIXED_LEN_BYTE_ARRAY leaves.
    fn of(physical: ParquetPhysicalType, type_length: Option<i32>) -> Result<Self, IoError> {
        Ok(match physical {
            ParquetPhysicalType::Boolean => ValueLayout::Fixed(1),
            ParquetPhysicalType::Int32 | ParquetPhysicalType::Float => ValueLayout::Fixed(4),
            ParquetPhysicalType::Int64 | ParquetPhysicalType::Double => ValueLayout::Fixed(8),
            ParquetPhysicalType::ByteArray => ValueLayout::LengthPrefixed,
            ParquetPhysicalType::FixedLenByteArray => match type_length {
                Some(width) if width > 0 => ValueLayout::Fixed(width as usize),
                _ => {
                    return Err(IoError::Format(
                        "FIXED_LEN_BYTE_ARRAY leaf without a type_length".into(),
                    ));
                }
            },
        })
    }
}

/// Append a page's PLAIN entries to `out` with a zero entry at every null
/// row, so the column buffer holds one entry per row.
///
/// Parquet value sections hold non-null values only, while the PLAIN
/// decoders behind [`decode_column`] expect one entry per row with the null
/// mask applied afterwards. A null row becomes zero bytes for a fixed-width
/// value or a zero length for a length-prefixed value.
fn scatter_by_definition_levels(
    out: &mut Vec<u8>,
    plain: &[u8],
    def_levels: &[bool],
    layout: ValueLayout,
) -> Result<(), IoError> {
    if def_levels.iter().all(|&v| v) {
        out.extend_from_slice(plain);
        return Ok(());
    }
    let mut pos = 0usize;
    for &valid in def_levels {
        if !valid {
            match layout {
                ValueLayout::Fixed(width) => out.extend(std::iter::repeat_n(0u8, width)),
                ValueLayout::LengthPrefixed => out.extend_from_slice(&[0u8; 4]),
            }
            continue;
        }
        let width = match layout {
            ValueLayout::Fixed(width) => width,
            ValueLayout::LengthPrefixed => {
                let len = plain
                    .get(pos..pos + 4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                    .ok_or_else(|| {
                        IoError::Format("value section shorter than its definition levels".into())
                    })?;
                4 + len
            }
        };
        let entry = plain.get(pos..pos + width).ok_or_else(|| {
            IoError::Format("value section shorter than its definition levels".into())
        })?;
        out.extend_from_slice(entry);
        pos += width;
    }
    Ok(())
}

/// Split a 4-byte little-endian length off the front of a level or RLE
/// boolean stream and return the run it introduces.
fn length_prefixed_run(buf: &[u8]) -> Result<&[u8], IoError> {
    let len = buf
        .get(..4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .ok_or_else(|| IoError::Format("truncated length-prefixed run".into()))?;
    buf.get(4..4 + len)
        .ok_or_else(|| IoError::Format("truncated length-prefixed run".into()))
}

/// DataPageV1 reader.
///
/// The whole page body is compressed as one unit. Inside it, repetition
/// levels (absent for a flat schema) and definition levels (absent for a
/// REQUIRED column) each lead with a 4-byte length, and the values follow.
/// Returns the definition levels, the value encoding and the raw values.
fn read_data_page_v1<R: Read>(
    r: &mut R,
    ph: &PageHeader,
    codec: Option<Compression>,
    max_def_level: u8,
) -> Result<(Vec<bool>, ParquetEncoding, Vec<u8>), IoError> {
    let h = ph
        .data_page_header
        .as_ref()
        .ok_or_else(|| IoError::Format("missing DataPageHeader".into()))?;

    let mut body = vec![0u8; ph.compressed_page_size as usize];
    r.read_exact(&mut body)?;
    let body = match codec {
        Some(c) => decompress(&body, c)?,
        None => body,
    };

    let n = h.num_values as usize;
    let mut pos = 0usize;
    let def_levels = if max_def_level == 0 {
        vec![true; n]
    } else {
        if h.definition_level_encoding != ParquetEncoding::Rle {
            return Err(IoError::UnsupportedEncoding(format!(
                "{:?} definition levels",
                h.definition_level_encoding
            )));
        }
        let run = length_prefixed_run(&body[pos..])?;
        pos += 4 + run.len();
        decode_hybrid(run, max_def_level, n)?
            .into_iter()
            .map(|v| v == max_def_level as u32)
            .collect()
    };

    Ok((def_levels, h.encoding, body[pos..].to_vec()))
}

/// DataPageV2 reader.
///
/// Repetition and definition levels sit uncompressed at the front of the
/// page with their byte lengths in the header. The value section that
/// follows is compressed only when the header says so. Returns the
/// definition levels, the value encoding and the raw values.
fn read_data_page_v2<R: Read>(
    r: &mut R,
    ph: &PageHeader,
    codec: Option<Compression>,
    max_def_level: u8,
) -> Result<(Vec<bool>, ParquetEncoding, Vec<u8>), IoError> {
    let h = ph
        .data_page_header_v2
        .as_ref()
        .ok_or_else(|| IoError::Format("missing DataPageHeaderV2".into()))?;

    // 1) consume repetition-levels bytes. A flat schema has no repetition
    //    levels, though a writer may still emit a run for them.
    let mut rep = vec![0u8; h.repetition_levels_byte_length as usize];
    r.read_exact(&mut rep)?;

    // 2) consume definition-levels bytes
    let mut def = vec![0u8; h.definition_levels_byte_length as usize];
    r.read_exact(&mut def)?;

    // 3) the remainder of this page is the `compressed_page_size - R - D`
    let body_len = (ph.compressed_page_size as usize)
        .checked_sub(rep.len() + def.len())
        .ok_or_else(|| IoError::Format("bad compressed_page_size".into()))?;
    let mut vs = vec![0u8; body_len];
    r.read_exact(&mut vs)?;

    // 4) decompress if needed
    let values_raw = match (h.is_compressed, codec) {
        (true, Some(c)) => decompress(&vs, c)?,
        _ => vs,
    };

    // 5) decode definition levels. num_values counts every row of the
    //    page, nulls included, which is the definition level count.
    let n = h.num_values as usize;
    let def_levels = if max_def_level == 0 || def.is_empty() {
        vec![true; n]
    } else {
        decode_hybrid(&def, max_def_level, n)?
            .into_iter()
            .map(|v| v == max_def_level as u32)
            .collect()
    };

    Ok((def_levels, h.encoding, values_raw))
}

// Column-value decoder

/// Build the column array from PLAIN entries, one per row.
///
/// `buf` holds one entry per row in the layout the PLAIN decoders expect,
/// with a zero entry under each null. `physical` and `layout` describe the
/// entries for types with more than one storage form, which for now means
/// decimals stored as INT32, INT64 or FIXED_LEN_BYTE_ARRAY.
#[cfg_attr(not(feature = "decimal"), allow(unused_variables))]
fn decode_column(
    ty: &ArrowType,
    physical: ParquetPhysicalType,
    layout: ValueLayout,
    buf: &[u8],
    len: usize,
    def_levels: Vec<bool>,
) -> Result<Array, IoError> {
    let mask = Some(Bitmask::from_bools(&def_levels));

    Ok(match ty {
        // numerics
        ArrowType::Int32 => Array::NumericArray(NumericArray::Int32(Arc::new(
            IntegerArray::from_vec64(decode_int32_plain(buf)?, mask),
        ))),
        ArrowType::UInt32 => Array::NumericArray(NumericArray::UInt32(Arc::new(
            IntegerArray::from_vec64(decode_uint32_as_int32_plain(buf)?, mask),
        ))),
        ArrowType::Int64 => Array::NumericArray(NumericArray::Int64(Arc::new(
            IntegerArray::from_vec64(decode_int64_plain(buf)?, mask),
        ))),
        ArrowType::UInt64 => Array::NumericArray(NumericArray::UInt64(Arc::new(
            IntegerArray::from_vec64(decode_uint64_as_int64_plain(buf)?, mask),
        ))),
        ArrowType::Float32 => Array::NumericArray(NumericArray::Float32(Arc::new(
            FloatArray::from_vec64(decode_float32_plain(buf)?, mask),
        ))),
        ArrowType::Float64 => Array::NumericArray(NumericArray::Float64(Arc::new(
            FloatArray::from_vec64(decode_float64_plain(buf)?, mask),
        ))),

        // booleans, one byte per row
        ArrowType::Boolean => {
            let bits: Vec<bool> = buf.iter().map(|&b| b != 0).collect();
            Array::BooleanArray(Arc::new(BooleanArray::new(Bitmask::from_bools(&bits), mask)))
        }

        // strings
        ArrowType::String => {
            let (offsets, data) = decode_string_plain(buf, len)?;
            Array::TextArray(TextArray::String32(Arc::new(StringArray {
                offsets: offsets.into(),
                data: data.into(),
                null_mask: mask,
            })))
        }
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => {
            use crate::models::decoders::parquet::decode_large_string_plain;

            let (offsets, data) = decode_large_string_plain(buf, len)?;
            Array::TextArray(TextArray::String64(Arc::new(StringArray {
                offsets: offsets.into(),
                data: data.into(),
                null_mask: mask,
            })))
        }

        // temporal. DATE and TIME(MILLIS) are INT32, every other unit INT64.
        #[cfg(feature = "datetime")]
        ArrowType::Date32 => Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
            DatetimeArray {
                data: decode_datetime32_plain(buf)?.into(),
                null_mask: mask,
                time_unit: TimeUnit::Days,
            },
        ))),
        #[cfg(feature = "datetime")]
        ArrowType::Time32(unit) => Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
            DatetimeArray {
                data: decode_datetime32_plain(buf)?.into(),
                null_mask: mask,
                time_unit: unit.clone(),
            },
        ))),
        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(unit, _) | ArrowType::Time64(unit) => {
            Array::TemporalArray(TemporalArray::Datetime64(Arc::new(DatetimeArray {
                data: decode_datetime64_plain(buf)?.into(),
                null_mask: mask,
                time_unit: unit.clone(),
            })))
        }

        // decimals
        #[cfg(feature = "decimal")]
        ArrowType::Decimal32(precision, scale) => {
            let data = match (physical, layout) {
                (ParquetPhysicalType::Int32, _) => decode_int32_plain(buf)?,
                (ParquetPhysicalType::FixedLenByteArray, ValueLayout::Fixed(width)) => {
                    decode_decimal_fixed_plain::<i32>(buf, width)?
                }
                _ => return Err(decimal_storage_error(physical)),
            };
            Array::NumericArray(NumericArray::Decimal32(Arc::new(DecimalArray {
                data: data.into(),
                null_mask: mask,
                precision: *precision,
                scale: *scale,
            })))
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal64(precision, scale) => {
            let data = match (physical, layout) {
                (ParquetPhysicalType::Int64, _) => decode_int64_plain(buf)?,
                (ParquetPhysicalType::FixedLenByteArray, ValueLayout::Fixed(width)) => {
                    decode_decimal_fixed_plain::<i64>(buf, width)?
                }
                _ => return Err(decimal_storage_error(physical)),
            };
            Array::NumericArray(NumericArray::Decimal64(Arc::new(DecimalArray {
                data: data.into(),
                null_mask: mask,
                precision: *precision,
                scale: *scale,
            })))
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal128(precision, scale) => {
            let data = match (physical, layout) {
                (ParquetPhysicalType::FixedLenByteArray, ValueLayout::Fixed(width)) => {
                    decode_decimal_fixed_plain::<i128>(buf, width)?
                }
                _ => return Err(decimal_storage_error(physical)),
            };
            Array::NumericArray(NumericArray::Decimal128(Arc::new(DecimalArray {
                data: data.into(),
                null_mask: mask,
                precision: *precision,
                scale: *scale,
            })))
        }

        _ => {
            return Err(IoError::UnsupportedType(format!("decode {:?}", ty)));
        }
    })
}

// categorical builders

fn build_cat32(idx: Vec64<u32>, dict_raw: &[Vec<u8>], mask: Option<Bitmask>) -> Array {
    let dict = dict_raw
        .iter()
        .map(|b| String::from_utf8(b.clone()).unwrap())
        .collect::<Vec64<_>>()
        .into();
    Array::TextArray(TextArray::Categorical32(Arc::new(CategoricalArray::new(
        idx,
        dict,
        mask,
    ))))
}

#[cfg(feature = "default_categorical_8")]
fn build_cat8(idx: Vec64<u32>, dict_raw: &[Vec<u8>], mask: Option<Bitmask>) -> Array {
    let dict = dict_raw
        .iter()
        .map(|b| String::from_utf8(b.clone()).unwrap())
        .collect::<Vec64<_>>();
    let idx8: Vec64<u8> = idx.iter().map(|&v| v as u8).collect();
    Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray::new(
        idx8,
        dict,
        mask,
    ))))
}

#[cfg(all(feature = "extended_categorical", feature = "large_string"))]
fn build_cat64(idx: Vec64<u64>, dict_raw: &[Vec<u8>], mask: Option<Bitmask>) -> Array {
    let dict = dict_raw
        .iter()
        .map(|b| String::from_utf8(b.clone()).unwrap())
        .collect::<Vec64<_>>()
        .into();
    Array::TextArray(TextArray::Categorical64(Arc::new(CategoricalArray::new(
        idx,
        dict,
        mask,
    ))))
}

// RLE/bit-packed Hybrid decoder

fn decode_hybrid(buf: &[u8], bit_width: u8, n: usize) -> Result<Vec64<u32>, IoError> {
    if bit_width == 0 {
        return Ok(vec64![0; n]);
    }
    let mut out = Vec64::with_capacity(n);
    let mut pos = 0usize;
    while out.len() < n {
        let (header, used) = read_uleb128(&buf[pos..])?;
        pos += used;
        if header & 1 == 0 {
            // RLE
            let run_len = (header >> 1) as usize;
            let bytes_per_value = bit_width.div_ceil(8) as usize;
            if pos + bytes_per_value > buf.len() {
                return Err(IoError::Format("truncated RLE run".into()));
            }
            let mut v_bytes = [0u8; 4];
            v_bytes[..bytes_per_value].copy_from_slice(&buf[pos..pos + bytes_per_value]);
            let v = u32::from_le_bytes(v_bytes);
            pos += bytes_per_value;
            let take = run_len.min(n - out.len());
            out.extend(std::iter::repeat_n(v, take));
        } else {
            // bit-packed
            let groups = (header >> 1) as usize; // 1 group = 8 values
            let total_values = groups * 8;
            let total_bits = total_values * (bit_width as usize);
            let total_bytes = total_bits.div_ceil(8);
            if pos + total_bytes > buf.len() {
                return Err(IoError::Format("truncated bit-packed run".into()));
            }
            let slice = &buf[pos..pos + total_bytes];
            // Unpack the run LSB-first with values contiguous inside each
            // 8-value group, per the Parquet hybrid RLE/bit-packing layout.
            let mut scratch = vec![0u32; total_values];
            for (idx, slot) in scratch.iter_mut().enumerate() {
                let bit_base = idx * bit_width as usize;
                for k in 0..bit_width as usize {
                    let bit = bit_base + k;
                    if (slice[bit / 8] >> (bit % 8)) & 1 != 0 {
                        *slot |= 1 << k;
                    }
                }
            }
            // push only as many as we actually need, dropping the padding
            let needed = n - out.len();
            out.extend(scratch.into_iter().take(needed));
            pos += total_bytes;
        }
    }
    Ok(out)
}

fn read_uleb128(buf: &[u8]) -> Result<(u64, usize), IoError> {
    let mut val = 0u64;
    let mut shift = 0u32;
    for (i, &b) in buf.iter().enumerate() {
        val |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((val, i + 1));
        }
        shift += 7;
        if shift > 63 {
            break;
        }
    }
    Err(IoError::Format("ULEB128 overflow/truncate".into()))
}

// Misc helpers

/// Resolve the column chunk's codec id. Uncompressed is `None`. Codecs
/// outside the enabled compression features are an error, since their
/// pages cannot be decoded.
fn map_codec(id: i32) -> Result<Option<Compression>, IoError> {
    match id {
        0 => Ok(None),
        #[cfg(feature = "snappy")]
        1 => Ok(Some(Compression::Snappy)),
        #[cfg(feature = "zstd")]
        6 => Ok(Some(Compression::Zstd)), // spec: ZSTD = 6
        other => Err(IoError::Compression(format!(
            "unsupported parquet compression codec {other}"
        ))),
    }
}

/// Split a PLAIN BYTE_ARRAY dictionary page body into its entries.
fn parse_dictionary_values(buf: &[u8]) -> Result<Vec<Vec<u8>>, IoError> {
    let mut c = Cursor::new(buf);
    let mut out = Vec::new();
    while (c.position() as usize) < buf.len() {
        let mut l4 = [0u8; 4];
        c.read_exact(&mut l4)?;
        let len = u32::from_le_bytes(l4) as usize;
        let mut s = vec![0u8; len];
        c.read_exact(&mut s)?;
        out.push(s);
    }
    Ok(out)
}

// Thrift Parsers

fn parse_file_metadata<R: Read>(r: &mut R) -> Result<FileMetaData, IoError> {
    let mut last = 0i16;
    let mut version = None;
    let mut schema = Vec::new();
    let mut num_rows = None;
    let mut row_groups = Vec::new();
    let mut kv_meta = None;
    let mut created_by = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            1 => version = Some(thrift_read_i32(r)?),
            2 => {
                let (_elem_tpe, len) = thrift_read_list_begin(r)?;
                for _ in 0..len {
                    schema.push(parse_schema_element(r)?);
                }
            }
            3 => num_rows = Some(thrift_read_i64(r)?),
            4 => {
                let (_elem_tpe, len) = thrift_read_list_begin(r)?;
                for _ in 0..len {
                    row_groups.push(parse_row_group(r)?);
                }
            }
            5 => {
                // key_value_metadata is a list<KeyValue>, where each KeyValue
                // is a struct with 1=key and 2=value (value optional).
                let (_elem, len) = thrift_read_list_begin(r)?;
                let mut map = BTreeMap::new();
                for _ in 0..len {
                    let mut kv_last = 0i16;
                    let mut key = String::new();
                    let mut value = String::new();
                    loop {
                        let (t, fid) = thrift_read_field_begin(r, &mut kv_last)?;
                        if t == 0 {
                            break;
                        }
                        match fid {
                            1 => key = thrift_read_string(r)?,
                            2 => value = thrift_read_string(r)?,
                            _ => thrift_skip_field(r, t)?,
                        }
                    }
                    map.insert(key, value);
                }
                kv_meta = Some(map);
            }
            6 => created_by = Some(thrift_read_string(r)?),
            _ => thrift_skip_field(r, tpe)?,
        }
    }

    Ok(FileMetaData {
        version: version.ok_or_else(|| IoError::Format("Missing version".into()))?,
        schema,
        num_rows: num_rows.ok_or_else(|| IoError::Format("Missing num_rows".into()))?,
        row_groups,
        key_value_metadata: kv_meta,
        created_by,
    })
}

fn parse_schema_element<R: Read>(r: &mut R) -> Result<SchemaElement, IoError> {
    let mut last = 0i16;
    let mut name = None;
    let mut repetition_type = None;
    let mut type_ = None;
    let mut converted_type = None;
    let mut type_length = None;
    let mut precision = None;
    let mut scale = None;
    let mut field_id = None;
    let mut num_children = None;
    let mut logical_type = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            1 => {
                let v = thrift_read_i32(r)?;
                type_ = Some(ParquetPhysicalType::from_i32(v).ok_or_else(|| {
                    if v == 3 {
                        IoError::UnsupportedType("INT96 columns are not supported".into())
                    } else {
                        IoError::Format(format!("invalid physical type {v}"))
                    }
                })?)
            }
            2 => type_length = Some(thrift_read_i32(r)?),
            3 => repetition_type = Some(thrift_read_i32(r)?),
            4 => name = Some(thrift_read_string(r)?),
            5 => num_children = Some(thrift_read_i32(r)?),
            6 => converted_type = Some(thrift_read_i32(r)?),
            7 => scale = Some(thrift_read_i32(r)?),
            8 => precision = Some(thrift_read_i32(r)?),
            9 => field_id = Some(thrift_read_i32(r)?),
            10 => logical_type = parse_logical_type(r)?,
            _ => thrift_skip_field(r, tpe)?,
        }
    }

    Ok(SchemaElement {
        name: name.ok_or_else(|| IoError::Format("SchemaElement missing name".into()))?,
        repetition_type: repetition_type.unwrap_or(0),
        type_,
        converted_type,
        logical_type,
        type_length,
        precision,
        scale,
        field_id,
        num_children,
    })
}

/// Parse the `LogicalType` union of a schema element.
///
/// The union holds one struct field whose id names the annotation: STRING
/// 1, DECIMAL 5, DATE 6, TIME 7, TIMESTAMP 8, INTEGER 10. TIME and
/// TIMESTAMP carry `isAdjustedToUTC` and a `TimeUnit` union of MILLIS 1,
/// MICROS 2 and NANOS 3. Annotations outside the supported set, such as
/// MAP, LIST, JSON and UUID, return `None` and the column falls back to
/// its converted type.
fn parse_logical_type<R: Read>(r: &mut R) -> Result<Option<ParquetLogicalType>, IoError> {
    let mut last = 0i16;
    let mut logical = None;
    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        if tpe != TC_STRUCT {
            thrift_skip_field(r, tpe)?;
            continue;
        }
        logical = match id {
            1 => {
                thrift_skip_value(r, TC_STRUCT)?;
                Some(ParquetLogicalType::Utf8)
            }
            #[cfg(feature = "decimal")]
            5 => {
                let mut inner = 0i16;
                let mut scale = 0i32;
                let mut precision = 0i32;
                loop {
                    let (t, f) = thrift_read_field_begin(r, &mut inner)?;
                    if t == 0 {
                        break;
                    }
                    match f {
                        1 => scale = thrift_read_i32(r)?,
                        2 => precision = thrift_read_i32(r)?,
                        _ => thrift_skip_field(r, t)?,
                    }
                }
                Some(ParquetLogicalType::Decimal {
                    precision: precision as u8,
                    scale: scale as i8,
                })
            }
            #[cfg(feature = "datetime")]
            6 => {
                thrift_skip_value(r, TC_STRUCT)?;
                Some(ParquetLogicalType::Date32)
            }
            #[cfg(feature = "datetime")]
            7 | 8 => {
                let mut inner = 0i16;
                let mut unit = 0i16;
                let mut utc = false;
                loop {
                    let (t, f) = thrift_read_field_begin(r, &mut inner)?;
                    if t == 0 {
                        break;
                    }
                    match f {
                        1 if t == TC_BOOL_TRUE || t == TC_BOOL_FALSE => utc = t == TC_BOOL_TRUE,
                        2 if t == TC_STRUCT => {
                            let mut unit_last = 0i16;
                            loop {
                                let (ut, uf) = thrift_read_field_begin(r, &mut unit_last)?;
                                if ut == 0 {
                                    break;
                                }
                                unit = uf;
                                thrift_skip_field(r, ut)?;
                            }
                        }
                        _ => thrift_skip_field(r, t)?,
                    }
                }
                match (id, unit) {
                    (7, 1) => Some(ParquetLogicalType::TimeMillis),
                    (7, 2) => Some(ParquetLogicalType::TimeMicros),
                    (7, 3) => Some(ParquetLogicalType::TimeNanos),
                    (8, 1) => Some(ParquetLogicalType::TimestampMillis { utc }),
                    (8, 2) => Some(ParquetLogicalType::TimestampMicros { utc }),
                    (8, 3) => Some(ParquetLogicalType::TimestampNanos { utc }),
                    _ => None,
                }
            }
            10 => {
                let mut inner = 0i16;
                let mut bit_width = 0u8;
                let mut is_signed = true;
                loop {
                    let (t, f) = thrift_read_field_begin(r, &mut inner)?;
                    if t == 0 {
                        break;
                    }
                    match (f, t) {
                        (1, TC_BYTE) => {
                            let mut b = [0u8; 1];
                            r.read_exact(&mut b)?;
                            bit_width = b[0];
                        }
                        (2, TC_BOOL_TRUE) => is_signed = true,
                        (2, TC_BOOL_FALSE) => is_signed = false,
                        _ => thrift_skip_field(r, t)?,
                    }
                }
                Some(ParquetLogicalType::IntType {
                    bit_width,
                    is_signed,
                })
            }
            _ => {
                thrift_skip_value(r, TC_STRUCT)?;
                None
            }
        };
    }
    Ok(logical)
}

fn parse_row_group<R: Read>(r: &mut R) -> Result<RowGroupMeta, IoError> {
    let mut last = 0i16;
    let mut columns = Vec::new();
    let mut total_byte_size = None;
    let mut num_rows = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            1 => {
                let (_elem_tpe, len) = thrift_read_list_begin(r)?;
                for _ in 0..len {
                    columns.push(parse_column_chunk(r)?);
                }
            }
            2 => total_byte_size = Some(thrift_read_i64(r)?),
            3 => num_rows = Some(thrift_read_i64(r)?),
            _ => thrift_skip_field(r, tpe)?,
        }
    }
    Ok(RowGroupMeta {
        columns,
        total_byte_size: total_byte_size.unwrap_or(0),
        num_rows: num_rows.unwrap_or(0),
    })
}

fn parse_column_chunk<R: Read>(r: &mut R) -> Result<ColumnChunkMeta, IoError> {
    let mut last = 0i16;
    let mut file_offset = None;
    let mut meta_data = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            2 => file_offset = Some(thrift_read_i64(r)?),
            3 => meta_data = Some(parse_column_meta_data(r)?),
            _ => thrift_skip_field(r, tpe)?,
        }
    }
    Ok(ColumnChunkMeta {
        file_offset: file_offset.unwrap_or(0),
        meta_data: meta_data.ok_or_else(|| IoError::Format("Missing ColumnMetaData".into()))?,
    })
}

fn parse_column_meta_data<R: Read>(r: &mut R) -> Result<ColumnMetadata, IoError> {
    let mut type_ = None;
    let mut encodings = Vec::new();
    let mut path_in_schema = Vec::new();
    let mut codec = None;
    let mut num_values = None;
    let mut total_uncompressed_size = None;
    let mut total_compressed_size = None;
    let mut data_page_offset = None;
    let mut dictionary_page_offset = None;
    let mut statistics = None;
    let mut last = 0i16;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            1 => {
                let v = thrift_read_i32(r)?;
                type_ = Some(
                    ParquetPhysicalType::from_i32(v)
                        .ok_or_else(|| IoError::Format("Invalid physical type".into()))?,
                );
            }
            2 => {
                let (_elem_tpe, len) = thrift_read_list_begin(r)?;
                for _ in 0..len {
                    let v = thrift_read_i32(r)?;
                    encodings.push(
                        ParquetEncoding::from_i32(v)
                            .ok_or_else(|| IoError::Format("Invalid encoding".into()))?,
                    );
                }
            }
            3 => {
                let (_elem_tpe, len) = thrift_read_list_begin(r)?;
                for _ in 0..len {
                    path_in_schema.push(thrift_read_string(r)?);
                }
            }
            4 => codec = Some(thrift_read_i32(r)?),
            5 => num_values = Some(thrift_read_i64(r)?),
            6 => total_uncompressed_size = Some(thrift_read_i64(r)?),
            7 => total_compressed_size = Some(thrift_read_i64(r)?),
            9 => data_page_offset = Some(thrift_read_i64(r)?),
            11 => dictionary_page_offset = Some(thrift_read_i64(r)?),
            12 => statistics = Some(parse_statistics(r)?),
            _ => thrift_skip_field(r, tpe)?,
        }
    }

    Ok(ColumnMetadata {
        type_: type_.unwrap(),
        encodings,
        path_in_schema,
        codec: codec.unwrap_or(0),
        num_values: num_values.unwrap_or(0),
        total_uncompressed_size: total_uncompressed_size.unwrap_or(0),
        total_compressed_size: total_compressed_size.unwrap_or(0),
        data_page_offset: data_page_offset.unwrap_or(0),
        dictionary_page_offset,
        statistics,
    })
}

fn parse_statistics<R: Read>(r: &mut R) -> Result<Statistics, IoError> {
    let mut last = 0i16;
    let mut null_count = None;
    let mut distinct_count = None;
    let mut min = None;
    let mut max = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        // Statistics carries both the legacy signed min/max (fields 1/2) and
        // the modern unsigned-order max_value/min_value (fields 5/6). Prefer
        // whichever the writer emitted; the modern fields win when present.
        match id {
            1 => max = Some(thrift_read_bytes(r)?),
            2 => min = Some(thrift_read_bytes(r)?),
            3 => null_count = Some(thrift_read_i64(r)?),
            4 => distinct_count = Some(thrift_read_i64(r)?),
            5 => max = Some(thrift_read_bytes(r)?),
            6 => min = Some(thrift_read_bytes(r)?),
            _ => thrift_skip_field(r, tpe)?,
        }
    }
    Ok(Statistics {
        null_count,
        distinct_count,
        min,
        max,
    })
}

fn parse_page_header<R: Read + Seek>(r: &mut R) -> Result<PageHeader, IoError> {
    let mut last = 0i16;
    let mut ptype = None;
    let mut uncomp = None;
    let mut compr = None;
    let mut data_ph = None;
    let mut data_ph_v2 = None;
    let mut dict_ph = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            1 => {
                ptype = Some(
                    PageType::from_i32(thrift_read_i32(r)?)
                        .ok_or_else(|| IoError::Format("Invalid PageType".into()))?,
                )
            }
            2 => uncomp = Some(thrift_read_i32(r)?),
            3 => compr = Some(thrift_read_i32(r)?),
            5 => {
                // DataPageHeader (v1), a nested compact struct at field id 5.
                data_ph = Some(parse_data_page_header(r)?);
            }
            7 => {
                // DictionaryPageHeader, a nested compact struct at field id 7.
                // Read its fields with a struct-local last-field-id starting
                // at zero, stopping on the field-stop byte and skipping any
                // unrecognised field.
                let mut inner = 0i16;
                let mut num_values = 0i32;
                let mut encoding = ParquetEncoding::Plain;
                let mut is_sorted = None;
                loop {
                    let (tpe2, id2) = thrift_read_field_begin(r, &mut inner)?;
                    if tpe2 == 0 {
                        break;
                    }
                    match id2 {
                        1 => num_values = thrift_read_i32(r)?,
                        2 => {
                            encoding =
                                ParquetEncoding::from_i32(thrift_read_i32(r)?).ok_or_else(|| {
                                    IoError::Format(
                                        "Invalid encoding in DictionaryPageHeader".into(),
                                    )
                                })?
                        }
                        3 => is_sorted = Some(tpe2 == TC_BOOL_TRUE),
                        _ => thrift_skip_field(r, tpe2)?,
                    }
                }
                dict_ph = Some(DictionaryPageHeader {
                    num_values,
                    encoding,
                    is_sorted,
                });
            }
            8 => {
                // DataPageHeaderV2, a nested compact struct at field id 8.
                let mut inner = 0i16;
                let mut num_rows = None;
                let mut num_nulls = None;
                let mut num_values = None;
                let mut encoding = None;
                let mut def_len = None;
                let mut rep_len = None;
                let mut is_compressed = None;
                let mut statistics = None;

                loop {
                    let (tpe2, id2) = thrift_read_field_begin(r, &mut inner)?;
                    if tpe2 == 0 {
                        break;
                    }
                    match id2 {
                        1 => num_values = Some(thrift_read_i32(r)?),
                        2 => num_nulls = Some(thrift_read_i32(r)?),
                        3 => num_rows = Some(thrift_read_i32(r)?),
                        4 => {
                            encoding =
                                Some(ParquetEncoding::from_i32(thrift_read_i32(r)?).ok_or_else(
                                    || {
                                        IoError::Format(
                                            "Invalid encoding in DataPageHeaderV2".into(),
                                        )
                                    },
                                )?)
                        }
                        5 => def_len = Some(thrift_read_i32(r)?),
                        6 => rep_len = Some(thrift_read_i32(r)?),
                        7 => is_compressed = Some(tpe2 == TC_BOOL_TRUE),
                        8 => statistics = Some(parse_statistics(r)?),
                        _ => thrift_skip_field(r, tpe2)?,
                    }
                }

                data_ph_v2 = Some(DataPageHeaderV2 {
                    num_rows: num_rows.unwrap_or(0),
                    num_nulls: num_nulls.unwrap_or(0),
                    num_values: num_values.unwrap_or(0),
                    encoding: encoding.ok_or_else(|| {
                        IoError::Format("Missing encoding in DataPageHeaderV2".into())
                    })?,
                    definition_levels_byte_length: def_len.unwrap_or(0),
                    repetition_levels_byte_length: rep_len.unwrap_or(0),
                    is_compressed: is_compressed.unwrap_or(false),
                    statistics,
                });
            }
            _ => thrift_skip_field(r, tpe)?,
        }
    }

    Ok(PageHeader {
        type_: ptype.ok_or_else(|| IoError::Format("Missing PageType".into()))?,
        uncompressed_page_size: uncomp.unwrap_or(0),
        compressed_page_size: compr.unwrap_or(0),
        data_page_header: data_ph,
        data_page_header_v2: data_ph_v2,
        dictionary_page_header: dict_ph,
    })
}

fn parse_data_page_header<R: Read>(r: &mut R) -> Result<DataPageHeader, IoError> {
    let mut last = 0i16;
    let mut num_values = None;
    let mut encoding = None;
    let mut dlev = None;
    let mut rlev = None;
    let mut stats = None;

    loop {
        let (tpe, id) = thrift_read_field_begin(r, &mut last)?;
        if tpe == 0 {
            break;
        }
        match id {
            1 => num_values = Some(thrift_read_i32(r)?),
            2 => encoding = Some(ParquetEncoding::from_i32(thrift_read_i32(r)?).unwrap()),
            3 => dlev = Some(ParquetEncoding::from_i32(thrift_read_i32(r)?).unwrap()),
            4 => rlev = Some(ParquetEncoding::from_i32(thrift_read_i32(r)?).unwrap()),
            5 => stats = Some(parse_statistics(r)?),
            _ => thrift_skip_field(r, tpe)?,
        }
    }
    Ok(DataPageHeader {
        num_values: num_values.unwrap_or(0),
        encoding: encoding.unwrap(),
        definition_level_encoding: dlev.unwrap(),
        repetition_level_encoding: rlev.unwrap(),
        statistics: stats,
    })
}

// Low-level TCompactProtocol readers.
//
// Compact type identifiers from the Thrift spec. Bool field values ride in
// the type nibble - TC_BOOL_TRUE and TC_BOOL_FALSE encode the value directly.

const TC_BOOL_TRUE: u8 = 1;
const TC_BOOL_FALSE: u8 = 2;
const TC_BYTE: u8 = 3;
const TC_I16: u8 = 4;
const TC_I32: u8 = 5;
const TC_I64: u8 = 6;
const TC_DOUBLE: u8 = 7;
const TC_BINARY: u8 = 8;
const TC_LIST: u8 = 9;
const TC_SET: u8 = 10;
const TC_MAP: u8 = 11;
const TC_STRUCT: u8 = 12;

/// Read an unsigned LEB128 varint.
fn thrift_read_varint<R: Read>(r: &mut R) -> Result<u64, IoError> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        val |= ((b[0] & 0x7f) as u64) << shift;
        if b[0] & 0x80 == 0 {
            return Ok(val);
        }
        shift += 7;
        if shift > 63 {
            return Err(IoError::Format("varint overflow".into()));
        }
    }
}

/// Decode a zigzag-encoded unsigned value back to a signed integer.
fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Read a compact field header, advancing the struct's running field id.
///
/// Returns the compact type nibble and the resolved field id. A short-form
/// header packs a 1..=15 delta into the high nibble; a zero delta signals the
/// long form with a zigzag varint field id following. The field-stop byte 0
/// returns `(0, 0)`.
fn thrift_read_field_begin<R: Read>(r: &mut R, last: &mut i16) -> Result<(u8, i16), IoError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    let byte = b[0];
    if byte == 0 {
        return Ok((0, 0));
    }
    let ctype = byte & 0x0f;
    let delta = (byte >> 4) & 0x0f;
    let id = if delta == 0 {
        zigzag_decode(thrift_read_varint(r)?) as i16
    } else {
        *last + delta as i16
    };
    *last = id;
    Ok((ctype, id))
}

/// Read a zigzag varint i32 value.
fn thrift_read_i32<R: Read>(r: &mut R) -> Result<i32, IoError> {
    Ok(zigzag_decode(thrift_read_varint(r)?) as i32)
}

/// Read a zigzag varint i64 value.
fn thrift_read_i64<R: Read>(r: &mut R) -> Result<i64, IoError> {
    Ok(zigzag_decode(thrift_read_varint(r)?))
}

/// Read a compact string - varint length followed by UTF-8 bytes.
fn thrift_read_string<R: Read>(r: &mut R) -> Result<String, IoError> {
    let buf = thrift_read_bytes(r)?;
    String::from_utf8(buf).map_err(|e| IoError::Format(format!("UTF8 error: {}", e)))
}

/// Read a compact binary value - varint length followed by the raw bytes.
fn thrift_read_bytes<R: Read>(r: &mut R) -> Result<Vec<u8>, IoError> {
    let len = thrift_read_varint(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read a compact list header, returning the element compact type and count.
///
/// One byte packs the size in the high nibble and the element type in the low
/// nibble. A size nibble of 0xF signals the long form with a varint count.
fn thrift_read_list_begin<R: Read>(r: &mut R) -> Result<(u8, usize), IoError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    let elem_tpe = b[0] & 0x0f;
    let size_nibble = (b[0] >> 4) & 0x0f;
    let len = if size_nibble == 0x0f {
        thrift_read_varint(r)? as usize
    } else {
        size_nibble as usize
    };
    Ok((elem_tpe, len))
}

/// Read a compact map header, returning the key type, value type, and count.
///
/// The count is a leading varint. An empty map stops there with no key/value
/// type byte; otherwise a single byte packs the key type in the high nibble
/// and the value type in the low nibble.
fn thrift_read_map_begin<R: Read>(r: &mut R) -> Result<(u8, u8, usize), IoError> {
    let len = thrift_read_varint(r)? as usize;
    if len == 0 {
        return Ok((0, 0, 0));
    }
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    let kt = (b[0] >> 4) & 0x0f;
    let vt = b[0] & 0x0f;
    Ok((kt, vt, len))
}

/// Skip a field value given its compact type. Bool field values ride in the
/// type nibble and carry no payload, so bool fields consume nothing here.
fn thrift_skip_field<R: Read>(r: &mut R, ctype: u8) -> Result<(), IoError> {
    match ctype {
        TC_BOOL_TRUE | TC_BOOL_FALSE => Ok(()),
        _ => thrift_skip_value(r, ctype),
    }
}

/// Skip a single value of the given compact type, recursing through nested
/// structs, lists, sets, and maps. Bool values encountered as container
/// elements consume one byte, unlike bool struct fields.
fn thrift_skip_value<R: Read>(r: &mut R, ctype: u8) -> Result<(), IoError> {
    match ctype {
        TC_BOOL_TRUE | TC_BOOL_FALSE | TC_BYTE => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
        }
        TC_I16 | TC_I32 | TC_I64 => {
            thrift_read_varint(r)?;
        }
        TC_DOUBLE => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
        }
        TC_BINARY => {
            let _ = thrift_read_bytes(r)?;
        }
        TC_LIST | TC_SET => {
            let (et, len) = thrift_read_list_begin(r)?;
            for _ in 0..len {
                thrift_skip_value(r, et)?;
            }
        }
        TC_MAP => {
            let (kt, vt, len) = thrift_read_map_begin(r)?;
            for _ in 0..len {
                thrift_skip_value(r, kt)?;
                thrift_skip_value(r, vt)?;
            }
        }
        TC_STRUCT => {
            let mut inner = 0i16;
            loop {
                let (ft, _) = thrift_read_field_begin(r, &mut inner)?;
                if ft == 0 {
                    break;
                }
                thrift_skip_field(r, ft)?;
            }
        }
        _ => {
            return Err(IoError::Format(format!(
                "Cannot skip unknown thrift type {}",
                ctype
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::models::encoders::parquet::data::encode_dictionary_indices_rle;

    use super::*;

    #[test]
    fn hybrid_rle_run() {
        // pattern: 6× value 3, bit-width = 2
        // header = run_len << 1 (=12)  => 0x0c
        // encoded value = 3 = 0b11  => two bytes (little-endian)
        let buf = [0x0c, 0x03, 0x00];
        let out = super::decode_hybrid(&buf, 2, 6).unwrap();
        assert_eq!(out.as_slice(), &[3, 3, 3, 3, 3, 3]);
    }

    #[test]
    fn hybrid_bitpacked_single_group() {
        // eight values: [1,0,1,0,1,0,1,0]  (bit-width 1)
        // header = (groups=1)<<1 | 1   => 3
        // packed byte = 0b01010101 = 0x55
        let buf = [0x03, 0x55];
        let out = super::decode_hybrid(&buf, 1, 8).unwrap();
        assert_eq!(out.as_slice(), &[1, 0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn hybrid_mixed_runs() {
        let expect = vec64![7, 7, 7, 1, 2, 3, 4, 5, 7, 7, 7, 7];
        let mut tmp = Vec::new();
        encode_dictionary_indices_rle(&expect, &mut tmp).unwrap();
        let bit_width = tmp[0];
        let buf = &tmp[1..]; // hybrid wants stream after bitWidth
        let out = super::decode_hybrid(buf, bit_width, expect.len()).unwrap();
        assert_eq!(out.as_slice(), expect.as_slice());
    }
  
    // Dictionary encoding is a page storage detail: the reader expands it
    // before decode_column and preserves the schema's UTF8 column type.
    #[cfg(feature = "snappy")]
    #[test]
    fn read_dictionary_encoded_strings() {
        use minarrow::MaskedArray;

        let bytes = include_bytes!("../../../pyarrow-roundtrip/pyarrow_simple.parquet");
        let table = load_parquet_table(Cursor::new(bytes.as_slice())).unwrap();
        assert_eq!(table.n_rows, 5);
        let column = &table.cols[1];
        assert_eq!(column.field.name, "name");
        assert_eq!(column.field.dtype, ArrowType::String);
        match &column.array {
            Array::TextArray(TextArray::String32(values)) => {
                let decoded: Vec<_> = (0..table.n_rows).map(|i| values.get(i)).collect();
                assert_eq!(
                    decoded,
                    [Some("Alice"), Some("Bob"), Some("Charlie"), Some("Diana"), Some("Eve")]
                );
                assert_eq!(column.null_count, 0);
            }
            other => panic!("expected decoded UTF8 strings, got {other:?}"),
        }
    }

    #[test]
    fn decode_column_plain_int32() {
        let values = [10i32, 20, -5];
        // Encode the values as PLAIN (little-endian bytes)
        let mut buf = Vec::new();
        for v in &values {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let def_levels = vec![true; values.len()];
        let array = decode_column(
            &ArrowType::Int32,
            ParquetPhysicalType::Int32,
            ValueLayout::Fixed(4),
            &buf,
            values.len(),
            def_levels.clone(),
        )
        .unwrap();

        match array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &values);
                assert!(arr.null_mask.as_ref().unwrap().all_true());
            }
            _ => panic!("unexpected array {:?}", array),
        }
    }

    #[test]
    fn decode_column_boolean_plain() {
        let bits = [true, false, true, true, false, false];
        // one byte per row, as the page readers unpack boolean pages
        let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
        let def_levels = vec![true; bits.len()];
        let array = decode_column(
            &ArrowType::Boolean,
            ParquetPhysicalType::Boolean,
            ValueLayout::Fixed(1),
            &bytes,
            bits.len(),
            def_levels,
        )
        .unwrap();

        match array {
            Array::BooleanArray(arr) => {
                let out: Vec<bool> = (0..bits.len()).map(|i| arr.data.get(i)).collect();
                assert_eq!(out.as_slice(), &bits);
            }
            _ => panic!("unexpected array {:?}", array),
        }
    }
}
