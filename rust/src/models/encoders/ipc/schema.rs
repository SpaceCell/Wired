// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # IPC FlatBuffers Builders
//!
//! Internal Helpers that construct FlatBuffer-encoded Arrow IPC artefacts used by the encoder.
//! This includes schema messages, record batches, dictionary batches, and the file footer with block metadata.
//!
//! Follows the IPC protocol as outlined
//! [here](https://arrow.apache.org/docs/format/Columnar.html#serialisation-and-interprocess-communication-ipc).
//!
//! Uses the generated FlatBuffers types under `src/arrow/*`, that are built from `src/flatb/*`.

use std::io::{self};

use minarrow::ffi::arrow_dtype::CategoricalIndexType;
#[cfg(feature = "datetime")]
use minarrow::enums::time_units::TimeUnit;
use minarrow::{ArrowType, Field};

use crate::arrow::file::org::apache::arrow::flatbuf as fbf;
use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
use crate::debug_println;
use crate::traits::stream_buffer::StreamBuffer;

/// Build the Arrow schema as a FlatBuffer-encoded IPC message.
///
/// # Arguments
/// - `fields`: Slice of Arrow [`Field`] definitions.
///
/// # Returns
/// IPC-encoded schema message (suitable for inclusion in an IPC frame).
///
/// # Errors
/// Returns error for any unsupported Arrow type.
pub fn build_flatbuf_schema(
    fbb: &mut flatbuffers::FlatBufferBuilder<'static>,
    schema: &[Field],
) -> io::Result<Vec<u8>> {
    fbb.reset();
    let fb_fields = build_flatbuf_fields(fbb, schema)?;
    let fields_vec = fbb.create_vector(&fb_fields);

    // let features_vec = fbb.create_vector(&[fbm::Feature::UNUSED]);
    let schema_obj = fbm::Schema::create(
        fbb,
        &fbm::SchemaArgs {
            endianness: fbm::Endianness::Little,
            fields: Some(fields_vec),
            custom_metadata: None,
            features: None, // features: Some(features_vec)
        },
    );
    let msg = fbm::Message::create(
        fbb,
        &fbm::MessageArgs {
            version: fbm::MetadataVersion::V5,
            header_type: fbm::MessageHeader::Schema,
            header: Some(schema_obj.as_union_value()),
            bodyLength: 0,
            custom_metadata: None,
        },
    );
    fbb.finish(msg, None);
    let flatbuffers = fbb.finished_data().to_vec();

    Ok(flatbuffers)
}

/// Build a vector of Arrow schema `Field` objects using the message-specific
/// FlatBuffers definitions.
///
/// This constructs FlatBuffers offsets for each Arrow field, with each field
/// encoded using the `fbm` (message) schema types.
///
/// # Arguments
/// - `fbb`: The FlatBufferBuilder used to allocate objects.
/// - `schema`: Slice of Arrow [`Field`] definitions.
///
/// # Returns
/// Vector of WIPOffset<fbm::Field> suitable for inclusion in an IPC message schema.
///
/// # Errors
/// Returns error for any unsupported Arrow type.
fn build_flatbuf_fields<'fbb>(
    fbb: &mut flatbuffers::FlatBufferBuilder<'fbb>,
    schema: &[Field],
) -> io::Result<Vec<flatbuffers::WIPOffset<fbm::Field<'fbb>>>> {
    let mut fb_fields = Vec::with_capacity(schema.len());

    // Build all fields
    for (idx, field) in schema.iter().enumerate() {
        fb_fields.push(build_flatbuf_field(fbb, field, idx)?);
    }

    Ok(fb_fields)
}

/// Build a single Arrow schema `Field` object using message-specific FlatBuffers definitions.
///
/// This encodes all Arrow logical types as a FlatBuffer offset, for use in Arrow IPC message metadata.
///
/// # Arguments
/// - `fbb`: The FlatBufferBuilder used to allocate objects.
/// - `field`: The Arrow [`Field`] schema element.
///
/// # Returns
/// FlatBuffers WIPOffset<fbm::Field> describing the field for the IPC message schema.
///
/// # Errors
/// Returns error for any unsupported Arrow type.
fn build_flatbuf_field<'fbb>(
    fbb: &mut flatbuffers::FlatBufferBuilder<'fbb>,
    field: &Field,
    col_idx: usize,
) -> io::Result<flatbuffers::WIPOffset<fbm::Field<'fbb>>> {
    // Create field name
    let fb_name = fbb.create_string(&field.name);

    // Create empty children vector (we don't use nested types)
    let children = fbb.create_vector::<flatbuffers::WIPOffset<fbm::Field>>(&[]);

    // No custom metadata
    let custom_metadata = None;

    // Build the type and optional dictionary encoding
    let (fb_type_type, fb_type_offset, fb_dict) = match &field.dtype {
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 8,
                    is_signed: true,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 16,
                    is_signed: true,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::Int32 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 32,
                    is_signed: true,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::Int64 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 64,
                    is_signed: true,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 8,
                    is_signed: false,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 16,
                    is_signed: false,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::UInt32 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 32,
                    is_signed: false,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::UInt64 => {
            let int = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: 64,
                    is_signed: false,
                },
            );
            (fbm::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::Float32 => {
            let fp = fbm::FloatingPoint::create(
                fbb,
                &fbm::FloatingPointArgs {
                    precision: fbm::Precision::SINGLE,
                },
            );
            (fbm::Type::FloatingPoint, Some(fp.as_union_value()), None)
        }
        ArrowType::Float64 => {
            let fp = fbm::FloatingPoint::create(
                fbb,
                &fbm::FloatingPointArgs {
                    precision: fbm::Precision::DOUBLE,
                },
            );
            (fbm::Type::FloatingPoint, Some(fp.as_union_value()), None)
        }
        ArrowType::Boolean => {
            let bl = fbm::Bool::create(fbb, &fbm::BoolArgs {});
            (fbm::Type::Bool, Some(bl.as_union_value()), None)
        }
        ArrowType::String => {
            let s = fbm::Utf8::create(fbb, &fbm::Utf8Args {});
            (fbm::Type::Utf8, Some(s.as_union_value()), None)
        }
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => {
            let s = fbm::LargeUtf8::create(fbb, &fbm::LargeUtf8Args {});
            (fbm::Type::LargeUtf8, Some(s.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Date32 => {
            let date = fbm::Date::create(
                fbb,
                &fbm::DateArgs {
                    unit: fbm::DateUnit::DAY,
                },
            );
            (fbm::Type::Date, Some(date.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => {
            let date = fbm::Date::create(
                fbb,
                &fbm::DateArgs {
                    unit: fbm::DateUnit::MILLISECOND,
                },
            );
            (fbm::Type::Date, Some(date.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(unit, tz) => {
            let unit = match unit {
                TimeUnit::Seconds => fbm::TimeUnit::SECOND,
                TimeUnit::Milliseconds => fbm::TimeUnit::MILLISECOND,
                TimeUnit::Microseconds => fbm::TimeUnit::MICROSECOND,
                TimeUnit::Nanoseconds => fbm::TimeUnit::NANOSECOND,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Timestamp field '{}' has non-clock unit {other:?}", field.name),
                    ));
                }
            };
            let timezone = tz.as_ref().map(|s| fbb.create_string(s));
            let ts = fbm::Timestamp::create(fbb, &fbm::TimestampArgs { unit, timezone });
            (fbm::Type::Timestamp, Some(ts.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Time32(unit) | ArrowType::Time64(unit) => {
            let bit_width: i32 =
                if matches!(&field.dtype, ArrowType::Time32(_)) { 32 } else { 64 };
            let unit = match unit {
                TimeUnit::Seconds => fbm::TimeUnit::SECOND,
                TimeUnit::Milliseconds => fbm::TimeUnit::MILLISECOND,
                TimeUnit::Microseconds => fbm::TimeUnit::MICROSECOND,
                TimeUnit::Nanoseconds => fbm::TimeUnit::NANOSECOND,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Time field '{}' has non-clock unit {other:?}", field.name),
                    ));
                }
            };
            let time = fbm::Time::create(fbb, &fbm::TimeArgs { unit, bitWidth: bit_width });
            (fbm::Type::Time, Some(time.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(unit) => {
            let unit = match unit {
                TimeUnit::Seconds => fbm::TimeUnit::SECOND,
                TimeUnit::Milliseconds => fbm::TimeUnit::MILLISECOND,
                TimeUnit::Microseconds => fbm::TimeUnit::MICROSECOND,
                TimeUnit::Nanoseconds => fbm::TimeUnit::NANOSECOND,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Duration field '{}' has non-clock unit {other:?}", field.name),
                    ));
                }
            };
            let duration = fbm::Duration::create(fbb, &fbm::DurationArgs { unit });
            (fbm::Type::Duration, Some(duration.as_union_value()), None)
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal32(p, s) => {
            let dec = fbm::Decimal::create(
                fbb,
                &fbm::DecimalArgs {
                    precision: *p as i32,
                    scale: *s as i32,
                    bitWidth: 32,
                },
            );
            (fbm::Type::Decimal, Some(dec.as_union_value()), None)
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal64(p, s) => {
            let dec = fbm::Decimal::create(
                fbb,
                &fbm::DecimalArgs {
                    precision: *p as i32,
                    scale: *s as i32,
                    bitWidth: 64,
                },
            );
            (fbm::Type::Decimal, Some(dec.as_union_value()), None)
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal128(p, s) => {
            let dec = fbm::Decimal::create(
                fbb,
                &fbm::DecimalArgs {
                    precision: *p as i32,
                    scale: *s as i32,
                    bitWidth: 128,
                },
            );
            (fbm::Type::Decimal, Some(dec.as_union_value()), None)
        }
        ArrowType::Dictionary(idx_ty) => {
            // Build index type for dictionary
            let idx_width = match idx_ty {
                CategoricalIndexType::UInt32 => 32,
                #[cfg(feature = "default_categorical_8")]
                CategoricalIndexType::UInt8 => 8,
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt64 => 64,
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt16 => 16,
            };

            // Create the index type Int
            let index_type = fbm::Int::create(
                fbb,
                &fbm::IntArgs {
                    bitWidth: idx_width,
                    is_signed: false,
                },
            );

            // Create dictionary encoding with column index as ID
            let dict = fbm::DictionaryEncoding::create(
                fbb,
                &fbm::DictionaryEncodingArgs {
                    id: col_idx as i64,
                    indexType: Some(index_type),
                    isOrdered: false,
                    dictionaryKind: fbm::DictionaryKind::DenseArray,
                },
            );

            // For dictionary-encoded fields, the type is Utf8 (string dictionary)
            // but the physical storage is Int (indices)
            let utf8_type = fbm::Utf8::create(fbb, &fbm::Utf8Args {});
            (
                fbm::Type::Utf8,
                Some(utf8_type.as_union_value()),
                Some(dict),
            )
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported type in schema: {}", field.name),
            ));
        }
    };

    // Create the field with all components
    Ok(fbm::Field::create(
        fbb,
        &fbm::FieldArgs {
            name: Some(fb_name),
            nullable: field.nullable,
            type_type: fb_type_type,
            type_: fb_type_offset,
            dictionary: fb_dict,
            children: Some(children), // Always include children, even if empty
            custom_metadata,
        },
    ))
}

/// Build an Arrow record batch flatbuffers message for the given table.
///
/// # Arguments
/// - `tbl`: Table to encode.
/// - `fields`: Fields describing schema.
/// - `dicts`: Dictionary mapping for categorical columns.
///
/// # Returns
/// FlatBuffer message
///
/// # Errors
/// Returns error if unsupported column type or shape mismatch.
pub(crate) fn build_flatbuf_recordbatch(
    fbb: &mut flatbuffers::FlatBufferBuilder<'static>,
    n_rows: usize,
    fb_field_nodes: &[fbm::FieldNode],
    fb_buffers: &[fbm::Buffer],
    body_len: usize,
    compression: Option<fbm::CompressionType>,
    custom_metadata: Option<&[(String, String)]>,
) -> io::Result<Vec<u8>> {
    fbb.reset();
    let fb_nodes_vec = fbb.create_vector(fb_field_nodes);
    let fb_buffers_vec = fbb.create_vector(fb_buffers);
    let fb_compression = compression.map(|codec| {
        fbm::BodyCompression::create(
            fbb,
            &fbm::BodyCompressionArgs {
                codec,
                method: fbm::BodyCompressionMethod::BUFFER,
            },
        )
    });
    let rb = fbm::RecordBatch::create(
        fbb,
        &fbm::RecordBatchArgs {
            length: n_rows as i64,
            nodes: Some(fb_nodes_vec),
            buffers: Some(fb_buffers_vec),
            compression: fb_compression,
            variadicBufferCounts: None,
        },
    );
    // Attach the supplied key/value pairs as Arrow custom_metadata on the
    // message envelope.
    let custom_metadata = match custom_metadata {
        None => None,
        Some(pairs) => {
            let kvs: Vec<_> = pairs
                .iter()
                .map(|(key, value)| {
                    let key = fbb.create_string(key);
                    let value = fbb.create_string(value);
                    fbm::KeyValue::create(
                        fbb,
                        &fbm::KeyValueArgs { key: Some(key), value: Some(value) },
                    )
                })
                .collect();
            Some(fbb.create_vector(&kvs))
        }
    };
    let meta = fbm::Message::create(
        fbb,
        &fbm::MessageArgs {
            version: fbm::MetadataVersion::V5,
            header_type: fbm::MessageHeader::RecordBatch,
            header: Some(rb.as_union_value()),
            bodyLength: body_len as i64,
            custom_metadata,
        },
    );
    fbb.finish(meta, None);
    Ok(fbb.finished_data().to_vec())
}

/// Build a dictionary batch (metadata and body) for the given dictionary ID and values.
///
/// # Arguments
/// - `dict_id`: Dictionary ID (matches column index).
/// - `uniques`: List of unique values for the dictionary.
///
/// # Returns
/// Tuple: (`FlatBuffer message`, `body buffer`)
///
/// # Errors
/// Returns error if input is inconsistent or unsupported.
pub fn encode_flatbuf_dictionary<B: StreamBuffer>(
    fbb: &mut flatbuffers::FlatBufferBuilder<'static>,
    id: i64,
    uniques: &[String],
) -> io::Result<(Vec<u8>, B)> {
    debug_println!("Encoding flatbuffers dictionary.");
    fbb.reset();

    // Build the string data and offsets. Note these are byte payload offsets,
    // not the offsets from `StringArray` in minarrow. Confusingly, we are dealing
    // with the unique values payload here.
    //
    // minarrow uses plain rust String for that (but, allows up to u64 for its internal `StringArray`
    // array offsets), targeting simplicity ergonomics, hence it is u32 for the String transport payload offsets
    // here always.
    let mut data_buf = Vec::<u8>::new();
    let mut offs = Vec::<u32>::with_capacity(uniques.len() + 1);
    offs.push(0);

    for s in uniques {
        data_buf.extend_from_slice(s.as_bytes());
        offs.push(data_buf.len() as u32);
    }

    let offset_size = std::mem::size_of::<u32>();

    // Create the body buffer with proper alignment
    let mut body = B::with_capacity(offset_size + offs.len() * offset_size + data_buf.len());

    // Important: Arrow expects buffers to be aligned, but the actual data layout is:
    // 1. Null buffer (empty for dictionary)
    // 2. Offsets buffer
    // 3. Data buffer

    // The offsets need to be written as raw bytes.
    // SAFETY: `offs` is a `Vec<u32>` so `offs.as_ptr()` points to
    // `offs.len() * size_of::<u32>()` valid bytes living for the
    // encode call. The integer offset type has no padding or invalid bit
    // patterns, so a byte view is sound for the duration of the body
    // construction.
    let offs_bytes =
        unsafe { std::slice::from_raw_parts(offs.as_ptr() as *const u8, offs.len() * offset_size) };

    // Write body: offsets then data
    body.extend_from_slice(offs_bytes);
    body.extend_from_slice(&data_buf);

    // Create field node
    let field_nodes = vec![fbm::FieldNode::new(uniques.len() as i64, 0)];

    // Create buffers metadata
    // Buffer 0: null buffer (empty)
    // Buffer 1: offsets
    // Buffer 2: string data
    let buffers = vec![
        fbm::Buffer::new(0, 0),                       // null buffer
        fbm::Buffer::new(0, offs_bytes.len() as i64), // offsets
        fbm::Buffer::new(offs_bytes.len() as i64, data_buf.len() as i64), // data
    ];

    // Build the FlatBuffer message
    let field_nodes_vec = fbb.create_vector(&field_nodes);
    let buffers_vec = fbb.create_vector(&buffers);

    let rb = fbm::RecordBatch::create(
        fbb,
        &fbm::RecordBatchArgs {
            length: uniques.len() as i64,
            nodes: Some(field_nodes_vec),
            buffers: Some(buffers_vec),
            compression: None,
            variadicBufferCounts: None,
        },
    );

    let dict_batch = fbm::DictionaryBatch::create(
        fbb,
        &fbm::DictionaryBatchArgs {
            id,
            data: Some(rb),
            isDelta: false,
        },
    );

    let msg = fbm::Message::create(
        fbb,
        &fbm::MessageArgs {
            version: fbm::MetadataVersion::V5,
            header_type: fbm::MessageHeader::DictionaryBatch,
            header: Some(dict_batch.as_union_value()),
            bodyLength: body.len() as i64,
            custom_metadata: None,
        },
    );

    fbb.finish(msg, None);
    Ok((fbb.finished_data().to_vec(), body))
}

/// Build a vector of Arrow schema `Field` objects (for the Arrow File footer),
/// using the file-specific FlatBuffers definitions.
///
/// This constructs FlatBuffers offsets for each Arrow field, with each field
/// encoded using the `fbf` (file) schema types.
///
/// # Arguments
/// - `fbb`: The FlatBufferBuilder used to allocate objects.
/// - `schema`: Slice of Arrow [`Field`] definitions (schema).
///
/// # Returns
/// Vector of WIPOffset<fbf::Field> for inclusion in the file footer schema.
///
/// # Errors
/// Returns error for any unsupported Arrow type.
fn build_flatbuf_fields_file<'fbb>(
    fbb: &mut flatbuffers::FlatBufferBuilder<'fbb>,
    schema: &[Field],
) -> io::Result<Vec<flatbuffers::WIPOffset<fbf::Field<'fbb>>>> {
    let mut fb_fields = Vec::with_capacity(schema.len());
    for (col_idx, f) in schema.iter().enumerate() {
        fb_fields.push(build_flatbuf_field_file(fbb, f, col_idx)?);
    }
    Ok(fb_fields)
}

/// Build a single Arrow schema `Field` object (for the Arrow File footer),
/// using file-specific FlatBuffers definitions.
///
/// This encodes all Arrow logical types as a FlatBuffer offset, for use in Arrow File metadata.
///
/// # Arguments
/// - `fbb`: The FlatBufferBuilder used to allocate objects.
/// - `field`: The Arrow [`Field`] schema element.
///
/// # Returns
/// FlatBuffers WIPOffset<fbf::Field> describing the field for the file footer.
///
/// # Errors
/// Returns error for any unsupported Arrow type.
fn build_flatbuf_field_file<'fbb>(
    fbb: &mut flatbuffers::FlatBufferBuilder<'fbb>,
    field: &Field,
    col_idx: usize,
) -> io::Result<flatbuffers::WIPOffset<fbf::Field<'fbb>>> {
    let fb_name = fbb.create_string(&field.name);
    let children = fbb.create_vector::<flatbuffers::WIPOffset<fbf::Field>>(&[]);
    let custom_metadata = None;

    let (fb_type_type, fb_type_offset, fb_dict) = match &field.dtype {
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 8,
                    is_signed: true,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 16,
                    is_signed: true,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::Int32 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 32,
                    is_signed: true,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::Int64 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 64,
                    is_signed: true,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 8,
                    is_signed: false,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 16,
                    is_signed: false,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::UInt32 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 32,
                    is_signed: false,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::UInt64 => {
            let int = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: 64,
                    is_signed: false,
                },
            );
            (fbf::Type::Int, Some(int.as_union_value()), None)
        }
        ArrowType::Float32 => {
            let fp = fbf::FloatingPoint::create(
                fbb,
                &fbf::FloatingPointArgs {
                    precision: fbf::Precision::SINGLE,
                },
            );
            (fbf::Type::FloatingPoint, Some(fp.as_union_value()), None)
        }
        ArrowType::Float64 => {
            let fp = fbf::FloatingPoint::create(
                fbb,
                &fbf::FloatingPointArgs {
                    precision: fbf::Precision::DOUBLE,
                },
            );
            (fbf::Type::FloatingPoint, Some(fp.as_union_value()), None)
        }
        ArrowType::Boolean => {
            let bl = fbf::Bool::create(fbb, &fbf::BoolArgs {});
            (fbf::Type::Bool, Some(bl.as_union_value()), None)
        }
        ArrowType::String => {
            let s = fbf::Utf8::create(fbb, &fbf::Utf8Args {});
            (fbf::Type::Utf8, Some(s.as_union_value()), None)
        }
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => {
            let s = fbf::LargeUtf8::create(fbb, &fbf::LargeUtf8Args {});
            (fbf::Type::LargeUtf8, Some(s.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Date32 => {
            let date = fbf::Date::create(
                fbb,
                &fbf::DateArgs {
                    unit: fbf::DateUnit::DAY,
                },
            );
            (fbf::Type::Date, Some(date.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => {
            let date = fbf::Date::create(
                fbb,
                &fbf::DateArgs {
                    unit: fbf::DateUnit::MILLISECOND,
                },
            );
            (fbf::Type::Date, Some(date.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(unit, tz) => {
            let unit = match unit {
                TimeUnit::Seconds => fbf::TimeUnit::SECOND,
                TimeUnit::Milliseconds => fbf::TimeUnit::MILLISECOND,
                TimeUnit::Microseconds => fbf::TimeUnit::MICROSECOND,
                TimeUnit::Nanoseconds => fbf::TimeUnit::NANOSECOND,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Timestamp field '{}' has non-clock unit {other:?}", field.name),
                    ));
                }
            };
            let timezone = tz.as_ref().map(|s| fbb.create_string(s));
            let ts = fbf::Timestamp::create(fbb, &fbf::TimestampArgs { unit, timezone });
            (fbf::Type::Timestamp, Some(ts.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Time32(unit) | ArrowType::Time64(unit) => {
            let bit_width: i32 =
                if matches!(&field.dtype, ArrowType::Time32(_)) { 32 } else { 64 };
            let unit = match unit {
                TimeUnit::Seconds => fbf::TimeUnit::SECOND,
                TimeUnit::Milliseconds => fbf::TimeUnit::MILLISECOND,
                TimeUnit::Microseconds => fbf::TimeUnit::MICROSECOND,
                TimeUnit::Nanoseconds => fbf::TimeUnit::NANOSECOND,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Time field '{}' has non-clock unit {other:?}", field.name),
                    ));
                }
            };
            let time = fbf::Time::create(fbb, &fbf::TimeArgs { unit, bitWidth: bit_width });
            (fbf::Type::Time, Some(time.as_union_value()), None)
        }
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(unit) => {
            let unit = match unit {
                TimeUnit::Seconds => fbf::TimeUnit::SECOND,
                TimeUnit::Milliseconds => fbf::TimeUnit::MILLISECOND,
                TimeUnit::Microseconds => fbf::TimeUnit::MICROSECOND,
                TimeUnit::Nanoseconds => fbf::TimeUnit::NANOSECOND,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Duration field '{}' has non-clock unit {other:?}", field.name),
                    ));
                }
            };
            let duration = fbf::Duration::create(fbb, &fbf::DurationArgs { unit });
            (fbf::Type::Duration, Some(duration.as_union_value()), None)
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal32(p, s) => {
            let dec = fbf::Decimal::create(
                fbb,
                &fbf::DecimalArgs {
                    precision: *p as i32,
                    scale: *s as i32,
                    bitWidth: 32,
                },
            );
            (fbf::Type::Decimal, Some(dec.as_union_value()), None)
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal64(p, s) => {
            let dec = fbf::Decimal::create(
                fbb,
                &fbf::DecimalArgs {
                    precision: *p as i32,
                    scale: *s as i32,
                    bitWidth: 64,
                },
            );
            (fbf::Type::Decimal, Some(dec.as_union_value()), None)
        }
        #[cfg(feature = "decimal")]
        ArrowType::Decimal128(p, s) => {
            let dec = fbf::Decimal::create(
                fbb,
                &fbf::DecimalArgs {
                    precision: *p as i32,
                    scale: *s as i32,
                    bitWidth: 128,
                },
            );
            (fbf::Type::Decimal, Some(dec.as_union_value()), None)
        }
        ArrowType::Dictionary(idx_ty) => {
            let idx_width = match idx_ty {
                CategoricalIndexType::UInt32 => 32,
                #[cfg(feature = "default_categorical_8")]
                CategoricalIndexType::UInt8 => 8,
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt64 => 64,
                #[cfg(feature = "extended_categorical")]
                CategoricalIndexType::UInt16 => 16,
            };

            let index_type = fbf::Int::create(
                fbb,
                &fbf::IntArgs {
                    bitWidth: idx_width,
                    is_signed: false,
                },
            );

            let dict = fbf::DictionaryEncoding::create(
                fbb,
                &fbf::DictionaryEncodingArgs {
                    id: col_idx as i64,
                    indexType: Some(index_type),
                    isOrdered: false,
                    dictionaryKind: fbf::DictionaryKind::DenseArray,
                },
            );

            // File footer uses Utf8 type for dictionary fields
            let utf8_type = fbf::Utf8::create(fbb, &fbf::Utf8Args {});
            (
                fbf::Type::Utf8,
                Some(utf8_type.as_union_value()),
                Some(dict),
            )
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported type in schema: {}", field.name),
            ));
        }
    };

    Ok(fbf::Field::create(
        fbb,
        &fbf::FieldArgs {
            name: Some(fb_name),
            nullable: field.nullable,
            type_type: fb_type_type,
            type_: fb_type_offset,
            dictionary: fb_dict,
            children: Some(children),
            custom_metadata,
        },
    ))
}

/// Arrow IPC block meta for file footer
#[derive(Debug, Clone)]
pub struct FooterBlockMeta {
    pub offset: u64,
    pub metadata_len: u32,
    pub body_len: u64,
}

/// Build the Arrow file footer FlatBuffer message, including batch and dictionary block metadata.
///
/// # Arguments
/// - `fields`: Table schema.
/// - `dict_blocks`: Dictionary block metadata.
/// - `batch_blocks`: Record batch block metadata.
///
/// # Returns
/// FlatBuffer-encoded Arrow file footer.
///
/// # Errors
/// Returns error for any unsupported Arrow type.
pub fn build_flatbuf_footer(
    fbb: &mut flatbuffers::FlatBufferBuilder<'static>,
    schema: &[Field],
    blocks_dictionaries: &[FooterBlockMeta],
    blocks_record_batches: &[FooterBlockMeta],
) -> io::Result<Vec<u8>> {
    fbb.reset();
    let fb_dict_blocks: Vec<_> = blocks_dictionaries
        .iter()
        .map(|b| fbf::Block::new(b.offset as i64, b.metadata_len as i32, b.body_len as i64))
        .collect();
    let fb_batch_blocks: Vec<_> = blocks_record_batches
        .iter()
        .map(|b| fbf::Block::new(b.offset as i64, b.metadata_len as i32, b.body_len as i64))
        .collect();
    let dict_vec = fbb.create_vector(&fb_dict_blocks);
    let batch_vec = fbb.create_vector(&fb_batch_blocks);
    let fb_fields = build_flatbuf_fields_file(fbb, schema)?;
    let fields_vec = fbb.create_vector(&fb_fields);
    let schema_obj = fbf::Schema::create(
        fbb,
        &fbf::SchemaArgs {
            endianness: fbf::Endianness::Little,
            fields: Some(fields_vec),
            custom_metadata: None,
            features: None,
        },
    );
    let footer = fbf::Footer::create(
        fbb,
        &fbf::FooterArgs {
            version: fbf::MetadataVersion::V5,
            schema: Some(schema_obj),
            dictionaries: Some(dict_vec),
            recordBatches: Some(batch_vec),
            custom_metadata: None,
        },
    );
    fbb.finish(footer, None);
    Ok(fbb.finished_data().to_vec())
}
