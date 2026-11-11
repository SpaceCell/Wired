// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Arrow IPC Table Encoder
//!
//! Serialises `Table` values into Arrow IPC metadata and body pairs.
//! Handles schema, dictionary, and record batch encoding with state
//! tracking so schema and dictionaries are emitted once.
//!
//! The encoder produces `(meta, body)` pairs. Writing, framing, queuing,
//! and protocol-level concerns like EOS markers and file footers are
//! handled by the writer layer in `writers/ipc/`.

use std::collections::{HashMap, HashSet};
use std::io;

use minarrow::{ArrowType, Field, TableV};

use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
use crate::compression::{Compression, compress};
use crate::constants::DEFAULT_FRAME_ALLOCATION_SIZE;
use crate::enums::{IPCMessageProtocol, WriterState};
use crate::models::encoders::ipc::schema::{
    build_flatbuf_recordbatch, build_flatbuf_schema, encode_flatbuf_dictionary,
};
use crate::traits::stream_buffer::StreamBuffer;
use crate::utils::align_to;

/// Arrow IPC table encoder.
///
/// Serialises schema, dictionaries, and record batches as `(meta, body)`
/// pairs. Tracks encoding state so schema and dictionaries are emitted
/// once. Does not handle framing, queuing, or I/O - those responsibilities
/// belong to the writer layer.
pub struct TableEncoder<B: StreamBuffer + 'static> {
    /// Arrow IPC protocol - affects alignment calculations
    pub protocol: IPCMessageProtocol,
    /// Current encoding state
    pub state: WriterState,
    /// Compression codec for record batch bodies. `None` writes uncompressed.
    pub compression: Option<Compression>,
    /// Arrow schema for this stream
    pub schema: Vec<Field>,
    /// Set of dictionary IDs already encoded
    pub written_dict_ids: HashSet<i64>,
    /// Registered dictionaries for categorical columns
    pub dictionaries: HashMap<i64, Vec<String>>,
    /// FlatBuffer builder instance for serialisation
    pub fbb: flatbuffers::FlatBufferBuilder<'static>,
    /// B controls wire alignment for frame boundary calculations
    _alignment: std::marker::PhantomData<B>,
}

impl<B: StreamBuffer> TableEncoder<B> {
    /// Create a new encoder. `None` for `compression` writes uncompressed
    /// batches; `Some(codec)` compresses every record-batch body.
    pub fn new(
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
    ) -> Self {
        Self {
            protocol,
            state: WriterState::Fresh,
            compression,
            schema,
            written_dict_ids: HashSet::new(),
            dictionaries: HashMap::new(),
            fbb: flatbuffers::FlatBufferBuilder::with_capacity(4096),
            _alignment: std::marker::PhantomData,
        }
    }

    /// Register a dictionary for a categorical column.
    pub fn register_dictionary(&mut self, id: i64, uniques: Vec<String>) {
        self.dictionaries.insert(id, uniques);
    }

    /// Encode the schema as a flatbuffer metadata blob.
    /// Updates state to SchemaDone.
    pub fn encode_schema(&mut self) -> io::Result<Vec<u8>> {
        let meta = build_flatbuf_schema(&mut self.fbb, &self.schema)?;
        self.state = WriterState::SchemaDone;
        Ok(meta)
    }

    /// Encode a dictionary batch, returning `(meta, body)`.
    /// Returns `Ok(None)` if this dictionary was already encoded.
    pub fn encode_dictionary(&mut self, id: i64) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
        if self.written_dict_ids.contains(&id) {
            return Ok(None);
        }
        let uniques = self.dictionaries.get(&id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("dictionary id {id} not registered"),
            )
        })?;
        let (meta, body) = encode_flatbuf_dictionary(&mut self.fbb, id, uniques)?;
        self.written_dict_ids.insert(id);
        Ok(Some((meta, body)))
    }

    /// Collect dictionary IDs from the schema that need encoding for categorical columns.
    pub fn pending_dict_ids(&self) -> Vec<i64> {
        self.schema
            .iter()
            .enumerate()
            .filter_map(|(col_idx, field)| {
                if let ArrowType::Dictionary(_) = field.dtype {
                    Some(col_idx as i64)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Serialise a table view as an Arrow IPC record batch, returning (metadata, body) buffers.
    ///
    /// Uses `compute_body_layout` to collect column data as zero-copy slices,
    /// then writes them into a single body buffer.
    pub fn encode_record_batch(&mut self, view: &TableV) -> io::Result<(Vec<u8>, B)> {
        use crate::models::encoders::ipc::record_batch::compute_body_layout;

        let layout = compute_body_layout::<B>(view)?;

        // When compression is active, compress each buffer with a u64 LE
        // uncompressed length prefix and rebuild the buffer metadata against
        // the compressed layout, per the Arrow IPC BodyCompression spec.
        let (body, fb_buffers, body_size) = if let Some(codec) = self.compression {
            let mut compressed: Vec<Vec<u8>> = Vec::with_capacity(layout.regions.len());
            for region in &layout.regions {
                if region.data.is_empty() {
                    compressed.push(Vec::new());
                } else {
                    let raw = region.data.compression_bytes();
                    let c = compress(&raw, codec)
                        .map_err(|e| io::Error::other(format!("{}", e)))?;
                    let mut wire = Vec::with_capacity(8 + c.len());
                    wire.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                    wire.extend_from_slice(&c);
                    compressed.push(wire);
                }
            }
            let mut fb_buffers = Vec::with_capacity(compressed.len());
            let mut body = B::with_capacity(DEFAULT_FRAME_ALLOCATION_SIZE);
            let mut offset = 0usize;
            for c in &compressed {
                fb_buffers.push(fbm::Buffer::new(offset as i64, c.len() as i64));
                body.extend_from_slice(c);
                let pad = align_to::<B>(c.len());
                if pad > 0 {
                    body.extend_from_slice(&[0u8; 64][..pad]);
                }
                offset += c.len() + pad;
            }
            (body, fb_buffers, offset)
        } else {
            let mut body = B::with_capacity(layout.body_size.max(DEFAULT_FRAME_ALLOCATION_SIZE));
            for region in &layout.regions {
                region.data.write_into(&mut body);
                if region.pad > 0 {
                    body.extend_from_slice(&[0u8; 64][..region.pad]);
                }
            }
            (body, layout.fb_buffers, layout.body_size)
        };

        let fb_compression = match self.compression {
            Some(c) => Some(c.to_arrow_ipc_type()?),
            None => None,
        };
        let meta = build_flatbuf_recordbatch(
            &mut self.fbb,
            view.len,
            &layout.fb_field_nodes,
            &fb_buffers,
            body_size,
            fb_compression,
            None,
        )?;
        Ok((meta, body))
    }
}

/// The below are 'logical' unit tests confirming:
/// - Unique dictionary IDs are respected.
/// - The macro populates buffers and field nodes as intended.
/// - Null handling works as expected for all supported index types.
///
/// For full roundtrip IO on the Stream reader and writer, see "../tests".
#[cfg(test)]
mod tests {
    use std::fs::File as StdFile;
    use std::io::{Read, Write};
    use std::sync::Arc;

    use minarrow::ffi::arrow_dtype::CategoricalIndexType;
    use minarrow::{
        Array, Bitmask, Buffer, CategoricalArray, Field, FieldArray, IntegerArray, NumericArray,
        Table, TextArray, Vec64,
    };
    use tempfile::NamedTempFile;

    use crate::constants::{ARROW_MAGIC_NUMBER, ARROW_MAGIC_NUMBER_PADDED};

    use super::*;
    use crate::models::writers::ipc::table_stream::TableStreamWriter;

    fn make_bitmask(valid: &[bool]) -> Bitmask {
        let mut bits = vec![0u8; (valid.len() + 7) / 8];
        for (i, v) in valid.iter().enumerate() {
            if *v {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        Bitmask::new(Buffer::from(Vec64::from_slice(&bits[..])), valid.len())
    }

    fn dict_strs() -> Vec<String> {
        vec![
            "apple".to_string(),
            "banana".to_string(),
            "pear".to_string(),
        ]
    }

    fn make_schema(idx_ty: CategoricalIndexType, nullable: bool) -> Vec<Field> {
        vec![Field {
            name: "col".to_string(),
            dtype: ArrowType::Dictionary(idx_ty),
            nullable,
            metadata: Default::default(),
        }]
    }

    fn make_table(arr: FieldArray, n_rows: usize) -> Table {
        Table {
            cols: vec![arr],
            n_rows,
            name: "tbl".to_string(),
            ..Default::default()
        }
    }

    fn read_file_bytes(path: &std::path::Path) -> Vec<u8> {
        let mut f = StdFile::open(path).expect("file open");
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).expect("file read");
        buf
    }

    fn check_ipc_padding(buf: &[u8]) {
        // Checks: Flatbuffers message header (8 bytes) + meta + 64-byte padding + data
        // Only a minimal check here - you can parse the header and verify offset alignment
        // but at minimum ensure buffer length is a multiple of 8 and/or 64 as Arrow requires.
        assert_eq!(buf.len() % 8, 0, "Arrow IPC frame should be 8-byte aligned");
    }

    #[cfg(not(feature = "default_categorical_8"))]
    #[test]
    fn test_write_categorical_column_u32_to_file() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer: TableStreamWriter = TableStreamWriter::new(
            make_schema(CategoricalIndexType::UInt32, true),
            IPCMessageProtocol::Stream,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u32, 0, 2, 1])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, false, true, true])),
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt32),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical32(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl.clone()).unwrap();
        writer.finish().unwrap();

        let mut file = StdFile::create(&path).unwrap();
        for frame in writer.drain_all_frames() {
            use std::io::Write;
            file.write_all(&frame).unwrap();
        }
        file.flush().unwrap();
        drop(file);

        let buf = read_file_bytes(&path);
        assert!(!buf.is_empty());
        check_ipc_padding(&buf);
    }

    #[cfg(feature = "default_categorical_8")]
    #[test]
    fn test_write_categorical_column_u8_to_file() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer = TableStreamWriter::<Vec<u8>>::new(
            make_schema(CategoricalIndexType::UInt8, true),
            IPCMessageProtocol::Stream,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u8, 0, 2, 1])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, false, true, true])),
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt8),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical8(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl.clone()).unwrap();
        writer.finish().unwrap();

        let mut file = StdFile::create(&path).unwrap();
        for frame in writer.drain_all_frames() {
            use std::io::Write;
            file.write_all(&frame).unwrap();
        }
        file.flush().unwrap();
        drop(file);

        let buf = read_file_bytes(&path);
        assert!(!buf.is_empty());
        check_ipc_padding(&buf);
    }

    #[cfg(feature = "extended_categorical")]
    #[test]
    fn test_write_categorical_column_u8() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer: TableStreamWriter = TableStreamWriter::new(
            make_schema(CategoricalIndexType::UInt8, true),
            IPCMessageProtocol::Stream,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[1u8, 0, 2, 1])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, true, false, true])),
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt8),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical8(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl).unwrap();
        writer.finish().unwrap();

        let mut buf = Vec::new();
        for frame in writer.drain_all_frames() {
            buf.extend_from_slice(&frame);
        }

        assert!(!buf.is_empty());
        check_ipc_padding(&buf);
    }

    #[cfg(feature = "extended_categorical")]
    #[test]
    fn test_write_categorical_column_u16() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer: TableStreamWriter = TableStreamWriter::new(
            make_schema(CategoricalIndexType::UInt16, false),
            IPCMessageProtocol::Stream,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[2u16, 1, 0, 2])),
            Vec64::from(dict_strs()),
            None,
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt16),
                    nullable: false,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical16(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl).unwrap();
        writer.finish().unwrap();

        let mut buf = Vec::new();
        for frame in writer.drain_all_frames() {
            buf.extend_from_slice(&frame);
        }

        assert!(!buf.is_empty());
        check_ipc_padding(&buf);
    }

    #[cfg(feature = "extended_categorical")]
    #[test]
    fn test_write_categorical_column_u64() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer: TableStreamWriter = TableStreamWriter::new(
            make_schema(CategoricalIndexType::UInt64, false),
            IPCMessageProtocol::Stream,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[0u64, 2, 1, 0])),
            Vec64::from(dict_strs()),
            None,
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt64),
                    nullable: false,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical64(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl).unwrap();
        writer.finish().unwrap();

        let mut buf = Vec::new();
        for frame in writer.drain_all_frames() {
            buf.extend_from_slice(&frame);
        }

        assert!(!buf.is_empty());
        check_ipc_padding(&buf);
    }

    #[cfg(not(feature = "default_categorical_8"))]
    #[test]
    fn test_ipc_file_write_read_dict() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer: TableStreamWriter = TableStreamWriter::new(
            make_schema(CategoricalIndexType::UInt32, true),
            IPCMessageProtocol::File,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[0u32, 1, 1, 2])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, false, true, true])),
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt32),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical32(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl.clone()).unwrap();
        writer.finish().unwrap();

        // Write to temp file
        let mut file = StdFile::create(&path).unwrap();
        for frame in writer.drain_all_frames() {
            use std::io::Write;
            file.write_all(&frame).unwrap();
        }
        file.flush().unwrap();
        drop(file);

        let buf = read_file_bytes(&path);
        println!("Written buffer:\n{:?}", buf);
        assert!(
            buf.starts_with(ARROW_MAGIC_NUMBER_PADDED),
            "file must start with Arrow magic"
        );
        assert!(
            buf.ends_with(ARROW_MAGIC_NUMBER),
            "file must end with Arrow magic"
        );
        println!("Written buffer len : {:?}", buf.len());
        // We add 2 for the end magic marker, and the 3rd 4-byte contination marker.
        // This is more of a sanity check than a starting alignment check.
        assert!(
            (buf.len() + 4 + 2) % 8 == 0,
            "Arrow IPC file must be a multiple of 8 bytes"
        );
    }

    #[cfg(feature = "default_categorical_8")]
    #[test]
    fn test_ipc_file_write_read_dict() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer = TableStreamWriter::<Vec<u8>>::new(
            make_schema(CategoricalIndexType::UInt8, true),
            IPCMessageProtocol::File,
            None,
        );

        let arr = CategoricalArray::new(
            Buffer::from(Vec64::from_slice(&[0u8, 1, 1, 2])),
            Vec64::from(dict_strs()),
            Some(make_bitmask(&[true, false, true, true])),
        );

        writer.register_dictionary(0, dict_strs());

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Dictionary(CategoricalIndexType::UInt8),
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::TextArray(TextArray::Categorical8(Arc::new(arr))),
            ),
            4,
        );

        writer.write(tbl.clone()).unwrap();
        writer.finish().unwrap();

        let mut file = StdFile::create(&path).unwrap();
        for frame in writer.drain_all_frames() {
            use std::io::Write;
            file.write_all(&frame).unwrap();
        }
        file.flush().unwrap();
        drop(file);

        let buf = read_file_bytes(&path);
        assert!(
            buf.starts_with(ARROW_MAGIC_NUMBER_PADDED),
            "file must start with Arrow magic"
        );
        assert!(
            buf.ends_with(ARROW_MAGIC_NUMBER),
            "file must end with Arrow magic"
        );
        assert!(
            (buf.len() + 4 + 2) % 8 == 0,
            "Arrow IPC file must be a multiple of 8 bytes"
        );
    }

    #[test]
    fn test_ipc_file_write_read_std() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        // Create schema with a single Int32 (non-dictionary) column
        let schema = vec![Field {
            name: "col".to_string(),
            dtype: ArrowType::Int32,
            nullable: true,
            metadata: Default::default(),
        }];

        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::File, None);

        // Create a simple Int32 array with null mask
        let data = vec![10i32, 20, 30, 40];
        let mask = make_bitmask(&[true, false, true, true]);
        let arr = NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&data)),
            null_mask: Some(mask),
        }));

        let tbl = make_table(
            FieldArray::new(
                Field {
                    name: "col".to_string(),
                    dtype: ArrowType::Int32,
                    nullable: true,
                    metadata: Default::default(),
                },
                Array::NumericArray(arr),
            ),
            4,
        );

        writer.write(tbl.clone()).unwrap();
        writer.finish().unwrap();

        // Write to temp file
        let mut file = StdFile::create(&path).unwrap();
        for frame in writer.drain_all_frames() {
            use std::io::Write;
            file.write_all(&frame).unwrap();
        }
        file.flush().unwrap();
        drop(file);

        let buf = read_file_bytes(&path);
        println!("Written buffer:\n{:?}", buf);
        assert!(
            buf.starts_with(ARROW_MAGIC_NUMBER_PADDED),
            "file must start with Arrow magic"
        );
        assert!(
            buf.ends_with(ARROW_MAGIC_NUMBER),
            "file must end with Arrow magic"
        );
    }
}
