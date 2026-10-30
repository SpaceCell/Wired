// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod integration {
    use std::sync::Arc;

    use ::lightstream::arrow::message::org::apache::arrow::flatbuf as fb;
    use ::lightstream::enums::BufferChunkSize;
    use ::lightstream::enums::DecodeResult;
    use ::lightstream::enums::IPCMessageProtocol;
    use ::lightstream::models::decoders::ipc::ArrowIPCFrameDecoder;
    use ::lightstream::models::readers::ipc::file_table::FileTableReader;
    use ::lightstream::models::readers::ipc::table::TableReader;
    use ::lightstream::models::streams::disk::DiskByteStream;
    use ::lightstream::models::writers::ipc::table_stream::write_tables_to_stream;
    use ::lightstream::models::writers::ipc::table::TableWriter;
    use ::lightstream::traits::frame_decoder::FrameDecoder;
    use futures_util::stream::StreamExt;
    use minarrow::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
    use minarrow::*;
    use minarrow::{Array, TextArray, Vec64};

    /// Write tables to a file in Arrow stream format
    async fn write_stream_tables_to_file(
        file_path: &std::path::Path,
        tables: &[Table],
        schema: &[Field],
    ) -> std::io::Result<()> {
        let file = tokio::fs::File::create(file_path).await?;

        // Use the existing stream writer function to write to the file
        write_tables_to_stream::<_, Vec64<u8>>(
            file,
            tables,
            schema.to_vec(),
            IPCMessageProtocol::Stream,
        )
        .await?;

        Ok(())
    }

    pub fn vec64_to_vec(v: Vec64<String>) -> Vec<String> {
        v.into_iter().collect()
    }

    pub fn make_alternating_mask(len: usize) -> minarrow::Bitmask {
        let mut bm = minarrow::Bitmask::new_set_all(len, true);
        for i in (1..len).step_by(2) {
            bm.set_false(i);
        }
        bm
    }

    fn build_all_types_table(n: usize) -> Table {
        let int32 = NumericArray::Int32(Arc::new(IntegerArray::from_vec64(
            Vec64::from_slice(&(1..=n as i32).collect::<Vec<_>>()),
            None,
        )));
        let int64 = NumericArray::Int64(Arc::new(IntegerArray::from_vec64(
            Vec64::from_slice(&(100..(100 + n as i64)).collect::<Vec<_>>()),
            None,
        )));
        let uint32 = NumericArray::UInt32(Arc::new(IntegerArray::from_vec64(
            Vec64::from_slice(&(0..n as u32).collect::<Vec<_>>()),
            None,
        )));
        let uint64 = NumericArray::UInt64(Arc::new(IntegerArray::from_vec64(
            Vec64::from_slice(&(10..(10 + n as u64)).collect::<Vec<_>>()),
            None,
        )));
        let float32 = NumericArray::Float32(Arc::new(FloatArray::from_vec64(
            Vec64::from_slice(&(0..n).map(|i| i as f32 * 1.25 - 2.5).collect::<Vec<_>>()),
            None,
        )));
        let float64 = NumericArray::Float64(Arc::new(FloatArray::from_vec64(
            Vec64::from_slice(&(0..n).map(|i| i as f64 * 3.5 - 1.0).collect::<Vec<_>>()),
            None,
        )));
        let bools = BooleanArray::from_slice(&(0..n).map(|i| i % 2 == 0).collect::<Vec<_>>());

        let strs: Vec<String> = (0..n).map(|i| format!("str{i}")).collect();
        let str_refs: Vec<&str> = strs.iter().map(|s| &**s).collect();
        let string32 = StringArray::from_vec(str_refs, None);

        #[cfg(feature = "large_string")]
        let large_string = {
            let ls_refs: Vec<&str> = strs.iter().map(|s| &**s).collect();
            StringArray::from_vec(ls_refs, None)
        };

        let cat32 = {
            let cat_keys: Vec64<u32> =
                Vec64::from_slice(&(0..n as u32).map(|i| i % 3).collect::<Vec<_>>());
            let cat_values: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
            CategoricalArray::new(
                minarrow::Buffer::from(cat_keys),
                Vec64::from(cat_values.clone()),
                None,
            )
        };

        #[cfg(feature = "default_categorical_8")]
        let cat8_default = {
            let cat_keys: Vec64<u8> =
                Vec64::from_slice(&(0..n as u8).map(|i| (i % 3) as u8).collect::<Vec<_>>());
            let cat_values: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
            CategoricalArray::new(
                minarrow::Buffer::from(cat_keys),
                Vec64::from(cat_values.clone()),
                None,
            )
        };

        #[cfg(feature = "extended_categorical")]
        let cat_u8 = CategoricalArray::new(
            minarrow::Buffer::from(Vec64::from_slice(
                &(0..n as u8).map(|i| i % 3).collect::<Vec<_>>(),
            )),
            Vec64::from(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
            None,
        );
        #[cfg(feature = "extended_categorical")]
        let cat_u16 = CategoricalArray::new(
            minarrow::Buffer::from(Vec64::from_slice(
                &(0..n as u16)
                    .map(|i| (2 - (i % 3)) as u16)
                    .collect::<Vec<_>>(),
            )),
            Vec64::from(vec![
                "foo".to_string(),
                "bar".to_string(),
                "baz".to_string(),
            ]),
            None,
        );
        #[cfg(feature = "extended_categorical")]
        let cat_u64 = CategoricalArray::new(
            minarrow::Buffer::from(Vec64::from_slice(
                &(0..n as u64).map(|i| (i % 3) as u64).collect::<Vec<_>>(),
            )),
            Vec64::from(vec!["m".to_string(), "n".to_string(), "o".to_string()]),
            None,
        );

        #[allow(unused_mut)]
        let mut cols = vec![
            FieldArray::new(
                Field::new("int32", ArrowType::Int32, false, None),
                Array::NumericArray(int32),
            ),
            FieldArray::new(
                Field::new("int64", ArrowType::Int64, false, None),
                Array::NumericArray(int64),
            ),
            FieldArray::new(
                Field::new("uint32", ArrowType::UInt32, false, None),
                Array::NumericArray(uint32),
            ),
            FieldArray::new(
                Field::new("uint64", ArrowType::UInt64, false, None),
                Array::NumericArray(uint64),
            ),
            FieldArray::new(
                Field::new("float32", ArrowType::Float32, false, None),
                Array::NumericArray(float32),
            ),
            FieldArray::new(
                Field::new("float64", ArrowType::Float64, false, None),
                Array::NumericArray(float64),
            ),
            FieldArray::new(
                Field::new("bool", ArrowType::Boolean, false, None),
                Array::BooleanArray(Arc::new(bools)),
            ),
            FieldArray::new(
                Field::new("string", ArrowType::String, false, None),
                Array::TextArray(TextArray::String32(Arc::new(string32))),
            ),
        ];
        cols.push(FieldArray::new(
            Field::new(
                "cat32",
                ArrowType::Dictionary(CategoricalIndexType::UInt32),
                false,
                None,
            ),
            Array::TextArray(TextArray::Categorical32(Arc::new(cat32))),
        ));
        #[cfg(feature = "default_categorical_8")]
        cols.push(FieldArray::new(
            Field::new(
                "cat8",
                ArrowType::Dictionary(CategoricalIndexType::UInt8),
                false,
                None,
            ),
            Array::TextArray(TextArray::Categorical8(Arc::new(cat8_default))),
        ));
        #[cfg(feature = "large_string")]
        cols.push(FieldArray::new(
            Field::new("large_string", ArrowType::LargeString, false, None),
            Array::TextArray(TextArray::String64(Arc::new(large_string))),
        ));
        #[cfg(feature = "extended_categorical")]
        {
            cols.push(FieldArray::new(
                Field::new(
                    "cat_u8",
                    ArrowType::Dictionary(CategoricalIndexType::UInt8),
                    false,
                    None,
                ),
                Array::TextArray(TextArray::Categorical8(Arc::new(cat_u8))),
            ));
            cols.push(FieldArray::new(
                Field::new(
                    "cat_u16",
                    ArrowType::Dictionary(CategoricalIndexType::UInt16),
                    false,
                    None,
                ),
                Array::TextArray(TextArray::Categorical16(Arc::new(cat_u16))),
            ));
            cols.push(FieldArray::new(
                Field::new(
                    "cat_u64",
                    ArrowType::Dictionary(CategoricalIndexType::UInt64),
                    false,
                    None,
                ),
                Array::TextArray(TextArray::Categorical64(Arc::new(cat_u64))),
            ));
        }
        Table {
            cols,
            n_rows: n,
            name: "test".to_owned(),
            ..Default::default()
        }
    }

    fn dicts_for_table(table: &Table) -> Vec<(i64, Vec64<String>)> {
        table
            .cols
            .iter()
            .enumerate()
            .filter_map(|(i, col)| match &col.array {
                Array::TextArray(TextArray::Categorical32(arr)) => {
                    Some((i as i64, arr.unique_values.clone()))
                }
                #[cfg(feature = "default_categorical_8")]
                Array::TextArray(TextArray::Categorical8(arr)) => {
                    Some((i as i64, arr.unique_values.clone()))
                }
                #[cfg(feature = "extended_categorical")]
                Array::TextArray(TextArray::Categorical16(arr)) => {
                    Some((i as i64, arr.unique_values.clone()))
                }
                #[cfg(feature = "extended_categorical")]
                Array::TextArray(TextArray::Categorical64(arr)) => {
                    Some((i as i64, arr.unique_values.clone()))
                }
                _ => None,
            })
            .collect()
    }

    async fn roundtrip_ipc(mode: IPCMessageProtocol, n: usize) {
        use tempfile::tempdir;
        let table = build_all_types_table(n);
        let schema: Vec<Field> = table
            .cols
            .iter()
            .map(|c| c.field.as_ref().clone())
            .collect();

        // --- Write to disk buffer
        let dir = tempdir().unwrap();
        let path = dir.path().join("arrow_roundtrip_ipc.bin");
        {
            match mode {
                IPCMessageProtocol::File => {
                    let file = tokio::fs::File::create(&path).await.unwrap();
                    let mut writer = TableWriter::new(file, schema, mode, None).unwrap();
                    for (dict_id, unique) in dicts_for_table(&table) {
                        writer.register_dictionary(dict_id, unique.to_vec());
                    }
                    writer.write_all_tables(vec![table.clone()]).await.unwrap();
                }
                IPCMessageProtocol::Stream => {
                    // Use TableWriter consistently for both protocols
                    let file = tokio::fs::File::create(&path).await.unwrap();
                    let mut writer = TableWriter::new(file, schema, mode, None).unwrap();
                    for (dict_id, unique) in dicts_for_table(&table) {
                        writer.register_dictionary(dict_id, unique.to_vec());
                    }
                    writer.write_all_tables(vec![table.clone()]).await.unwrap();
                }
            }
        }

        // --- Read using appropriate reader for protocol
        let mut table2 = match mode {
            IPCMessageProtocol::File => {
                // Debug: check what's in the file format
                let file_data = std::fs::read(&path).unwrap();
                println!(
                    "FILE format file size: {}, first 16 bytes: {:02X?}",
                    file_data.len(),
                    &file_data[..16.min(file_data.len())]
                );

                let reader = FileTableReader::open(&path).unwrap();
                reader.read_batch(0).unwrap()
            }
            IPCMessageProtocol::Stream => {
                // Debug: check what's in the file
                let file_data = std::fs::read(&path).unwrap();
                println!(
                    "Stream file size: {}, first 16 bytes: {:02X?}",
                    file_data.len(),
                    &file_data[..16.min(file_data.len())]
                );

                let mut stream = DiskByteStream::open(&path, BufferChunkSize::Custom(128 * 1024))
                    .await
                    .unwrap();

                // Debug: try to read some data from the stream directly and examine frame parsing
                use futures_util::StreamExt;
                if let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(chunk) => {
                            println!(
                                "Stream first chunk: {} bytes, first 16: {:02X?}",
                                chunk.len(),
                                &chunk.as_ref()[..16.min(chunk.len())]
                            );

                            // Test frame decoder directly
                            let mut decoder = ArrowIPCFrameDecoder::<Vec64<u8>>::new(
                                IPCMessageProtocol::Stream,
                                None,
                            );
                            match decoder.decode(chunk.as_ref()) {
                                Ok(result) => {
                                    println!(
                                        "Frame decode result: {}",
                                        if let DecodeResult::Frame { .. } = &result {
                                            "frame produced"
                                        } else {
                                            "incomplete"
                                        }
                                    );

                                    // Examine the message header if we got a frame
                                    if let DecodeResult::Frame { frame, .. } = result {
                                        if !frame.message_range.is_empty() {
                                            let msg_bytes =
                                                &chunk.as_ref()[frame.message_range.clone()];
                                            match flatbuffers::root::<fb::Message>(msg_bytes) {
                                                Ok(af_msg) => {
                                                    println!(
                                                        "Message header type: {:?}",
                                                        af_msg.header_type()
                                                    );
                                                    println!(
                                                        "Message version: {:?}",
                                                        af_msg.version()
                                                    );
                                                }
                                                Err(e) => println!(
                                                    "Failed to parse flatbuffer message: {}",
                                                    e
                                                ),
                                            }
                                        } else {
                                            println!("Frame has empty message range");
                                        }
                                    }
                                }
                                Err(e) => println!("Frame decode error: {}", e),
                            }
                        }
                        Err(e) => println!("Stream error: {}", e),
                    }
                } else {
                    println!("Stream returned None immediately");
                }

                // Reset stream for actual reading
                let stream = DiskByteStream::open(&path, BufferChunkSize::Custom(128 * 1024))
                    .await
                    .unwrap();
                let mut reader =
                    TableReader::<Vec64<u8>>::new(stream, 128 * 1024, IPCMessageProtocol::Stream, None);

                match reader.next().await {
                    Some(Ok(table)) => table,
                    Some(Err(e)) => panic!("Reader error: {}", e),
                    None => panic!("Reader returned None - no data found"),
                }
            }
        };

        // Fix table name to match original for comparison
        table2.name = table.name.clone();

        // --- Data equality
        assert_eq!(table2.n_rows, table.n_rows);
        assert_eq!(table2.cols.len(), table.cols.len());
        for (a, b) in table.cols.iter().zip(table2.cols.iter()) {
            assert_eq!(a.field, b.field, "Field mismatch");
            match (&a.array, &b.array) {
                (Array::NumericArray(an), Array::NumericArray(bn)) => assert_eq!(an, bn),
                (Array::BooleanArray(an), Array::BooleanArray(bn)) => assert_eq!(an, bn),
                (Array::TextArray(at), Array::TextArray(bt)) => assert_eq!(at, bt),
                _ => panic!("Mismatched array types: {:?} vs {:?}", a.array, b.array),
            }
        }
    }

    /// In-memory stream roundtrip test (proper stream protocol test)
    async fn roundtrip_ipc_stream_memory(n_rows: usize) {
        use ::lightstream::models::readers::ipc::table::TableReader;
        use ::lightstream::models::writers::ipc::table_stream::TableStreamWriter;
        use futures_core::Stream;
        use std::pin::Pin;
        use std::task::{Context, Poll};
        use tokio::io::{AsyncRead, AsyncWriteExt, duplex};

        let table = build_all_types_table(n_rows);
        let schema: Vec<Field> = table
            .cols
            .iter()
            .map(|c| c.field.as_ref().clone())
            .collect();

        // Write to in-memory stream using TableStreamWriter
        let (mut tx, rx) = duplex(64 * 1024);

        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        for (dict_id, unique) in dicts_for_table(&table) {
            writer.register_dictionary(dict_id, unique.to_vec());
        }
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();

        // Collect frames and write to duplex
        let mut all_bytes = Vec::new();
        while let Some(frame) = writer.next_frame() {
            let frame_bytes = frame.unwrap();
            all_bytes.extend_from_slice(frame_bytes.as_ref());
        }
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        // Create a combined stream wrapper (like in working unit tests)
        struct Combined<R> {
            reader: R,
        }
        impl<R: AsyncRead + Unpin> Stream for Combined<R> {
            type Item = Result<Vec<u8>, std::io::Error>;
            fn poll_next(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                use tokio::io::AsyncReadExt;
                let mut buf = vec![0u8; 8192];
                let fut = self.reader.read(&mut buf);
                let mut fut = Box::pin(fut);
                match fut.as_mut().poll(cx) {
                    Poll::Ready(Ok(0)) => Poll::Ready(None),
                    Poll::Ready(Ok(n)) => {
                        buf.truncate(n);
                        Poll::Ready(Some(Ok(buf)))
                    }
                    Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                    Poll::Pending => Poll::Pending,
                }
            }
        }
        impl<R: AsyncRead + Unpin> AsyncRead for Combined<R> {
            fn poll_read(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut tokio::io::ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                Pin::new(&mut self.reader).poll_read(cx, buf)
            }
        }

        let combined = Combined { reader: rx };
        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let tables = reader.read_all_tables().await.unwrap();

        assert_eq!(tables.len(), 1);
        let mut table2 = tables.into_iter().next().unwrap();
        table2.name = table.name.clone(); // Fix name for comparison

        assert_eq!(table, table2);
    }

    /// In-memory stream roundtrip test with nulls (proper stream protocol test)
    async fn roundtrip_ipc_stream_memory_with_nulls(n_rows: usize) {
        use ::lightstream::models::readers::ipc::table::TableReader;
        use ::lightstream::models::writers::ipc::table_stream::TableStreamWriter;
        use futures_core::Stream;
        use std::pin::Pin;
        use std::task::{Context, Poll};
        use tokio::io::{AsyncRead, AsyncWriteExt, duplex};

        let table = build_all_types_table(n_rows); // Use regular table for now
        let schema: Vec<Field> = table
            .cols
            .iter()
            .map(|c| c.field.as_ref().clone())
            .collect();

        // Write to in-memory stream using TableStreamWriter
        let (mut tx, rx) = duplex(64 * 1024);

        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        for (dict_id, unique) in dicts_for_table(&table) {
            writer.register_dictionary(dict_id, unique.to_vec());
        }
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();

        // Collect frames and write to duplex
        let mut all_bytes = Vec::new();
        while let Some(frame) = writer.next_frame() {
            let frame_bytes = frame.unwrap();
            all_bytes.extend_from_slice(frame_bytes.as_ref());
        }
        tx.write_all(&all_bytes).await.unwrap();
        drop(tx);

        // Create a combined stream wrapper (like in working unit tests)
        struct Combined<R> {
            reader: R,
        }
        impl<R: AsyncRead + Unpin> Stream for Combined<R> {
            type Item = Result<Vec<u8>, std::io::Error>;
            fn poll_next(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                use tokio::io::AsyncReadExt;
                let mut buf = vec![0u8; 8192];
                let fut = self.reader.read(&mut buf);
                let mut fut = Box::pin(fut);
                match fut.as_mut().poll(cx) {
                    Poll::Ready(Ok(0)) => Poll::Ready(None),
                    Poll::Ready(Ok(n)) => {
                        buf.truncate(n);
                        Poll::Ready(Some(Ok(buf)))
                    }
                    Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                    Poll::Pending => Poll::Pending,
                }
            }
        }
        impl<R: AsyncRead + Unpin> AsyncRead for Combined<R> {
            fn poll_read(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut tokio::io::ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                Pin::new(&mut self.reader).poll_read(cx, buf)
            }
        }

        let combined = Combined { reader: rx };
        let reader: TableReader<Vec64<u8>> =
            TableReader::new(combined, 1024, IPCMessageProtocol::Stream, None);
        let tables = reader.read_all_tables().await.unwrap();

        assert_eq!(tables.len(), 1);
        let mut table2 = tables.into_iter().next().unwrap();
        table2.name = table.name.clone(); // Fix name for comparison

        assert_eq!(table, table2);
    }

    #[tokio::test]
    async fn test_roundtrip_stream_6rows() {
        roundtrip_ipc_stream_memory(6).await;
    }

    #[tokio::test]
    async fn test_roundtrip_stream_1row() {
        roundtrip_ipc_stream_memory(1).await;
    }

    #[tokio::test]
    async fn test_roundtrip_file_6rows() {
        roundtrip_ipc(IPCMessageProtocol::File, 6).await;
    }

    #[tokio::test]
    async fn test_roundtrip_file_1row() {
        roundtrip_ipc(IPCMessageProtocol::File, 1).await;
    }

    #[tokio::test]
    async fn test_roundtrip_stream_nulls() {
        roundtrip_ipc_stream_memory_with_nulls(7).await;
    }

    #[tokio::test]
    async fn test_roundtrip_file_nulls() {
        roundtrip_ipc_with_nulls(IPCMessageProtocol::File, 7).await;
    }

    async fn roundtrip_ipc_with_nulls(fmt: IPCMessageProtocol, n: usize) {
        use tempfile::tempdir;

        // -------- build a table where EVERY column has a null‑mask ----------
        let mut cols = Vec::new();

        // numeric with mask
        let data_i32 = Vec64::from_slice(&(0..n as i32).collect::<Vec<_>>());
        let arr_i32 = IntegerArray::from_vec64(data_i32, Some(make_alternating_mask(n)));
        cols.push(FieldArray::new(
            Field::new("num32n", ArrowType::Int32, true, None),
            Array::NumericArray(NumericArray::Int32(Arc::new(arr_i32))),
        ));

        // bool with mask
        let bools = BooleanArray::from_slice(&(0..n).map(|i| i % 3 == 0).collect::<Vec<_>>());
        let mut bools_masked = bools.clone();
        bools_masked.null_mask = Some(make_alternating_mask(n));
        cols.push(FieldArray::new(
            Field::new("booln", ArrowType::Boolean, true, None),
            Array::BooleanArray(Arc::new(bools_masked)),
        ));

        // utf8 with mask
        let strings: Vec<String> = (0..n).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = strings.iter().map(|s| &**s).collect();
        let str_arr = StringArray::from_vec(refs, Some(make_alternating_mask(n)));
        cols.push(FieldArray::new(
            Field::new("strn", ArrowType::String, true, None),
            Array::TextArray(TextArray::String32(Arc::new(str_arr))),
        ));

        // dictionary with mask
        let uniqs: Vec<String> = vec!["A".into(), "B".into()];
        #[cfg(not(feature = "default_categorical_8"))]
        {
            let keys: Vec64<u32> =
                Vec64::from_slice(&(0..n as u32).map(|i| i % 2).collect::<Vec<_>>());
            let cat_arr = CategoricalArray::new(
                minarrow::Buffer::from(keys),
                Vec64::from(uniqs.clone()),
                Some(make_alternating_mask(n)),
            );
            cols.push(FieldArray::new(
                Field::new(
                    "catn",
                    ArrowType::Dictionary(CategoricalIndexType::UInt32),
                    true,
                    None,
                ),
                Array::TextArray(TextArray::Categorical32(Arc::new(cat_arr))),
            ));
        }
        #[cfg(feature = "default_categorical_8")]
        {
            let keys: Vec64<u8> =
                Vec64::from_slice(&(0..n as u8).map(|i| (i % 2) as u8).collect::<Vec<_>>());
            let cat_arr = CategoricalArray::new(
                minarrow::Buffer::from(keys),
                Vec64::from(uniqs.clone()),
                Some(make_alternating_mask(n)),
            );
            cols.push(FieldArray::new(
                Field::new(
                    "catn",
                    ArrowType::Dictionary(CategoricalIndexType::UInt8),
                    true,
                    None,
                ),
                Array::TextArray(TextArray::Categorical8(Arc::new(cat_arr))),
            ));
        }

        let table = Table {
            cols,
            n_rows: n,
            name: "nulls".into(),
            ..Default::default()
        };

        // ---------- write to file -------------
        let dir = tempdir().unwrap();
        let path = dir.path().join("arrow_nulls_ipc.bin");
        let schema: Vec<Field> = table
            .cols
            .iter()
            .map(|c| c.field.as_ref().clone())
            .collect();
        {
            match fmt {
                IPCMessageProtocol::File => {
                    let file = tokio::fs::File::create(&path).await.unwrap();
                    let mut wr = TableWriter::new(file, schema, fmt, None).unwrap();
                    wr.register_dictionary(3, uniqs); // cat column is 3rd here
                    wr.write_all_tables(vec![table.clone()]).await.unwrap();
                }
                IPCMessageProtocol::Stream => {
                    let mut table_with_dict = table.clone();
                    // Register dictionary in the table's categorical column
                    #[cfg(not(feature = "default_categorical_8"))]
                    if let Array::TextArray(TextArray::Categorical32(cat_arr)) =
                        &mut table_with_dict.cols[3].array
                    {
                        let cat_arr = Arc::make_mut(cat_arr);
                        cat_arr.unique_values = Vec64::from(uniqs);
                    }
                    #[cfg(feature = "default_categorical_8")]
                    if let Array::TextArray(TextArray::Categorical8(cat_arr)) =
                        &mut table_with_dict.cols[3].array
                    {
                        let cat_arr = Arc::make_mut(cat_arr);
                        cat_arr.unique_values = Vec64::from(uniqs);
                    }
                    write_stream_tables_to_file(&path, &[table_with_dict], &schema)
                        .await
                        .unwrap();
                }
            }
        }

        // ---------- read using appropriate reader for protocol ----------
        let mut roundtripped = match fmt {
            IPCMessageProtocol::File => {
                let reader = FileTableReader::open(&path).unwrap();
                reader.read_batch(0).unwrap()
            }
            IPCMessageProtocol::Stream => {
                let stream = DiskByteStream::open(&path, BufferChunkSize::Custom(128 * 1024))
                    .await
                    .unwrap();
                let mut reader =
                    TableReader::<Vec64<u8>>::new(stream, 128 * 1024, IPCMessageProtocol::Stream, None);
                reader.next().await.unwrap().unwrap()
            }
        };

        // Fix table name to match original for comparison
        roundtripped.name = table.name.clone();
        assert_eq!(roundtripped, table); // derives PartialEq so includes masks
    }

    /// Every `Dt` variant that the IPC schema supports, covering each clock
    /// unit and timezone form.
    #[cfg(feature = "datetime")]
    fn ipc_temporal_dtypes() -> Vec<Dt> {
        let units = [
            TimeUnit::Seconds,
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ];
        let mut dtypes = vec![Dt::Date32, Dt::Date64];
        for unit in units {
            dtypes.push(Dt::Time32(unit));
            dtypes.push(Dt::Time64(unit));
            dtypes.push(Dt::Duration64(unit));
            dtypes.push(Dt::Timestamp(unit, None));
            dtypes.push(Dt::Timestamp(unit, Some("UTC".to_string())));
            dtypes.push(Dt::Timestamp(unit, Some("America/New_York".to_string())));
        }
        dtypes
    }

    /// One nullable column of the given temporal type, with a null at row 1.
    #[cfg(feature = "datetime")]
    fn temporal_table(dtype: &Dt) -> Table {
        let mask = {
            let mut m = Bitmask::new_set_all(4, true);
            m.set_false(1);
            m
        };
        let array = match dtype {
            Dt::Date32 | Dt::Time32(_) | Dt::Duration32(_) => {
                Array::from_datetime_i32(DatetimeArray::from_vec64(
                    Vec64::from_slice(&[19_000, 0, -5, 86_399]),
                    Some(mask),
                    dtype.clone(),
                ))
            }
            _ => Array::from_datetime_i64(DatetimeArray::from_vec64(
                Vec64::from_slice(&[1_782_999_000_000_000_000, 0, -1, 1]),
                Some(mask),
                dtype.clone(),
            )),
        };
        let col = FieldArray::new(
            Field::new("t", ArrowType::DateTime(dtype.clone()), true, None),
            array,
        );
        Table::new("temporal".to_string(), Some(vec![col]))
    }

    /// Asserts that the decoded column keeps the field type and the array's
    /// own `Dt`, including unit and timezone.
    #[cfg(feature = "datetime")]
    fn assert_temporal_roundtrip(got: &Table, want: &Table, dtype: &Dt, via: &str) {
        assert_eq!(got.n_rows, want.n_rows, "{dtype:?} via {via}");
        assert_eq!(
            got.cols[0].field.dtype,
            ArrowType::DateTime(dtype.clone()),
            "{dtype:?} field dtype via {via}"
        );
        let got_dtype = match &got.cols[0].array {
            Array::TemporalArray(TemporalArray::Datetime32(a)) => a.dtype.clone(),
            Array::TemporalArray(TemporalArray::Datetime64(a)) => a.dtype.clone(),
            other => panic!("{dtype:?} via {via} decoded as {other:?}"),
        };
        assert_eq!(&got_dtype, dtype, "{dtype:?} array dtype via {via}");
        assert_eq!(got.cols[0].array, want.cols[0].array, "{dtype:?} values via {via}");
    }

    #[cfg(feature = "datetime")]
    #[tokio::test]
    async fn test_temporal_dtypes_roundtrip_file_and_stream() {
        use tempfile::tempdir;
        for dtype in ipc_temporal_dtypes() {
            let table = temporal_table(&dtype);
            let schema: Vec<Field> = table.cols.iter().map(|c| c.field.as_ref().clone()).collect();
            let dir = tempdir().unwrap();

            // File protocol read through the file reader and the mmap reader.
            let file_path = dir.path().join("temporal_file.arrow");
            {
                let file = tokio::fs::File::create(&file_path).await.unwrap();
                let mut writer =
                    TableWriter::new(file, schema.clone(), IPCMessageProtocol::File, None).unwrap();
                writer.write_all_tables(vec![table.clone()]).await.unwrap();
            }
            let got = FileTableReader::open(&file_path).unwrap().read_batch(0).unwrap();
            assert_temporal_roundtrip(&got, &table, &dtype, "file");
            #[cfg(feature = "mmap")]
            {
                use ::lightstream::models::readers::ipc::mmap_table::MmapTableReader;
                let got = MmapTableReader::open(&file_path).unwrap().read_batch(0).unwrap();
                assert_temporal_roundtrip(&got, &table, &dtype, "mmap");
            }

            // Stream protocol.
            let stream_path = dir.path().join("temporal_stream.arrow");
            {
                let file = tokio::fs::File::create(&stream_path).await.unwrap();
                let mut writer =
                    TableWriter::new(file, schema, IPCMessageProtocol::Stream, None).unwrap();
                writer.write_all_tables(vec![table.clone()]).await.unwrap();
            }
            let stream = DiskByteStream::open(&stream_path, BufferChunkSize::Custom(128 * 1024))
                .await
                .unwrap();
            let mut reader = TableReader::<Vec64<u8>>::new(
                stream,
                128 * 1024,
                IPCMessageProtocol::Stream,
                None,
            );
            let got = match reader.next().await {
                Some(Ok(t)) => t,
                other => panic!("{dtype:?} stream read: {other:?}"),
            };
            assert_temporal_roundtrip(&got, &table, &dtype, "stream");
        }
    }

    /// Duration32 and Interval have no IPC schema encoding.
    #[cfg(feature = "datetime")]
    #[tokio::test]
    async fn test_unsupported_temporal_dtypes_are_rejected_on_write() {
        use tempfile::tempdir;
        let interval = Table::new(
            "interval".to_string(),
            Some(vec![FieldArray::new(
                Field::new(
                    "t",
                    ArrowType::DateTime(Dt::Interval(IntervalUnit::DaysTime)),
                    false,
                    None,
                ),
                Array::from_int32(IntegerArray::from_vec64(Vec64::from_slice(&[1, 2]), None)),
            )]),
        );
        for table in [temporal_table(&Dt::Duration32(TimeUnit::Seconds)), interval] {
            let schema: Vec<Field> = table.cols.iter().map(|c| c.field.as_ref().clone()).collect();
            let dir = tempdir().unwrap();
            let file = tokio::fs::File::create(dir.path().join("rejected.arrow"))
                .await
                .unwrap();
            let outcome = match TableWriter::new(file, schema, IPCMessageProtocol::Stream, None) {
                Err(e) => Err(e),
                Ok(mut writer) => writer.write_all_tables(vec![table.clone()]).await,
            };
            assert!(outcome.is_err(), "{:?} should be rejected", table.cols[0].field.dtype);
        }
    }

    #[tokio::test]
    async fn test_file_roundtrip_via_disk() {
        use tempfile::tempdir;
        let n = 5;
        let table = build_all_types_table(n);
        let schema: Vec<Field> = table
            .cols
            .iter()
            .map(|c| c.field.as_ref().clone())
            .collect();

        // ---------- create a temp file ----------
        let dir = tempdir().unwrap();
        let p = dir.path().join("arrow.bin");
        {
            let file = tokio::fs::File::create(&p).await.unwrap();
            let mut writer = TableWriter::new(file, schema, IPCMessageProtocol::File, None).unwrap();
            for (id, u) in dicts_for_table(&table) {
                writer.register_dictionary(id, vec64_to_vec(u));
            }
            writer.write_all_tables(vec![table.clone()]).await.unwrap();
        }

        // ---------- read file via FileTableReader ----------
        let reader = FileTableReader::open(&p).unwrap();
        let mut roundtripped = reader.read_batch(0).unwrap();

        // Fix table name to match original for comparison
        roundtripped.name = table.name.clone();

        assert_eq!(roundtripped, table);
    }
}
