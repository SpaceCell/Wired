# Changelog

Notable changes are recorded from 0.5.0 onward.

## Unreleased

### Enhancements

- Added `SyncTableReader`, a blocking Arrow IPC stream reader for `std::io::Read` sources. It uses the same decoding implementation and `DecodeLimits` as `TableReader` and returns tables.
- Added `SyncUdsTableReader` and `SyncUdsTableWriter` under the `uds` feature for persistent, high-volume feeds over Unix domain sockets. They run on dedicated threads without an async runtime. Both provide `connect`, `accept` and `from_stream` and share the wire format used by `UdsTableReader` and `UdsTableWriter`, allowing synchronous and asynchronous peers to interoperate.
- Added `SyncTableSink`, a blocking Arrow IPC table sink for `std::io::Write` destinations and the synchronous counterpart to `GTableSink`. It encodes Stream protocol batches using a reusable buffer. Both `SyncTableWriter` and `SyncUdsTableWriter` write through this sink.
- Added `SyncWebSocketTableReader` and `SyncWebSocketTableWriter` under the `websocket` feature for high-volume feeds over plaintext WebSocket. They run on dedicated threads without an async runtime, wrap `SyncTableReader` and `SyncTableSink`, and provide `connect`, `accept` and the `from_halves` constructors of the async types, with the same `DecodeLimits` and `Compression` options. The wire format and message boundaries match `WebSocketTableReader` and `WebSocketTableWriter`, allowing synchronous and asynchronous peers to interoperate in both roles.
- Added `SyncWebSocketTransport`, with `connect`, `bind` and `accept` over `std::net`, and the blocking byte stream adapters `SyncWsRead` and `SyncWsWrite`. `SyncWsRead::read_message` reads whole text or binary messages, joins fragments and answers pings, which suits record feeds that don't use the table reader.
- Added `WsFrameDecoder` and `WsFrameEncoder`, which hold the read-side and write-side WebSocket protocol shared by `WsRead`, `WsWrite`, `SyncWsRead` and `SyncWsWrite`. Neither type performs any IO, and the adapters perform the reads and writes. `WsRead` and `WsWrite` keep their API and behaviour.

### Breaking Changes

- Renamed `TableStreamDecoder` to `TableDecoder`. Decoding is now synchronous, with readers responsible for I/O. `TableDecoder::new` no longer accepts a source. Readers fill `spare_uninit`, commit bytes with `advance` and call `decode_next`, which returns `TableDecodeResult`.
- Moved the `Stream` implementation and `read_keyed` from the decoder to `TableReader`, which owns the `AsyncRead` source.
- Renamed `TableStreamEncoder` to `TableEncoder`.
- Renamed `models::decoders::ipc::table_stream_decoder` to `models::decoders::ipc::table_decoder`.
- Renamed `models::encoders::ipc::table_stream` to `models::encoders::ipc::table_encoder`.

### Fixed

- `TableStreamWriter::finish` now writes the Stream protocol end-of-stream marker when no tables have been written. Empty streams previously didn't produce any bytes, which `TableReader` rejected as truncated.

## v0.8.0 27 Oct 2026

### Enhancements
- Added object storage behind the `aws`, `gcp` and `azure` features. File readers and writers, and the Python `read` and `write` functions, accept object storage URLs in place of paths.
- Supporting the object storage, Lightstream now supports partitioning for its chunked writers, and nested partition directories in the chunked readers.
- Lightstream now includes support for WebAssembly targets.

### Breaking
- Chunked writer constructors now require specifying a `partition` argument as `Option<PartitionType>`.
- The `default_categorical_8` feature flag now pulls in the `Categorical32` enum arm, as per minarrow 0.19.0.

### Maintenance
- Minarrow has been upgraded to 0.19.0. lightstream-py pins minarrow and minarrow-pyo3 at 0.19.0.

## v0.7.1 1 Oct 2026

### Maintenance
- Compatibility with the 2026-09-30 Rust nightly, where `Vec::drain` on custom allocators requires the new `allocator_ext` feature (due to Rust planning to stabilise a lot of the allocator trait in the upcoming stable release).
Therefore, `Vec64` stream buffers now use `Vec64::delete_range` to avoid the feature gate.
- vec64 upgraded to 0.5.3.
- Minarrow upgraded to 0.18.3. The Python package pins minarrow and minarrow-pyo3 at 0.18.3.

## v0.7.0 19 Sep 2026

### Enhancements

- New `decimal` feature forwarding minarrow's `decimal` feature. Decimal32, Decimal64 and Decimal128 columns are supported in Arrow IPC and in Parquet, where they map to the DECIMAL logical type over INT32, INT64 and FIXED_LEN_BYTE_ARRAY.
- Improved Windows + Mac compatibility
- Added Windows + Mac + Python CI
- Parquet writer improvements:
  - The Parquet writer follows the Parquet value layout for nullable columns.
  - Several improvements in categorical, time, and date roundtripping.
  - Files with nulls written by earlier releases do not read back under this release.
  - Replaced packing for Duration and Interval columns with `UnsupportedType`

### Fixed

- The Parquet reader failed with `UnexpectedEof` on files written by pyarrow. It now reads DataPageV1 pages with compressed levels, `RLE` booleans, dictionary-encoded columns of any physical type, and files with several row groups.

### Maintenance
- Minarrow upgraded to 0.18.2 and vec64 0.5.1.
- The Python package pins minarrow and minarrow-pyo3 at 0.18.2.

## 0.6.1

### Changed

- minarrow version bump to 0.17.0, vec64 0.5.0, arrow 59.2.0 and polars 0.55.2.

## 0.6.0

### Changed

- Table writer entry points take `impl Into<TableV>`, so either owned `Table` or zero-copy `TableV`iew passes without a `.into()` conversion at the call site.

### Fixed

- `TableSink` kept only the most recent frame when several tables were admitted before a flush, corrupting streams driven through `Sink::feed` or `send_all`.

## 0.5.0

Initial public release.
