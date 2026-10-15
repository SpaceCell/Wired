# lightstream

Move Arrow tables between processes, services and storage from Python without adding a gRPC stack or writing transport-specific framing.

`lightstream` provides one streaming API across files, memory maps, sockets and network transports. Readers return [`minarrow`](https://github.com/pbower/minarrow) objects and implement the Arrow PyCapsule protocol, allowing PyArrow, Polars, DuckDB and other Arrow-compatible libraries to consume data **straight off the wire** without an intermediate conversion.

## Installation

```bash
pip install lightstream-io
```

Unix-domain sockets and memory-mapped reads are available on Linux and macOS.
Windows uses buffered file reads; `uds://` and explicit `mmap=True` requests are unsupported.

## Usage

Everything is `read` or `write`.

The URI selects the transport, `protocol` selects the wire framing, and file extensions select the storage format.

```python
import lightstream as ls
```

### Files

Read Arrow IPC, Parquet, CSV and JSON through the same interface.

```python
# Read an entire dataset.
table = ls.read("quotes.arrow").read_all()

# Stream batches without loading the complete dataset into memory.
for batch in ls.read("large.parquet"):
    process(batch)

# Write any Arrow-compatible object.
with ls.write("output.parquet", compression="zstd") as writer:
    writer.write(table)
```

Arrow IPC readers use memory mapping and out-of-core routing where appropriate.

Readers yield `minarrow.Table` batches. `read_all()` returns a `Table` when the result is contiguous or a `ChunkedTable` when multiple chunks must be preserved.

### Object storage

Builds with the `object_storage` feature accept object URLs wherever a file path is accepted. An object prefix with chunk objects beneath it reads in the same way as a chunk directory.

```python
ls.register_options("s3://telemetry", {"aws_region": "eu-west-2"})

table = ls.read("s3://telemetry/trades/2026-10-14.parquet").read_all()

with ls.write("s3://telemetry/trades/summary.arrow") as writer:
    writer.write(table)
```

| Store                    | URL                               | Feature          |
| ------------------------ | --------------------------------- | ---------------- |
| In-process memory        | `memory:///path`                  | `object_storage` |
| Amazon S3 and compatible | `s3://bucket/path`                | `aws`            |
| Google Cloud Storage     | `gs://bucket/path`                | `gcp`            |
| Azure Blob Storage       | `az://container/path`, `abfss://` | `azure`          |

Credentials come from each backend's standard environment variables. `register_options` sets backend options and the upload settings `part_size` and `max_concurrent_parts` for every URL beneath a prefix. Written objects become visible in their store when the writer closes. Memory-mapped reads apply to local files only.

### Network transports

The same interface streams raw Arrow IPC over TCP, Unix domain sockets, WebSocket, HTTP, QUIC, WebTransport and standard I/O.

```python
for batch in ls.read("tcp://feed.example.com:9000"):
    process(batch)
```

Supported URI schemes include:

| Transport             | URI                     |
| --------------------- | ----------------------- |
| TCP                   | `tcp://host:port`       |
| Unix domain socket    | `uds:///path/to/socket` |
| WebSocket             | `ws://host/path`        |
| Secure WebSocket      | `wss://host/path`       |
| HTTP                  | `http://host/path`      |
| HTTPS                 | `https://host/path`     |
| QUIC                  | `quic://host:port`      |
| WebTransport          | `wt://host/path`        |
| Standard input/output | `stdio://`              |

### Accepting connections

Either endpoint can accept the connection.

Set `accept=True` to bind the endpoint on first use and block until the connecting peer arrives.

```python
writer = ls.write(
    "uds:///tmp/feed.sock",
    accept=True,
)

writer.write(table)
writer.close()
```

The listener remains available for the lifetime of the process. Connections arriving while a serving loop is active wait in the listener backlog, allowing the accepting endpoint to operate as a persistent server.

### TLS

QUIC and WebTransport include TLS at the protocol level. The accepting endpoint presents a PEM certificate and private key, while the connecting endpoint verifies the certificate using PEM roots.

```python
writer = ls.write(
    "quic://0.0.0.0:4433",
    accept=True,
    tls_cert="server.pem",
    tls_key="server-key.pem",
)

for batch in ls.read(
    "quic://feed.example.com:4433",
    tls_ca="roots.pem",
):
    process(batch)
```

Secure WebSocket and HTTPS transports use their corresponding TLS-enabled URI schemes.

## Arrow interoperability

Every reader implements the Arrow PyCapsule stream protocol.

Arrow-compatible libraries can therefore consume a Lightstream reader directly.

```python
import duckdb
import lightstream as ls

reader = ls.read("quotes.arrow")

result = duckdb.sql(
    "SELECT SUM(qty) FROM reader"
)
```

The same reader can be passed to libraries such as Polars without first converting its batches into Python objects.

## Lightstream protocol

The Lightstream protocol multiplexes named Arrow tables and opaque messages over one connection.

Opaque payloads can carry formats such as Protobuf or MessagePack, while table channels retain their Arrow schema and batch representation.

```python
reader = ls.read(
    "uds:///tmp/feed.sock",
    protocol="lightstream",
)

reader.register_table(
    "quotes",
    representative_table,
)
reader.register_message("health")

for frame in reader:
    if frame.is_table():
        on_quotes(frame.table)
    else:
        on_health(frame.payload)
```

`frame.table` is a `minarrow.Table`. Message payloads are returned as `bytes`.

Both peers must register compatible channels before exchanging frames.

The protocol is transport-independent, so the same table and message definitions can be used over TCP, Unix domain sockets, WebSocket, HTTP, QUIC or WebTransport.

## Standard I/O pipelines

The standard I/O transport can stream Arrow IPC or line-oriented text.

Set `format="csv"` to write CSV records or `format="json"` to write NDJSON. Text emitted by another process is decoded back into table batches on read.

This allows Unix tools, command-line programs and agents to participate directly in a table pipeline.

```bash
python examples/run-example.py sed
python examples/run-example.py jq
python examples/run-example.py claude
python examples/run-example.py sql
```

The pipeline examples demonstrate:

* `sed` rewriting rows in flight
* `jq` filtering NDJSON and returning CSV
* an agent diagnosing invalid batches
* rolling DuckDB SQL over a live stream

## Transport examples

Worked examples are provided under `examples/transport/`.

Each transport includes:

* a Python server
* a Rust server
* a Python client
* the same million-row input table

Use the example router to select the transport and backend:

```bash
# Python backend over TCP.
python examples/run-example.py tcp

# Rust backend over QUIC.
python examples/run-example.py quic --rust
```

Available transports are:

```text
tcp
uds
ws
wss
http
https
quic
wt
stdio
```

TLS examples generate a private root certificate and a server certificate for local execution.

## Building from source

Lightstream currently requires the Rust nightly toolchain. The repository selects the required toolchain automatically through `rust-toolchain.toml`.

```bash
pip install maturin
maturin develop
pytest tests/
```

Optional features are enabled at build time, for example `maturin develop --features object_storage` or `maturin develop --features aws,gcp,azure`.

## Licence

Copyright © 2025–2026 Peter Garfield Bower.

Licensed under the Mozilla Public License 2.0. See `LICENSE` for the standard terms.

See [MPL 2.0 FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/) if you are unfamiliar with this open-source license.

## Affiliation notice

Lightstream is not affiliated with Apache Arrow or the Apache Software Foundation.

It implements public Arrow formats through Minarrow and interoperates with the Arrow ecosystem through the Arrow PyCapsule protocol.

`lightstream` is maintained by [SpaceCell](https://spacecell.com) and forms part of its open-source foundation for high-performance data computing.
