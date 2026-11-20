# Copyright Peter G. Bower 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The accepting peer role on wire sources and targets.

Each test pairs an accepting end with a connecting end, so both sides
run pure lightstream with no relay between them. The first accepting
call binds the endpoint and holds it for the life of the process, so
the connecting side only retries across server startup. A retry never
steals the accept, because a refused connection consumes nothing.
"""

import sys

import socket
import threading
import time

import lightstream as ls
import pyarrow as pa
import pytest


def sample_table(offset=0):
    return pa.table(
        {
            "sym": ["A", "B", "C"],
            "px": [1.5 + offset, 2.5 + offset, 3.5 + offset],
            "qty": [10 + offset, 20 + offset, 30 + offset],
        }
    )


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def with_retries(action, deadline=30.0):
    """Retries a connecting action until the accepting peer has bound."""
    last = None
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            return action()
        except ls.LightstreamError as exc:
            last = exc
            time.sleep(0.05)
    raise last


def run_accepting_writer(uri):
    def serve():
        writer = ls.write(uri, accept=True)
        writer.write(sample_table(0))
        writer.write(sample_table(100))
        writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    reader = with_retries(lambda: ls.read(uri))
    batches = list(reader)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert len(batches) == 2
    assert pa.table(batches[0]).to_pydict() == sample_table(0).to_pydict()
    assert pa.table(batches[1]).to_pydict() == sample_table(100).to_pydict()


def run_accepting_reader(uri):
    result = {}

    def serve():
        reader = ls.read(uri, accept=True)
        result["batches"] = list(reader)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    writer = ls.write(uri)
    with_retries(lambda: writer.write(sample_table(0)))
    writer.write(sample_table(100))
    writer.close()
    thread.join(timeout=30)
    assert not thread.is_alive()
    batches = result["batches"]
    assert len(batches) == 2
    assert pa.table(batches[0]).to_pydict() == sample_table(0).to_pydict()
    assert pa.table(batches[1]).to_pydict() == sample_table(100).to_pydict()


def test_tcp_arrow_accepting_writer():
    run_accepting_writer(f"tcp://127.0.0.1:{free_port()}")


def test_tcp_arrow_accepting_reader():
    run_accepting_reader(f"tcp://127.0.0.1:{free_port()}")


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-domain sockets are Unix-only")
def test_uds_arrow_accepting_writer(tmp_path):
    run_accepting_writer(f"uds://{tmp_path / 'writer.sock'}")


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-domain sockets are Unix-only")
def test_uds_arrow_accepting_reader(tmp_path):
    run_accepting_reader(f"uds://{tmp_path / 'reader.sock'}")


def test_ws_arrow_accepting_writer():
    run_accepting_writer(f"ws://127.0.0.1:{free_port()}")


def test_ws_arrow_accepting_reader():
    run_accepting_reader(f"ws://127.0.0.1:{free_port()}")


def test_http_arrow_accepting_writer():
    run_accepting_writer(f"http://127.0.0.1:{free_port()}/feed")


def test_http_arrow_accepting_reader():
    run_accepting_reader(f"http://127.0.0.1:{free_port()}/ingest")


def read_path(uri, deadline=30.0, **options):
    """Reads every batch an accepting writer sends on a URL path.

    Retries until the writer has bound the path. A request for a path
    that is not yet bound is refused without being served, therefore a
    retry never consumes another path's connection.
    """
    end = time.monotonic() + deadline
    while True:
        try:
            batches = list(ls.read(uri, **options))
            if batches:
                return batches
        except ls.LightstreamError:
            if time.monotonic() > end:
                raise
        assert time.monotonic() < end, f"no batches arrived from {uri}"
        time.sleep(0.05)


def run_accepting_writers_on_two_paths(base, **server_options):
    """Serves a different table on `/prices` and `/trades` of one authority."""

    def serve(path, offset):
        writer = ls.write(f"{base}{path}", accept=True, **server_options)
        writer.write(sample_table(offset))
        writer.close()

    threads = [
        threading.Thread(target=serve, args=(path, offset), daemon=True)
        for path, offset in [("/prices", 0), ("/trades", 100)]
    ]
    for thread in threads:
        thread.start()
    return threads


def assert_two_paths_served(base, threads, **client_options):
    trades = read_path(f"{base}/trades", **client_options)
    prices = read_path(f"{base}/prices", **client_options)
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert [pa.table(b).to_pydict() for b in prices] == [sample_table(0).to_pydict()]
    assert [pa.table(b).to_pydict() for b in trades] == [sample_table(100).to_pydict()]


def test_ws_accepting_writers_on_two_paths_serve_their_own_tables():
    base = f"ws://127.0.0.1:{free_port()}"
    threads = run_accepting_writers_on_two_paths(base)
    assert_two_paths_served(base, threads)


def test_http_accepting_writers_on_two_paths_serve_their_own_tables():
    base = f"http://127.0.0.1:{free_port()}"
    threads = run_accepting_writers_on_two_paths(base)
    assert_two_paths_served(base, threads)


def test_ws_accepting_readers_on_two_paths_receive_their_own_tables():
    base = f"ws://127.0.0.1:{free_port()}"
    result = {}

    def serve(path):
        result[path] = list(ls.read(f"{base}{path}", accept=True))

    threads = [
        threading.Thread(target=serve, args=(path,), daemon=True)
        for path in ["/prices", "/trades"]
    ]
    for thread in threads:
        thread.start()

    # A write to a path that is not yet bound is refused, and the retry
    # opens a new connection.
    for path, offset in [("/trades", 100), ("/prices", 0)]:
        writer = ls.write(f"{base}{path}")
        with_retries(lambda: writer.write(sample_table(offset)))
        writer.close()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert [pa.table(b).to_pydict() for b in result["/prices"]] == [sample_table(0).to_pydict()]
    assert [pa.table(b).to_pydict() for b in result["/trades"]] == [
        sample_table(100).to_pydict()
    ]


def test_tcp_lightstream_accepting_reader():
    uri = f"tcp://127.0.0.1:{free_port()}"
    result = {}

    def serve():
        reader = ls.read(uri, protocol="lightstream", accept=True)
        reader.register_table("quotes", sample_table())
        reader.register_message("health")
        result["frames"] = list(reader)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    writer = with_retries(lambda: ls.write(uri, protocol="lightstream"))
    writer.register_table("quotes", sample_table())
    writer.register_message("health")
    writer.write(sample_table(0), name="quotes")
    writer.write_message("health", b"\x07")
    writer.close()
    thread.join(timeout=30)
    assert not thread.is_alive()
    frames = result["frames"]
    assert [f.is_table() for f in frames] == [True, False]
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()
    assert frames[1].payload == b"\x07"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-domain sockets are Unix-only")
def test_uds_lightstream_accepting_writer(tmp_path):
    uri = f"uds://{tmp_path / 'tlv.sock'}"

    def serve():
        writer = ls.write(uri, protocol="lightstream", accept=True)
        writer.register_table("quotes", sample_table())
        writer.write(sample_table(0), name="quotes")
        writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    reader = with_retries(lambda: ls.read(uri, protocol="lightstream"))
    reader.register_table("quotes", sample_table())
    frames = list(reader)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert len(frames) == 1
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()


def test_ws_lightstream_accepting_reader():
    uri = f"ws://127.0.0.1:{free_port()}"
    result = {}

    def serve():
        reader = ls.read(uri, protocol="lightstream", accept=True)
        reader.register_table("quotes", sample_table())
        reader.register_message("health")
        result["frames"] = list(reader)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    writer = with_retries(lambda: ls.write(uri, protocol="lightstream"))
    writer.register_table("quotes", sample_table())
    writer.register_message("health")
    writer.write(sample_table(0), name="quotes")
    writer.write_message("health", b"\x07")
    writer.close()
    thread.join(timeout=30)
    assert not thread.is_alive()
    frames = result["frames"]
    assert [f.is_table() for f in frames] == [True, False]
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()
    assert frames[1].payload == b"\x07"


def test_http_lightstream_accepting_writer():
    uri = f"http://127.0.0.1:{free_port()}/feed"

    def serve():
        writer = ls.write(uri, protocol="lightstream", accept=True)
        writer.register_table("quotes", sample_table())
        writer.register_message("health")
        writer.write(sample_table(0), name="quotes")
        writer.write_message("health", b"\x09")
        writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    reader = with_retries(lambda: ls.read(uri, protocol="lightstream"))
    reader.register_table("quotes", sample_table())
    reader.register_message("health")
    frames = list(reader)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert [f.is_table() for f in frames] == [True, False]
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()
    assert frames[1].payload == b"\x09"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-domain sockets are Unix-only")
def test_accepting_writer_serves_requests_back_to_back(tmp_path):
    uri = f"uds://{tmp_path / 'serve.sock'}"

    def serve():
        for _ in range(2):
            writer = ls.write(uri, accept=True)
            writer.write(sample_table(0))
            writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    # The first fetch rides out server startup. The endpoint then stays
    # bound between servings, so the second fetch connects with no
    # retry.
    reader = with_retries(lambda: ls.read(uri))
    assert sum(t.n_rows for t in reader) == 3
    reader = ls.read(uri)
    assert sum(t.n_rows for t in reader) == 3
    thread.join(timeout=30)
    assert not thread.is_alive()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-domain sockets are Unix-only")
def test_clients_queue_while_the_listener_is_busy(tmp_path):
    uri = f"uds://{tmp_path / 'queue.sock'}"

    def serve():
        for _ in range(3):
            writer = ls.write(uri, accept=True)
            writer.write(sample_table(0))
            writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    reader = with_retries(lambda: ls.read(uri))
    assert sum(t.n_rows for t in reader) == 3

    # Two clients connect together. The bound endpoint queues them in
    # its backlog and each is served in turn with no retry.
    rows = []

    def fetch():
        rows.append(sum(t.n_rows for t in ls.read(uri)))

    clients = [threading.Thread(target=fetch, daemon=True) for _ in range(2)]
    for client in clients:
        client.start()
    for client in clients:
        client.join(timeout=30)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert rows == [3, 3]


def test_accept_rejected_on_files(tmp_path):
    with pytest.raises(ls.TransportError, match="no accepting form"):
        ls.read(str(tmp_path / "data.arrow"), accept=True)
    with pytest.raises(ls.TransportError, match="no accepting form"):
        ls.write(str(tmp_path / "data.arrow"), accept=True)


def test_accept_rejected_on_stdio():
    with pytest.raises(ls.TransportError, match="no accepting form"):
        ls.read("stdio:", accept=True)
    with pytest.raises(ls.TransportError, match="no accepting form"):
        ls.write("stdio:", accept=True)


def test_accept_rejected_with_parallel_write():
    with pytest.raises(ls.TransportError, match="no accepting form"):
        ls.write("tcp://127.0.0.1:9", parallel=2, accept=True)
