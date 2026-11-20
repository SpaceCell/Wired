# Copyright Peter G. Bower 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The TLS-carrying wires: quic, wt, wss, and https.

The accepting peer presents the PEM pair in tls_cert and tls_key, and
the connecting peer verifies against the roots in tls_ca. The tests
generate a private root and a server certificate signed by it.
"""

import socket
import subprocess
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


@pytest.fixture(scope="module")
def tls_pair(tmp_path_factory):
    """A test root plus a server certificate for 127.0.0.1 signed by it.

    rustls requires the trust anchor and the end-entity certificate to
    be distinct, so the fixture builds a small chain rather than one
    self-signed certificate.
    """
    directory = tmp_path_factory.mktemp("tls")
    ca = directory / "ca.pem"
    ca_key = directory / "ca-key.pem"
    cert = directory / "cert.pem"
    key = directory / "key.pem"
    csr = directory / "server.csr"
    ext = directory / "server.ext"
    ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"]
    run = lambda args: subprocess.run(args, check=True, capture_output=True)

    run(
        ["openssl", "req", "-x509", *ec, "-keyout", str(ca_key), "-out", str(ca)]
        + ["-days", "2", "-subj", "/CN=lightstream test root"]
    )
    run(
        ["openssl", "req", "-new", *ec, "-keyout", str(key), "-out", str(csr)]
        + ["-subj", "/CN=localhost"]
    )
    ext.write_text(
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "basicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature\n"
        "extendedKeyUsage=serverAuth\n"
    )
    run(
        ["openssl", "x509", "-req", "-in", str(csr), "-CA", str(ca), "-CAkey", str(ca_key)]
        + ["-CAcreateserial", "-out", str(cert), "-days", "2", "-extfile", str(ext)]
    )
    return str(cert), str(key), str(ca)


def run_accepting_writer(uri, cert, key, ca):
    def serve():
        writer = ls.write(uri, accept=True, tls_cert=cert, tls_key=key)
        writer.write(sample_table(0))
        writer.write(sample_table(100))
        writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.5)

    reader = with_retries(lambda: ls.read(uri, tls_ca=ca))
    batches = list(reader)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert len(batches) == 2
    assert pa.table(batches[0]).to_pydict() == sample_table(0).to_pydict()
    assert pa.table(batches[1]).to_pydict() == sample_table(100).to_pydict()


def run_accepting_reader(uri, cert, key, ca):
    result = {}

    def serve():
        reader = ls.read(uri, accept=True, tls_cert=cert, tls_key=key)
        result["batches"] = list(reader)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.5)

    writer = ls.write(uri, tls_ca=ca)
    with_retries(lambda: writer.write(sample_table(0)))
    writer.write(sample_table(100))
    writer.close()
    thread.join(timeout=30)
    assert not thread.is_alive()
    batches = result["batches"]
    assert len(batches) == 2
    assert pa.table(batches[0]).to_pydict() == sample_table(0).to_pydict()
    assert pa.table(batches[1]).to_pydict() == sample_table(100).to_pydict()


def test_quic_accepting_writer(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_writer(f"quic://127.0.0.1:{free_port()}", cert, key, ca)


def test_quic_accepting_reader(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_reader(f"quic://127.0.0.1:{free_port()}", cert, key, ca)


def test_webtransport_accepting_writer(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_writer(f"wt://127.0.0.1:{free_port()}", cert, key, ca)


def test_webtransport_accepting_reader(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_reader(f"wt://127.0.0.1:{free_port()}", cert, key, ca)


def test_wss_accepting_writer(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_writer(f"wss://127.0.0.1:{free_port()}", cert, key, ca)


def test_wss_accepting_reader(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_reader(f"wss://127.0.0.1:{free_port()}", cert, key, ca)


def test_https_accepting_writer(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_writer(f"https://127.0.0.1:{free_port()}/feed", cert, key, ca)


def test_https_accepting_reader(tls_pair):
    cert, key, ca = tls_pair
    run_accepting_reader(f"https://127.0.0.1:{free_port()}/ingest", cert, key, ca)


def serve_two_paths(base, cert, key):
    """Serves a different table on `/prices` and `/trades` of one authority."""

    def serve(path, offset):
        writer = ls.write(f"{base}{path}", accept=True, tls_cert=cert, tls_key=key)
        writer.write(sample_table(offset))
        writer.close()

    threads = [
        threading.Thread(target=serve, args=(path, offset), daemon=True)
        for path, offset in [("/prices", 0), ("/trades", 100)]
    ]
    for thread in threads:
        thread.start()
    return threads


def read_path(uri, ca, deadline=30.0):
    """Reads every batch an accepting writer sends on a URL path, retrying
    until the writer has bound the path."""
    end = time.monotonic() + deadline
    while True:
        try:
            batches = list(ls.read(uri, tls_ca=ca))
            if batches:
                return batches
        except ls.LightstreamError:
            if time.monotonic() > end:
                raise
        assert time.monotonic() < end, f"no batches arrived from {uri}"
        time.sleep(0.05)


def run_two_paths(base, cert, key, ca):
    threads = serve_two_paths(base, cert, key)
    trades = read_path(f"{base}/trades", ca)
    prices = read_path(f"{base}/prices", ca)
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert [pa.table(b).to_pydict() for b in prices] == [sample_table(0).to_pydict()]
    assert [pa.table(b).to_pydict() for b in trades] == [sample_table(100).to_pydict()]


def test_wss_accepting_writers_on_two_paths_serve_their_own_tables(tls_pair):
    cert, key, ca = tls_pair
    run_two_paths(f"wss://127.0.0.1:{free_port()}", cert, key, ca)


def test_https_accepting_writers_on_two_paths_serve_their_own_tables(tls_pair):
    cert, key, ca = tls_pair
    run_two_paths(f"https://127.0.0.1:{free_port()}", cert, key, ca)


def test_webtransport_accepting_writers_on_two_paths_serve_their_own_tables(tls_pair):
    cert, key, ca = tls_pair
    run_two_paths(f"wt://127.0.0.1:{free_port()}", cert, key, ca)


def test_wss_lightstream_round_trip(tls_pair):
    cert, key, ca = tls_pair
    uri = f"wss://127.0.0.1:{free_port()}"
    result = {}

    def serve():
        reader = ls.read(uri, protocol="lightstream", accept=True, tls_cert=cert, tls_key=key)
        reader.register_table("quotes", sample_table())
        result["frames"] = list(reader)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.5)

    writer = with_retries(lambda: ls.write(uri, protocol="lightstream", tls_ca=ca))
    writer.register_table("quotes", sample_table())
    writer.write(sample_table(0), name="quotes")
    writer.close()
    thread.join(timeout=30)
    assert not thread.is_alive()
    frames = result["frames"]
    assert len(frames) == 1
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()


def test_https_lightstream_round_trip(tls_pair):
    cert, key, ca = tls_pair
    uri = f"https://127.0.0.1:{free_port()}/feed"

    def serve():
        writer = ls.write(uri, protocol="lightstream", accept=True, tls_cert=cert, tls_key=key)
        writer.register_table("quotes", sample_table())
        writer.write(sample_table(0), name="quotes")
        writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.5)

    reader = with_retries(lambda: ls.read(uri, protocol="lightstream", tls_ca=ca))
    reader.register_table("quotes", sample_table())
    frames = list(reader)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert len(frames) == 1
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()


def test_quic_lightstream_round_trip(tls_pair):
    cert, key, ca = tls_pair
    uri = f"quic://127.0.0.1:{free_port()}"
    result = {}

    def serve():
        reader = ls.read(uri, protocol="lightstream", accept=True, tls_cert=cert, tls_key=key)
        reader.register_table("quotes", sample_table())
        reader.register_message("health")
        result["frames"] = list(reader)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.5)

    writer = with_retries(lambda: ls.write(uri, protocol="lightstream", tls_ca=ca))
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


def test_webtransport_lightstream_round_trip(tls_pair):
    cert, key, ca = tls_pair
    uri = f"wt://127.0.0.1:{free_port()}"

    def serve():
        writer = ls.write(uri, protocol="lightstream", accept=True, tls_cert=cert, tls_key=key)
        writer.register_table("quotes", sample_table())
        writer.write(sample_table(0), name="quotes")
        writer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.5)

    reader = with_retries(lambda: ls.read(uri, protocol="lightstream", tls_ca=ca))
    reader.register_table("quotes", sample_table())
    frames = list(reader)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert len(frames) == 1
    assert pa.table(frames[0].table).to_pydict() == sample_table(0).to_pydict()


def test_connecting_requires_roots():
    for scheme in ("quic", "wt", "wss", "https"):
        with pytest.raises(ls.TransportError, match="requires tls_ca"):
            ls.read(f"{scheme}://127.0.0.1:9")


def test_accepting_requires_certificate(tls_pair):
    cert, _key, _ca = tls_pair
    for scheme in ("quic", "wt", "wss", "https"):
        with pytest.raises(ls.TransportError, match="requires tls_cert and tls_key"):
            ls.write(f"{scheme}://127.0.0.1:9", accept=True, tls_cert=cert)


def test_tls_arguments_rejected_on_plain_wires(tls_pair):
    cert, key, _ca = tls_pair
    with pytest.raises(ls.TransportError, match="apply to the quic, wt, wss, and https"):
        ls.read("tcp://127.0.0.1:9", tls_ca=cert)
    with pytest.raises(ls.TransportError, match="apply to the quic, wt, wss, and https"):
        ls.write("tcp://127.0.0.1:9", accept=True, tls_cert=cert, tls_key=key)
