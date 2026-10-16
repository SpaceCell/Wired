# Copyright Peter G. Bower 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Object storage URLs through ls.read and ls.write.

Runs the file-format round-trips against `memory://` URLs, which address
an in-memory store shared across the process. Each test writes beneath
its own prefix to keep its objects separate. The module requires a build
with the `object_storage` feature, which also adds `ls.register_options`.
"""

import gc
import uuid

import lightstream as ls
import minarrow
import pyarrow as pa
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ls, "register_options"),
    reason="built without the object_storage feature",
)


def sample_table(offset=0):
    return pa.table(
        {
            "sym": ["A", "B", "C"],
            "px": [1.5 + offset, 2.5 + offset, 3.5 + offset],
            "qty": [10 + offset, 20 + offset, 30 + offset],
        }
    )


@pytest.fixture
def base():
    """A `memory://` prefix unique to the test."""
    return f"memory:///tests/{uuid.uuid4().hex}"


def test_ipc_round_trip(base):
    url = f"{base}/quotes.arrow"
    with ls.write(url) as w:
        w.write(sample_table())

    result = ls.read(url).read_all()
    assert isinstance(result, minarrow.Table)
    assert pa.table(result).equals(sample_table())


def test_ipc_multi_batch_reads_as_chunked_table(base):
    url = f"{base}/quotes.arrow"
    with ls.write(url, compression="zstd") as w:
        w.write(sample_table(0))
        w.write(sample_table(100))

    batches = list(ls.read(url))
    assert len(batches) == 2
    assert pa.table(batches[1]).equals(sample_table(100))
    assert isinstance(ls.read(url).read_all(), minarrow.ChunkedTable)


def test_out_of_core_ipc_read(base):
    url = f"{base}/quotes.arrow"
    with ls.write(url) as w:
        w.write(sample_table())

    result = ls.read(url, out_of_core=True).read_all()
    assert pa.table(result).equals(sample_table())


def test_parquet_round_trip(base):
    url = f"{base}/quotes.parquet"
    with ls.write(url) as w:
        w.write(sample_table(0))
        w.write(sample_table(100))

    result = pa.table(ls.read(url).read_all())
    assert result.num_rows == 6
    assert result.column("qty").to_pylist() == [10, 20, 30, 110, 120, 130]


def test_csv_round_trip(base):
    url = f"{base}/quotes.csv"
    with ls.write(url, delimiter=";") as w:
        w.write(sample_table())

    result = pa.table(ls.read(url, delimiter=";").read_all())
    assert result.to_pydict() == sample_table().to_pydict()


def test_csv_batch_size_splits_batches(base):
    url = f"{base}/quotes.csv"
    with ls.write(url) as w:
        w.write(sample_table())

    assert len(list(ls.read(url, batch_size=2))) == 2


@pytest.mark.parametrize("name", ["quotes.json", "quotes.ndjson"])
def test_json_write_creates_the_object(base, name):
    url = f"{base}/{name}"
    with ls.write(url) as w:
        w.write(sample_table())

    # The JSON reader needs a schema, so the read fails at decoding after
    # the object opens.
    with pytest.raises(ls.LightstreamError, match="schema"):
        ls.read(url).read_all()


def test_lightstream_protocol_round_trip(base):
    url = f"{base}/feed.tlv"
    w = ls.write(url, protocol="lightstream")
    w.register_table("quotes", sample_table())
    w.register_message("health")
    w.write(sample_table(0), name="quotes")
    w.write_message("health", b"\x01\x02\x03")
    w.write(sample_table(100), name="quotes")
    w.close()

    r = ls.read(url, protocol="lightstream")
    r.register_table("quotes", sample_table())
    r.register_message("health")
    frames = list(r)
    assert [f.is_table() for f in frames] == [True, False, True]
    assert frames[1].payload == b"\x01\x02\x03"
    assert pa.table(frames[2].table).to_pydict() == sample_table(100).to_pydict()


@pytest.mark.parametrize("ext", ["arrow", "parquet", "csv"])
def test_chunk_prefix_reads_in_order(base, ext):
    for idx in range(2):
        with ls.write(f"{base}/part-{idx:010}.{ext}") as w:
            w.write(sample_table(idx * 100))

    batches = list(ls.read(base))
    assert len(batches) == 2
    assert pa.table(batches[0]).to_pydict() == sample_table(0).to_pydict()
    assert pa.table(batches[1]).to_pydict() == sample_table(100).to_pydict()


def test_chunk_prefix_base_disambiguation(base):
    for name in ("left", "right"):
        with ls.write(f"{base}/{name}-0000000000.arrow") as w:
            w.write(sample_table())

    with pytest.raises(ls.FormatError, match="multiple chunk sets"):
        ls.read(base)

    assert len(list(ls.read(base, base="left"))) == 1


def test_reader_exports_through_the_capsule_stream(base):
    url = f"{base}/quotes.arrow"
    with ls.write(url) as w:
        w.write(sample_table(0))
        w.write(sample_table(100))

    reader = pa.RecordBatchReader.from_stream(ls.read(url))
    assert reader.read_all().num_rows == 6


def test_writer_drop_completes_the_object(base):
    url = f"{base}/quotes.arrow"
    w = ls.write(url)
    w.write(sample_table())
    del w
    gc.collect()

    assert pa.table(ls.read(url).read_all()).equals(sample_table())


def test_missing_object_raises(base):
    with pytest.raises(ls.LightstreamError):
        ls.read(f"{base}/missing.arrow")


def test_mmap_on_an_object_url_raises(base):
    with pytest.raises(ls.FormatError, match="local files only"):
        ls.read(f"{base}/quotes.arrow", mmap=True)


def test_registered_options_apply_beneath_the_prefix(base):
    ls.register_options(base, {"max_concurrent_parts": "1", "part_size": "6291456"})
    url = f"{base}/quotes.parquet"
    with ls.write(url) as w:
        w.write(sample_table())

    assert pa.table(ls.read(url).read_all()).to_pydict() == sample_table().to_pydict()


def test_register_options_rejects_a_local_path(tmp_path):
    with pytest.raises(ls.LightstreamError, match="not an object storage URL"):
        ls.register_options(str(tmp_path), {})


def test_register_options_rejects_an_invalid_upload_setting(base):
    with pytest.raises(ls.LightstreamError, match="max_concurrent_parts"):
        ls.register_options(base, {"max_concurrent_parts": "0"})
