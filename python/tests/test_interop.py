# Copyright Peter G. Bower 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Arrow stream protocol interop.

A Reader exports through `__arrow_c_stream__` as a pull-driven stream, so
pyarrow, polars, duckdb, and minarrow consume it natively without
materialising ahead of the consumer.
"""

import duckdb
import lightstream as ls
import minarrow
import polars as pl
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


@pytest.fixture
def two_batch_file(tmp_path):
    path = str(tmp_path / "quotes.arrow")
    with ls.write(path) as w:
        w.write(sample_table(0))
        w.write(sample_table(100))
    return path


def test_pyarrow_consumes_reader(two_batch_file):
    result = pa.table(ls.read(two_batch_file))
    assert result.num_rows == 6
    assert result.column("qty").to_pylist() == [10, 20, 30, 110, 120, 130]


def test_pyarrow_stream_preserves_batches(two_batch_file):
    reader = pa.RecordBatchReader.from_stream(ls.read(two_batch_file))
    batches = list(reader)
    assert len(batches) == 2
    assert batches[0].num_rows == 3


def test_polars_consumes_reader(two_batch_file):
    df = pl.DataFrame(ls.read(two_batch_file))
    assert df.height == 6
    assert df["px"].sum() == pytest.approx(1.5 + 2.5 + 3.5 + 101.5 + 102.5 + 103.5)


def test_duckdb_queries_reader(two_batch_file):
    reader = pa.RecordBatchReader.from_stream(ls.read(two_batch_file))
    total = duckdb.sql("SELECT sum(qty) AS total FROM reader").fetchall()[0][0]
    assert total == 420


def test_minarrow_consumes_reader(two_batch_file):
    chunked = minarrow.ChunkedTable.from_arrow(ls.read(two_batch_file))
    assert isinstance(chunked, minarrow.ChunkedTable)


def test_export_consumes_reader(two_batch_file):
    reader = ls.read(two_batch_file)
    pa.table(reader)
    with pytest.raises(ls.LightstreamError, match="closed"):
        reader.read_all()


def test_export_of_drained_reader_raises(two_batch_file):
    reader = ls.read(two_batch_file)
    list(reader)
    with pytest.raises(ls.LightstreamError, match="empty source"):
        pa.table(reader)


def test_unconsumed_capsule_releases_cleanly(two_batch_file):
    import gc

    capsule = ls.read(two_batch_file).__arrow_c_stream__()
    del capsule
    gc.collect()


def test_stream_export_from_csv_source(tmp_path):
    path = str(tmp_path / "quotes.csv")
    with ls.write(path) as w:
        w.write(sample_table())

    result = pa.table(ls.read(path))
    assert result.column("sym").to_pylist() == ["A", "B", "C"]
