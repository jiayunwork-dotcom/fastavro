"""Regression tests for the opt-in union branch read mode.

These tests intentionally exercise the public API only (plus a raw binary probe
for checking the branch index that actually landed in the file).  They pass
unchanged on the Cython extension and on the pure Python backend.

Re-running on CPython 3.12, Cython extension::

    python setup.py build_ext --inplace
    python -m pytest tests/test_union_branch.py

Re-running on CPython 3.12, pure Python backend (no compiled modules)::

    rm -rf /tmp/fastavro-pure
    cp -r fastavro /tmp/fastavro-pure
    find /tmp/fastavro-pure -name '*.so' -delete
    PYTHONPATH=/tmp/fastavro-pure python -m pytest tests/test_union_branch.py

``conftest.is_testing_pure_python()`` reports which backend a run used, and the
``TEST_BACKEND`` environment variable records it for logs.
"""

import json
import os
from io import BytesIO

import pytest

import fastavro
from fastavro import (
    UnionValue,
    get_union_branch,
    get_union_value,
    is_union_value,
    make_union_value,
)
from fastavro._read_common import HEADER_SCHEMA, SYNC_SIZE
from fastavro import _read_py
from fastavro.validation import ValidationError

from .conftest import is_testing_pure_python


def backend_name():
    return "pure-python" if is_testing_pure_python() else "cython"


def test_backend_is_reported():
    # Makes the backend visible in verbose output so a failing run says which
    # implementation reproduced it.
    assert backend_name() in ("pure-python", "cython")
    assert os.environ.get("TEST_BACKEND", backend_name()) == backend_name()


CODECS = ["null", "deflate"]


MEASUREMENT_SCHEMA = {
    "type": "record",
    "name": "Measurement",
    "namespace": "example",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "value", "type": ["float", "double"]},
        {"name": "label", "type": "string"},
    ],
}


def write_bytes(records, schema=MEASUREMENT_SCHEMA, codec="null", **kwargs):
    fo = BytesIO()
    fastavro.writer(fo, schema, records, codec=codec, **kwargs)
    fo.seek(0)
    return fo


def write_file(path, records, schema=MEASUREMENT_SCHEMA, codec="null", **kwargs):
    with open(path, "wb") as fo:
        fastavro.writer(fo, schema, records, codec=codec, **kwargs)
    return path


def read(fo, **kwargs):
    fo.seek(0)
    return list(fastavro.reader(fo, **kwargs))


def read_path(path, **kwargs):
    with open(path, "rb") as fo:
        return list(fastavro.reader(fo, **kwargs))


def union_indices_in_file(path_or_fo, field_index=None, field_name=None):
    """Return the on-disk union index for one field of every record.

    The file is decoded with the *pure Python reference decoder* and only the
    raw union index is read, so this cannot be satisfied by the branch marker
    machinery under test.
    """
    if isinstance(path_or_fo, (str, bytes, os.PathLike)):
        with open(path_or_fo, "rb") as fo:
            return union_indices_in_file(fo, field_index, field_name)

    fo = path_or_fo
    fo.seek(0)
    decoder = _read_py.BinaryDecoder(fo)
    named_schemas = _read_py._default_named_schemas()
    header = _read_py.read_data(decoder, HEADER_SCHEMA, named_schemas, None, {})

    metadata = {key: value.decode() for key, value in header["meta"].items()}
    schema = json.loads(metadata["avro.schema"])
    codec = metadata.get("avro.codec", "null")
    writer_schema = _read_py.parse_schema(
        schema, named_schemas["writer"], _write_hint=False, _force=True
    )
    sync_marker = header["sync"]

    fields = writer_schema["fields"]
    if field_name is not None:
        field_index = [f["name"] for f in fields].index(field_name)
    if field_index is None:
        raise ValueError("field_index or field_name is required")

    read_block = _read_py.BLOCK_READERS[codec]

    indices = []
    while True:
        try:
            count = decoder.read_long()
        except EOFError:
            break
        block_fo = read_block(decoder)
        block_decoder = _read_py.BinaryDecoder(block_fo)
        for _ in range(count):
            for index, field in enumerate(fields):
                if index != field_index:
                    _read_py.skip_data(block_decoder, field["type"], named_schemas)
                else:
                    union_schema = field["type"]
                    branch = block_decoder.read_index()
                    indices.append(branch)
                    _read_py.skip_data(
                        block_decoder, union_schema[branch], named_schemas
                    )
        assert block_decoder.fo.read() == b""
        assert decoder.fo.read(SYNC_SIZE) == sync_marker

    return indices


def records_with_float_and_double():
    return [
        {
            "id": 1,
            "value": make_union_value("float", 1.25),
            "label": "a",
        },
        {
            "id": 2,
            "value": make_union_value("double", 1.25),
            "label": "b",
        },
    ]


@pytest.mark.parametrize("codec", CODECS)
def test_smoke_branch_identity(tmp_path, codec):
    path = tmp_path / f"smoke-{codec}.avro"

    # Two records with equal Python values.  Writing plain values makes the
    # regular writer choose `double` for both.
    write_file(
        path,
        [
            {"id": 1, "value": 1.25, "label": "a"},
            {"id": 2, "value": 1.25, "label": "b"},
        ],
        codec=codec,
    )
    assert read_path(path) == [
        {"id": 1, "value": 1.25, "label": "a"},
        {"id": 2, "value": 1.25, "label": "b"},
    ]
    branches = read_path(path, return_union_branch=True)
    assert [record["value"].branch for record in branches] == ["double", "double"]
    for record in branches:
        assert isinstance(record["value"], UnionValue)
        assert record["value"].value == 1.25

    # Now write the same Python values into different branches.
    write_file(path, records_with_float_and_double(), codec=codec)
    branches = read_path(path, return_union_branch=True)
    assert [record["value"].branch for record in branches] == ["float", "double"]
    assert [record["value"].index for record in branches] == [0, 1]
    assert branches[0]["value"].value == branches[1]["value"].value == 1.25

    # And those branches are what the bytes actually say.
    assert union_indices_in_file(path, 1) == [0, 1]


@pytest.mark.parametrize("codec", CODECS)
def test_edit_unrelated_field_preserves_branch(tmp_path, codec):
    path = tmp_path / f"edit-{codec}.avro"
    write_file(path, records_with_float_and_double(), codec=codec)

    records = read_path(path, return_union_branch=True)
    records[0]["label"] = "edited"
    records[1]["id"] = 20

    # Import-time validation accepts the markers ...
    for record in records:
        assert fastavro.validate(record, MEASUREMENT_SCHEMA)

    rewritten = tmp_path / f"rewritten-{codec}.avro"
    write_file(rewritten, records, codec=codec)

    # ... and the branches survive the edit, validation and rewrite.
    assert union_indices_in_file(rewritten, field_name="value") == [0, 1]
    back = read_path(rewritten, return_union_branch=True)
    assert [record["value"].branch for record in back] == ["float", "double"]
    assert back[0]["label"] == "edited"
    assert back[1]["id"] == 20


def test_validate_rejects_missing_branch_and_wrong_value():
    schema = fastavro.parse_schema(MEASUREMENT_SCHEMA)

    good = {"id": 1, "value": make_union_value("float", 1.25), "label": "x"}
    assert fastavro.validate(good, schema)

    # A branch that does not exist in the target union.
    missing_branch = {
        "id": 1,
        "value": make_union_value("bytes", b"\x00"),
        "label": "x",
    }
    with pytest.raises(ValidationError):
        fastavro.validate(missing_branch, schema)
    assert fastavro.validate(missing_branch, schema, raise_errors=False) is False

    # A value that only validates in a different branch must not be rerouted.
    wrong_value = {
        "id": 1,
        "value": make_union_value("float", "not a number"),
        "label": "x",
    }
    with pytest.raises(ValidationError):
        fastavro.validate(wrong_value, schema)
    assert fastavro.validate(wrong_value, schema, raise_errors=False) is False

    other_branch = {
        "id": 1,
        "value": make_union_value("float", None),
        "label": "x",
    }
    assert fastavro.validate(other_branch, schema, raise_errors=False) is False


@pytest.mark.parametrize("codec", CODECS)
def test_writer_checks_branch_without_validator(codec):
    schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "value", "type": ["float", "double"]}],
    }
    bad_value = {"value": make_union_value("float", "nope")}
    with pytest.raises(ValueError):
        fastavro.writer(BytesIO(), schema, [bad_value], codec=codec)

    missing_branch = {"value": make_union_value("bytes", b"x")}
    with pytest.raises(ValueError):
        fastavro.writer(BytesIO(), schema, [missing_branch], codec=codec)

    with pytest.raises(ValueError):
        fastavro.schemaless_writer(BytesIO(), schema, bad_value)


@pytest.mark.parametrize("codec", CODECS)
def test_nested_positions_with_reader_schema(tmp_path, codec):
    writer_schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [
            {"name": "a", "type": {"type": "array", "items": "int"}},
            {"name": "m", "type": {"type": "map", "values": "int"}},
            {
                "name": "child",
                "type": {
                    "type": "record",
                    "name": "Child",
                    "fields": [{"name": "n", "type": "int"}],
                },
            },
        ],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [
            {"name": "a", "type": {"type": "array", "items": ["int", "long"]}},
            {"name": "m", "type": {"type": "map", "values": ["null", "int"]}},
            {
                "name": "child",
                "type": {
                    "type": "record",
                    "name": "Child",
                    "fields": [{"name": "n", "type": ["int", "long"]}],
                },
            },
        ],
    }
    path = tmp_path / f"nested-resolution-{codec}.avro"
    write_file(
        path,
        [{"a": [1, 2], "m": {"x": 3}, "child": {"n": 4}}],
        writer_schema,
        codec=codec,
    )
    out = read_path(path, reader_schema=reader_schema, return_union_branch=True)[0]

    assert out["a"][0].branch == "int"
    assert out["a"][0].index == 0
    assert out["a"][1].branch == "int"
    assert out["m"]["x"].branch == "int"
    assert out["m"]["x"].index == 1
    assert out["child"]["n"].branch == "int"
    assert out["child"]["n"].index == 0

    # If the reader schema collapses a nested position, it is plain there.
    collapsed = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [
            {"name": "a", "type": {"type": "array", "items": "int"}},
            {"name": "m", "type": {"type": "map", "values": ["null", "int"]}},
            {
                "name": "child",
                "type": {
                    "type": "record",
                    "name": "Child",
                    "fields": [{"name": "n", "type": "int"}],
                },
            },
        ],
    }
    out = read_path(path, reader_schema=collapsed, return_union_branch=True)[0]
    assert out["a"] == [1, 2]
    assert not isinstance(out["a"][0], UnionValue)
    assert out["m"]["x"].branch == "int"
    assert out["child"]["n"] == 4
    assert not isinstance(out["child"]["n"], UnionValue)


@pytest.mark.parametrize("codec", CODECS)
def test_defaults_nested_in_record_array_and_map(tmp_path, codec):
    writer_schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [{"name": "a", "type": "int"}],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [
            {"name": "a", "type": "int"},
            {
                "name": "cfg",
                "type": {
                    "type": "record",
                    "name": "Config",
                    "fields": [
                        {"name": "v", "type": ["float", "double"], "default": 1.25}
                    ],
                },
                "default": {"v": 1.25},
            },
            {
                "name": "list",
                "type": {"type": "array", "items": ["float", "double"]},
                "default": [1.25, 2.5],
            },
            {
                "name": "dict",
                "type": {"type": "map", "values": ["null", "double"]},
                "default": {"k": 1.5},
            },
        ],
    }
    path = tmp_path / f"nested-defaults-{codec}.avro"
    write_file(path, [{"a": 1}], writer_schema, codec=codec)
    out = read_path(path, reader_schema=reader_schema, return_union_branch=True)[0]

    # `1.25` fits the first branch of ["float", "double"], so it is `float`.
    assert out["cfg"]["v"].branch == "float"
    assert out["cfg"]["v"].index == 0
    assert out["list"][0].branch == "float"
    assert out["list"][1].branch == "float"
    # `1.5` cannot inhabit `null`, so it lands in `double`.
    assert out["dict"]["k"].branch == "double"
    assert out["dict"]["k"].index == 1

    assert fastavro.validate(out, reader_schema)
    rewritten = tmp_path / f"nested-defaults-rewritten-{codec}.avro"
    write_file(rewritten, [out], reader_schema, codec=codec)
    assert read_path(rewritten, return_union_branch=True)[0] == out


def test_default_union_with_inline_record_branch():
    writer_schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [{"name": "a", "type": "int"}],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [
            {"name": "a", "type": "int"},
            {
                "name": "opt",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "Thing",
                        "fields": [{"name": "n", "type": "int"}],
                    },
                ],
                "default": {"n": 5},
            },
        ],
    }
    fo = write_bytes([{"a": 1}], writer_schema)
    out = read(fo, reader_schema=reader_schema, return_union_branch=True)[0]
    assert out["opt"].branch == "example.Thing"
    assert out["opt"].index == 1
    assert out["opt"].value == {"n": 5}
    assert fastavro.validate(out, reader_schema)
    assert read(write_bytes([out], reader_schema), return_union_branch=True)[0] == out


@pytest.mark.parametrize("codec", CODECS)
def test_named_reference_field_to_union_reader(tmp_path, codec):
    # The writer field refers to a named record by name; the reader schema
    # turns the same field into a union containing an inline record that
    # aliases the old name.  This path must resolve the reader branch instead
    # of trying to look up a dict in the named schema table.
    writer_schema = {
        "type": "record",
        "name": "Envelope",
        "namespace": "ex",
        "fields": [
            {
                "name": "body",
                "type": {
                    "type": "record",
                    "name": "Body",
                    "fields": [{"name": "n", "type": "int"}],
                },
            },
            {"name": "choice", "type": "Body"},
        ],
    }
    reader_schema = {
        "type": "record",
        "name": "Envelope",
        "namespace": "ex",
        "fields": [
            {
                "name": "body",
                "type": {
                    "type": "record",
                    "name": "Body",
                    "fields": [{"name": "n", "type": "int"}],
                },
            },
            {
                "name": "choice",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "Other",
                        "aliases": ["ex.Body"],
                        "fields": [{"name": "n", "type": "int"}],
                    },
                ],
            },
        ],
    }
    path = tmp_path / f"named-ref-{codec}.avro"
    write_file(
        path,
        [{"body": {"n": 1}, "choice": {"n": 2}}],
        writer_schema,
        codec=codec,
    )
    out = read_path(path, reader_schema=reader_schema, return_union_branch=True)
    assert out[0]["body"] == {"n": 1}
    assert out[0]["choice"].branch == "ex.Other"
    assert out[0]["choice"].index == 1
    assert out[0]["choice"].value == {"n": 2}


class _CountingReader:
    """File wrapper that records how many bytes have been read."""

    def __init__(self, fo):
        self._fo = fo
        self.bytes_read = 0

    def read(self, n=-1):
        data = self._fo.read(n)
        self.bytes_read += len(data)
        return data

    def seek(self, *args):
        return self._fo.seek(*args)

    def tell(self):
        return self._fo.tell()


@pytest.mark.parametrize("codec", CODECS)
def test_reader_and_block_reader_stay_streaming(codec):
    schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "v", "type": ["float", "double"]},
            {"name": "pad", "type": "string"},
        ],
    }
    records = [
        {
            "v": make_union_value("float" if i % 2 else "double", 1.25),
            "pad": "x" * 50,
        }
        for i in range(200)
    ]
    # A tiny sync interval produces many blocks so laziness is observable.
    fo = write_bytes(records, schema, codec=codec, sync_interval=1000)

    payload = fo.getvalue()
    total = len(payload)

    counting = _CountingReader(BytesIO(payload))
    stream = fastavro.reader(counting, return_union_branch=True)
    first = next(iter(stream))
    assert first["v"].branch == "double"
    assert counting.bytes_read < total / 2, "reader buffered the whole file"

    counting = _CountingReader(BytesIO(payload))
    blocks = fastavro.block_reader(counting, return_union_branch=True)
    next(iter(blocks))
    assert counting.bytes_read < total / 2, "block_reader buffered the whole file"


def test_binary_defaults_are_written_like_read_values():
    # Avro bytes/fixed defaults are JSON strings with one character per byte.
    # In branch mode they must be usable exactly like a value read from the
    # file, i.e. they validate and write.
    writer_schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "a", "type": "int"}],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": ["null", "bytes"], "default": "abc"},
            {
                "name": "f",
                "type": ["null", {"type": "fixed", "name": "F", "size": 2}],
                "default": "ab",
            },
        ],
    }
    out = read(
        write_bytes([{"a": 1}], writer_schema),
        reader_schema=reader_schema,
        return_union_branch=True,
    )[0]

    assert out["b"].branch == "bytes"
    assert out["b"].value == b"abc"
    assert out["f"].branch == "F"
    assert out["f"].value == b"ab"

    assert fastavro.validate(out, reader_schema)
    fo = write_bytes([out], reader_schema)
    assert read(fo, return_union_branch=True)[0] == out


def test_reader_schema_selects_target_branch():
    # The writer location is not a union; the reader schema introduces one.
    writer_schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "string"},
        ],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "x", "type": ["int", "long"]},
            {"name": "y", "type": "string"},
        ],
    }
    fo = write_bytes([{"x": 3, "y": "a"}], writer_schema)
    out = read(fo, reader_schema=reader_schema, return_union_branch=True)

    assert isinstance(out[0]["x"], UnionValue)
    assert out[0]["x"].branch == "int"
    assert out[0]["x"].index == 0
    assert out[0]["x"].value == 3
    assert out[0]["y"] == "a"


def test_reader_schema_collapsing_union_returns_plain_value():
    writer_schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": "double"}],
    }
    fo = write_bytes([{"v": make_union_value("double", 1.5)}], writer_schema)
    out = read(fo, reader_schema=reader_schema, return_union_branch=True)

    assert out[0]["v"] == 1.5
    assert not isinstance(out[0]["v"], UnionValue)


def test_reader_schema_promotion_reports_target_branch():
    writer_schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    # `double` is listed first, so writer `float` resolves to it.
    reader_schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["double", "float"]}],
    }
    fo = write_bytes([{"v": make_union_value("float", 1.25)}], writer_schema)
    out = read(fo, reader_schema=reader_schema, return_union_branch=True)
    assert out[0]["v"].branch == "double"
    assert out[0]["v"].index == 0
    assert out[0]["v"].value == 1.25

    # An extra branch in front must not confuse the branch identity.
    reader_schema["fields"][0]["type"] = ["null", "float", "double"]
    out = read(fo, reader_schema=reader_schema, return_union_branch=True)
    assert out[0]["v"].branch == "float"
    assert out[0]["v"].index == 1


@pytest.mark.parametrize("codec", CODECS)
def test_reader_schema_defaults_are_marked_and_revalidate(tmp_path, codec):
    writer_schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "a", "type": "int"}],
    }
    reader_schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": ["null", "double"], "default": None},
            {"name": "c", "type": ["int", "double"], "default": 2.5},
            {"name": "d", "type": ["double", "int"], "default": 2.5},
        ],
    }
    path = tmp_path / f"defaults-{codec}.avro"
    write_file(path, [{"a": 1}], writer_schema, codec=codec)

    out = read_path(path, reader_schema=reader_schema, return_union_branch=True)
    assert out[0]["b"].branch == "null"
    assert out[0]["b"].value is None
    # 2.5 cannot inhabit `int`, so it lands in `double` in both unions.
    assert out[0]["c"].branch == "double"
    assert out[0]["c"].index == 1
    assert out[0]["d"].branch == "double"
    assert out[0]["d"].index == 0

    # The default-derived markers are validated and written like any other
    # union value.
    assert fastavro.validate(out[0], reader_schema)
    rewritten = tmp_path / f"defaults-rewritten-{codec}.avro"
    write_file(rewritten, out, reader_schema, codec=codec)
    back = read_path(rewritten, return_union_branch=True)
    assert back[0]["b"].branch == "null"
    assert back[0]["c"].branch == "double"
    assert back[0]["d"].branch == "double"
    assert back[0]["d"].value == 2.5


@pytest.mark.parametrize("codec", CODECS)
def test_nested_arrays_maps_and_recursion(tmp_path, codec):
    schema = {
        "type": "record",
        "name": "Node",
        "namespace": "example",
        "fields": [
            {
                "name": "vals",
                "type": {"type": "array", "items": ["float", "double"]},
            },
            {
                "name": "m",
                "type": {"type": "map", "values": ["null", "string"]},
            },
            {"name": "child", "type": ["null", "Node"], "default": None},
        ],
    }
    record = {
        "vals": [
            make_union_value("float", 1.25),
            make_union_value("double", 1.25),
        ],
        "m": {"k": make_union_value("string", "v")},
        "child": make_union_value(
            "example.Node",
            {
                "vals": [],
                "m": {},
                "child": make_union_value("null", None),
            },
        ),
    }
    path = tmp_path / f"nested-{codec}.avro"
    write_file(path, [record], schema, codec=codec)
    out = read_path(path, return_union_branch=True)[0]

    assert out["vals"][0].branch == "float"
    assert out["vals"][0].index == 0
    assert out["vals"][1].branch == "double"
    assert out["vals"][1].index == 1
    assert out["m"]["k"].branch == "string"
    assert out["m"]["k"].index == 1
    assert out["child"].branch == "example.Node"
    assert out["child"].value["child"].branch == "null"

    assert fastavro.validate(out, schema)
    rewritten = tmp_path / f"nested-rewritten-{codec}.avro"
    write_file(rewritten, [out], schema, codec=codec)
    assert read_path(rewritten, return_union_branch=True)[0] == out


def test_enum_and_fixed_named_branches():
    schema = {
        "type": "record",
        "name": "R",
        "namespace": "example",
        "fields": [
            {
                "name": "e",
                "type": [
                    "null",
                    {"type": "enum", "name": "Color", "symbols": ["RED", "BLUE"]},
                ],
            },
            {
                "name": "f",
                "type": [
                    {"type": "fixed", "name": "Hash", "size": 2},
                    "null",
                ],
            },
        ],
    }
    record = {
        "e": make_union_value("example.Color", "BLUE"),
        "f": make_union_value("example.Hash", b"ab"),
    }
    out = read(write_bytes([record], schema), return_union_branch=True)[0]
    assert out["e"].branch == "example.Color"
    assert out["e"].index == 1
    assert out["f"].branch == "example.Hash"
    assert out["f"].index == 0

    assert fastavro.validate(out, schema)
    assert read(write_bytes([out], schema), return_union_branch=True)[0] == out


@pytest.mark.parametrize("codec", CODECS)
def test_aliases_and_defaults_from_historic_schemas(tmp_path, codec):
    old_schema = {
        "type": "record",
        "name": "Measurement",
        "namespace": "example",
        "fields": [
            {"name": "id", "type": "long"},
            {"name": "value", "type": ["float", "double"]},
        ],
    }
    new_schema = {
        "type": "record",
        "name": "Measurement",
        "namespace": "example",
        "fields": [
            {"name": "id", "type": "long"},
            {"name": "value", "type": ["float", "double"]},
            {"name": "unit", "type": "string", "default": "C"},
            {"name": "quality", "type": ["null", "string"], "default": None},
        ],
    }
    path = tmp_path / f"historic-{codec}.avro"
    write_file(
        path,
        [
            {"id": 1, "value": make_union_value("float", 1.25)},
            {"id": 2, "value": make_union_value("double", 1.25)},
        ],
        old_schema,
        codec=codec,
    )
    out = read_path(path, reader_schema=new_schema, return_union_branch=True)
    assert out[0]["unit"] == "C"
    assert out[0]["quality"].branch == "null"
    assert out[0]["value"].branch == "float"
    assert out[1]["value"].branch == "double"

    # A field alias adds a second default-containing field to the mix.
    aliased_schema = {
        "type": "record",
        "name": "Measurement",
        "namespace": "example",
        "fields": [
            {"name": "id", "type": "long"},
            {"name": "value", "type": ["float", "double"]},
            {
                "name": "unit_label",
                "type": "string",
                "default": "C",
                "aliases": ["unit"],
            },
            {"name": "quality", "type": ["null", "string"], "default": None},
        ],
    }
    out = read_path(path, reader_schema=aliased_schema, return_union_branch=True)
    assert out[0]["unit_label"] == "C"
    assert out[0]["quality"].branch == "null"
    assert out[0]["value"].branch == "float"

    assert fastavro.validate(out[0], aliased_schema)
    rewritten = tmp_path / f"historic-rewritten-{codec}.avro"
    write_file(rewritten, out, aliased_schema, codec=codec)
    assert read_path(rewritten, return_union_branch=True) == out


def test_renamed_record_branch_through_alias():
    old_schema = {
        "type": "record",
        "name": "Envelope",
        "namespace": "example",
        "fields": [
            {
                "name": "body",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "OldBody",
                        "fields": [{"name": "n", "type": "int"}],
                    },
                ],
            }
        ],
    }
    new_schema = {
        "type": "record",
        "name": "Envelope",
        "namespace": "example",
        "fields": [
            {
                "name": "body",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "NewBody",
                        "aliases": ["example.OldBody"],
                        "fields": [{"name": "n", "type": "int"}],
                    },
                ],
            }
        ],
    }
    fo = write_bytes(
        [{"body": make_union_value("example.OldBody", {"n": 1})}], old_schema
    )
    out = read(fo, reader_schema=new_schema, return_union_branch=True)
    # The target (reader) branch identity is reported.
    assert out[0]["body"].branch == "example.NewBody"
    assert out[0]["body"].index == 1

    # Reading from the old schema and writing to the new one works because the
    # new branch aliases the old name.
    direct = read(fo, return_union_branch=True)
    assert direct[0]["body"].branch == "example.OldBody"
    rewritten = write_bytes(direct, new_schema)
    assert read(rewritten, return_union_branch=True)[0]["body"].branch == (
        "example.NewBody"
    )


@pytest.mark.parametrize("codec", CODECS)
def test_error_union_branch(tmp_path, codec):
    schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {
                "name": "result",
                "type": [
                    {
                        "type": "record",
                        "name": "Success",
                        "fields": [{"name": "code", "type": "int"}],
                    },
                    {
                        "type": "error",
                        "name": "Failure",
                        "fields": [{"name": "message", "type": "string"}],
                    },
                ],
            }
        ],
    }
    records = [
        {"result": make_union_value("Success", {"code": 0})},
        {"result": make_union_value("Failure", {"message": "boom"})},
    ]
    path = tmp_path / f"errors-{codec}.avro"
    write_file(path, records, schema, codec=codec)

    out = read_path(path, return_union_branch=True)
    assert out[0]["result"].branch == "Success"
    assert out[0]["result"].index == 0
    assert out[1]["result"].branch == "Failure"
    assert out[1]["result"].index == 1
    assert union_indices_in_file(path, 0) == [0, 1]

    assert fastavro.validate(out[1], schema)
    rewritten = tmp_path / f"errors-rewritten-{codec}.avro"
    write_file(rewritten, out, schema, codec=codec)
    assert read_path(rewritten, return_union_branch=True) == out


def test_tuple_notation_writer_still_works():
    schema = {
        "namespace": "namespace",
        "name": "name",
        "type": "record",
        "fields": [
            {"name": "foo", "type": ["string", {"type": "array", "items": "string"}]}
        ],
    }
    # The tuple selects the `string` branch.
    assert read(write_bytes([{"foo": ("string", "0")}], schema)) == [{"foo": "0"}]

    # With tuple notation disabled the tuple is ordinary data for the array.
    assert read(
        write_bytes([{"foo": ("string", "0")}], schema, disable_tuple_notation=True)
    ) == [{"foo": ["string", "0"]}]


def test_disable_tuple_notation_keeps_branch_markers():
    schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    out = read(
        write_bytes(
            [
                {"v": make_union_value("float", 1.25)},
                {"v": make_union_value("double", 1.25)},
            ],
            schema,
            disable_tuple_notation=True,
        ),
        return_union_branch=True,
    )
    assert [record["v"].branch for record in out] == ["float", "double"]

    # A tuple is data here, not a branch hint.
    data_schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "v", "type": ["string", {"type": "array", "items": "string"}]}
        ],
    }
    out = read(
        write_bytes([{"v": ("string", "0")}], data_schema, disable_tuple_notation=True)
    )
    assert out[0]["v"] == ["string", "0"]


def test_validate_disable_tuple_notation_keeps_branch_markers():
    schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    good = {"v": make_union_value("float", 1.25)}
    assert fastavro.validate(good, schema, disable_tuple_notation=True)

    bad = {"v": make_union_value("float", "nope")}
    assert (
        fastavro.validate(bad, schema, raise_errors=False, disable_tuple_notation=True)
        is False
    )


def test_conflicting_reader_options_are_rejected():
    fo = write_bytes([{"id": 1, "value": 1.25, "label": "a"}])
    for option in (
        "return_record_name",
        "return_record_name_override",
        "return_named_type",
        "return_named_type_override",
    ):
        with pytest.raises(ValueError, match="return_union_branch"):
            read(fo, return_union_branch=True, **{option: True})

        with pytest.raises(ValueError, match="return_union_branch"):
            fastavro.block_reader(
                BytesIO(fo.getvalue()), return_union_branch=True, **{option: True}
            )

        with pytest.raises(ValueError, match="return_union_branch"):
            fastavro.schemaless_reader(
                BytesIO(),
                MEASUREMENT_SCHEMA,
                return_union_branch=True,
                **{option: True},
            )


def test_block_reader_streams_branch_markers():
    fo = write_bytes(records_with_float_and_double())
    blocks = list(fastavro.block_reader(fo, return_union_branch=True))
    records = [record for block in blocks for record in block]
    assert [record["value"].branch for record in records] == ["float", "double"]

    # The same file read through a plain block reader is unchanged.
    fo.seek(0)
    plain = [record for block in fastavro.block_reader(fo) for record in block]
    assert [record["value"] for record in plain] == [1.25, 1.25]


def test_legacy_reader_behavior_unchanged():
    fo = write_bytes(
        [
            {"id": 1, "value": 1.25, "label": "a"},
            {"id": 2, "value": 1.25, "label": "b"},
        ]
    )
    assert read(fo) == [
        {"id": 1, "value": 1.25, "label": "a"},
        {"id": 2, "value": 1.25, "label": "b"},
    ]

    union_schema = {
        "type": "record",
        "name": "R",
        "fields": [
            {
                "name": "u",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "Rec",
                        "fields": [{"name": "a", "type": "int"}],
                    },
                ],
            }
        ],
    }
    fo = write_bytes([{"u": ("Rec", {"a": 1})}], union_schema)
    assert read(fo, return_record_name=True) == [{"u": ("Rec", {"a": 1})}]
    assert read(fo, return_named_type=True) == [{"u": ("Rec", {"a": 1})}]
    assert read(fo, return_record_name_override=True) == [{"u": {"a": 1}}]


def test_helpers():
    marker = make_union_value("float", 1.25, 0)
    assert is_union_value(marker)
    assert not is_union_value(1.25)
    assert get_union_branch(marker) == "float"
    assert get_union_value(marker) == 1.25
    assert marker.name == marker.branch
    assert marker.branch_index == marker.index

    # Equality ignores the index hint.
    assert marker == make_union_value("float", 1.25)
    assert marker != make_union_value("double", 1.25)

    with pytest.raises(TypeError):
        get_union_branch(1.25)
    with pytest.raises(ValueError):
        make_union_value("", 1)


def test_marker_for_tampered_index_uses_branch_name():
    schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    # An index hint that points at the wrong branch must not be trusted.
    fo = write_bytes([{"v": make_union_value("float", 1.25, 1)}], schema)
    assert union_indices_in_file(fo, 0) == [0]

    # And a marker whose branch does not exist at all is rejected.
    with pytest.raises(ValueError):
        write_bytes([{"v": make_union_value("bytes", b"x")}], schema)


def test_branch_write_after_union_reordered():
    schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    marker = make_union_value("double", 1.25, 0)
    fo = write_bytes([{"v": marker}], schema)
    assert union_indices_in_file(fo, 0) == [1]


def test_schemaless_roundtrip():
    schema = {
        "type": "record",
        "name": "R",
        "fields": [{"name": "v", "type": ["float", "double"]}],
    }
    fo = BytesIO()
    fastavro.schemaless_writer(fo, schema, {"v": make_union_value("float", 1.25)})
    fo.seek(0)
    record = fastavro.schemaless_reader(fo, schema, return_union_branch=True)
    assert record["v"].branch == "float"
    assert record["v"].value == 1.25
