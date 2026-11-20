"""Writer failure recovery regression tests.

A record that the encoder rejects (for example an illegal enum symbol deep in a
nested record, array or map) must not be partially published, must surface the
error to the caller, and must leave both writer backends able to encode the
next record.  Records accepted before the failure must survive unchanged.

The tests run unchanged against the Cython extension and the pure Python
backend.  ``conftest.is_testing_pure_python()`` reports which one a run used::

    # Cython extension (built in place)
    python setup.py build_ext --inplace
    python -m pytest tests/test_write_recovery.py

    # Pure Python backend, no compiled modules
    rm -rf /tmp/fastavro-pure
    cp -r fastavro /tmp/fastavro-pure
    find /tmp/fastavro-pure -name '*.so' -delete
    PYTHONPATH=/tmp/fastavro-pure python -m pytest tests/test_write_recovery.py \
        --import-mode=importlib

``--import-mode=importlib`` matters when only ``PYTHONPATH`` points at the
pure copy: the default import mode prepends the test file's root directory
(which contains the built ``fastavro`` package) to ``sys.path``.
"""

import json
import os
from io import BytesIO, StringIO

import pytest

import fastavro
from fastavro import make_union_value, parse_schema
from fastavro._write_py import JSONWriter as PureJSONWriter
from fastavro._write_py import Writer as PureWriter
from fastavro.io.json_encoder import AvroJSONEncoder
from fastavro.validation import ValidationError

from .conftest import is_testing_pure_python

try:
    from fastavro.write import _write as _cython_write
except ImportError:  # pragma: no cover - pure Python checkout
    _cython_write = None

CYTHON_WRITER = (
    _cython_write.Writer
    if _cython_write is not None and hasattr(_cython_write, "CYTHON_MODULE")
    else None
)

CODECS = ["null", "deflate"]

ENUM = {"type": "enum", "name": "E", "symbols": ["OK"]}

RECORD_SCHEMA = {
    "type": "record",
    "name": "Record",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "v", "type": ENUM},
    ],
}


def backend_name():
    return "pure-python" if is_testing_pure_python() else "cython"


def writer_classes():
    """Return the built-in binary writers that are importable in this run.

    Both classes are exercised in either run mode: the Cython ``Writer`` when
    the extension is available and the pure Python ``Writer`` always.  The
    assertions are direct class construction, so the selected class is
    unambiguous.
    """
    classes = [("pure", PureWriter)]
    if CYTHON_WRITER is not None:
        classes.append(("cython", CYTHON_WRITER))
    return classes


def read_bytes(fo, **kwargs):
    fo.seek(0)
    return list(fastavro.reader(fo, **kwargs))


def read_json(fo, schema, **kwargs):
    fo.seek(0)
    return list(fastavro.json_reader(fo, schema, **kwargs))


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("backend", ["pure", "cython"])
def test_binary_writer_recovers_after_rejected_enum(backend, codec):
    classes = dict(writer_classes())
    if backend not in classes:
        pytest.skip("Cython writer is not available in this run")
    Writer = classes[backend]

    fo = BytesIO()
    w = Writer(fo, parse_schema(RECORD_SCHEMA), codec=codec)
    w.write({"id": 1, "v": "OK"})

    with pytest.raises(ValueError):
        w.write({"id": 99, "v": "BAD"})

    w.write({"id": 2, "v": "OK"})
    w.flush()

    assert read_bytes(fo) == [{"id": 1, "v": "OK"}, {"id": 2, "v": "OK"}]


@pytest.mark.parametrize("codec", CODECS)
def test_binary_writer_flush_right_after_failure_publishes_only_accepted(codec):
    fo = BytesIO()
    w = PureWriter(fo, parse_schema(RECORD_SCHEMA), codec=codec)
    w.write({"id": 1, "v": "OK"})
    with pytest.raises(ValueError):
        w.write({"id": 99, "v": "BAD"})

    # The pending block is flushed while the writer sits right after the
    # failure; the rejected record must not be part of it.
    w.flush()
    assert read_bytes(fo) == [{"id": 1, "v": "OK"}]

    w.write({"id": 2, "v": "OK"})
    w.flush()
    assert read_bytes(fo) == [{"id": 1, "v": "OK"}, {"id": 2, "v": "OK"}]


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("sync_interval", [0, 1, 1000 * 16000])
def test_binary_writer_autodump_boundary_is_not_crossed_by_failure(codec, sync_interval):
    fo = BytesIO()
    w = PureWriter(
        fo, parse_schema(RECORD_SCHEMA), codec=codec, sync_interval=sync_interval
    )
    w.write({"id": 1, "v": "OK"})
    size_before = fo.tell()
    count_before = w.block_count

    with pytest.raises(ValueError):
        w.write({"id": 99, "v": "BAD"})

    assert fo.tell() == size_before
    assert w.block_count == count_before

    w.write({"id": 2, "v": "OK"})
    w.flush()
    assert read_bytes(fo) == [{"id": 1, "v": "OK"}, {"id": 2, "v": "OK"}]


@pytest.mark.parametrize("codec", CODECS)
def test_binary_writer_recovers_inside_nested_record_array_and_map(codec):
    schema = {
        "type": "record",
        "name": "Deep",
        "fields": [
            {"name": "id", "type": "long"},
            {
                "name": "arr",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "Item",
                        "fields": [
                            {"name": "k", "type": "long"},
                            {"name": "e", "type": ENUM},
                        ],
                    },
                },
            },
            {
                "name": "mp",
                "type": {"type": "map", "values": {"type": "array", "items": "E"}},
            },
        ],
    }
    parsed = parse_schema(schema)

    bad = {"id": 9, "arr": [{"k": 1, "e": "OK"}, {"k": 2, "e": "BAD"}], "mp": {"a": ["OK"]}}
    # A failure in the map's array, after the record and array fields were
    # already encoded.
    bad_map = {"id": 8, "arr": [], "mp": {"a": ["OK", "NOPE"]}}

    fo = BytesIO()
    w = PureWriter(fo, parsed, codec=codec)
    w.write({"id": 1, "arr": [], "mp": {}})
    for record in (bad, bad_map):
        with pytest.raises(ValueError):
            w.write(record)
    w.write({"id": 2, "arr": [{"k": 3, "e": "OK"}], "mp": {"b": ["OK"]}})
    w.flush()

    assert read_bytes(fo) == [
        {"id": 1, "arr": [], "mp": {}},
        {"id": 2, "arr": [{"k": 3, "e": "OK"}], "mp": {"b": ["OK"]}},
    ]


@pytest.mark.parametrize("codec", CODECS)
def test_binary_writer_appends_after_reopen_following_failure(tmp_path, codec):
    path = tmp_path / "recover.avro"
    with open(path, "wb") as fo:
        w = PureWriter(fo, parse_schema(RECORD_SCHEMA), codec=codec)
        w.write({"id": 1, "v": "OK"})
        w.flush()

    with open(path, "a+b") as fo:
        w = PureWriter(fo, None, codec=codec)
        with pytest.raises(ValueError):
            w.write({"id": 99, "v": "BAD"})
        w.write({"id": 2, "v": "OK"})
        w.flush()

    with open(path, "rb") as fo:
        assert list(fastavro.reader(fo)) == [{"id": 1, "v": "OK"}, {"id": 2, "v": "OK"}]


def test_binary_writer_error_still_raised_and_reading_rejects_corruption():
    """The fix must not weaken the reader to make the happy path look right."""
    fo = BytesIO()
    w = PureWriter(fo, parse_schema(RECORD_SCHEMA))
    with pytest.raises(ValueError):
        w.write({"id": 99, "v": "BAD"})
    w.flush()

    # No record was accepted, so the file is header-only and readable.
    assert read_bytes(fo) == []

    # An accepted file whose enum index is corrupted must still be rejected by
    # the reader; the fix did not relax any read-side check.
    good = BytesIO()
    fastavro.writer(good, RECORD_SCHEMA, [{"id": 1, "v": "OK"}])
    data = bytearray(good.getvalue())
    # The enum index is the last record byte, directly before the 16 byte sync
    # marker.  0x08 is zig-zag 4, outside symbols == ["OK"].
    data[-17] = 0x08
    with pytest.raises(Exception):
        list(fastavro.reader(BytesIO(bytes(data))))


# ---------------------------------------------------------------------------
# JSON backend
# ---------------------------------------------------------------------------


def json_writer_for(schema, validator=False):
    fo = StringIO()
    writer = PureJSONWriter(AvroJSONEncoder(fo), schema, validator=validator)
    return fo, writer


@pytest.mark.parametrize("validator", [False, True])
def test_json_writer_recovers_after_rejected_enum(validator):
    fo, w = json_writer_for(parse_schema(RECORD_SCHEMA), validator=validator)
    w.write({"id": 1, "v": "OK"})
    with pytest.raises((ValueError, ValidationError)):
        w.write({"id": 99, "v": "BAD"})
    w.write({"id": 2, "v": "OK"})
    w.flush()

    assert read_json(fo, RECORD_SCHEMA) == [
        {"id": 1, "v": "OK"},
        {"id": 2, "v": "OK"},
    ]


def test_json_writer_recovers_after_two_consecutive_failures():
    fo, w = json_writer_for(parse_schema(RECORD_SCHEMA))
    w.write({"id": 1, "v": "OK"})
    for _ in range(2):
        with pytest.raises(ValueError):
            w.write({"id": 99, "v": "BAD"})
    w.write({"id": 2, "v": "OK"})
    w.flush()

    text = fo.getvalue()
    assert read_json(fo, RECORD_SCHEMA) == [
        {"id": 1, "v": "OK"},
        {"id": 2, "v": "OK"},
    ]
    # Every line is a complete JSON document.
    assert [json.loads(line) for line in text.splitlines()] == [
        {"id": 1, "v": "OK"},
        {"id": 2, "v": "OK"},
    ]


def test_json_writer_recovers_from_deep_failure():
    schema = {
        "type": "record",
        "name": "JsonDeep",
        "fields": [
            {"name": "id", "type": "long"},
            {
                "name": "arr",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "JsonItem",
                        "fields": [
                            {"name": "k", "type": "long"},
                            {"name": "e", "type": ENUM},
                        ],
                    },
                },
            },
            {"name": "mp", "type": {"type": "map", "values": {"type": "array", "items": "E"}}},
        ],
    }
    parsed = parse_schema(schema)
    fo, w = json_writer_for(parsed)

    good = {"id": 1, "arr": [], "mp": {}}
    w.write(good)
    # Fail in the array's last item and in the map's last array item, i.e.
    # after earlier siblings have already been encoded.
    with pytest.raises(ValueError):
        w.write({"id": 9, "arr": [{"k": 1, "e": "OK"}, {"k": 2, "e": "BAD"}], "mp": {}})
    with pytest.raises(ValueError):
        w.write({"id": 8, "arr": [], "mp": {"a": ["OK", "BAD"]}})
    w.write(good)
    w.flush()

    assert read_json(fo, schema) == [good, good]


def test_json_writer_custom_encoder_recovers():
    from fastavro.io.symbols import MapKeyMarker, String

    class UnderscoreEncoder(AvroJSONEncoder):
        def write_utf8(self, value):
            self._parser.advance(String())
            if self._parser.stack[-1] == MapKeyMarker():
                self._parser.advance(MapKeyMarker())
                self.write_object_key(value)
            else:
                self.write_value("_" + value)

    schema = {
        "type": "record",
        "name": "Custom",
        "fields": [{"name": "x", "type": "string"}],
    }
    fo = StringIO()
    w = PureJSONWriter(UnderscoreEncoder(fo), parse_schema(schema))
    w.write({"x": "a"})
    with pytest.raises(TypeError):
        w.write({"x": 123})
    w.write({"x": "b"})
    w.flush()

    assert read_json(fo, schema) == [{"x": "_a"}, {"x": "_b"}]


# ---------------------------------------------------------------------------
# Union branch markers
# ---------------------------------------------------------------------------


UNION_SCHEMA = {
    "type": "record",
    "name": "UnionRecord",
    "fields": [
        {
            "name": "u",
            "type": [
                "long",
                {
                    "type": "record",
                    "name": "Named",
                    "fields": [{"name": "x", "type": ENUM}],
                },
            ],
        }
    ],
}


@pytest.mark.parametrize("codec", CODECS)
def test_union_marker_keeps_branch_and_recovers_after_invalid_marker(codec):
    parsed = parse_schema(UNION_SCHEMA)
    valid = make_union_value("Named", {"x": "OK"})
    invalid = make_union_value("long", {"x": "OK"})

    fo = BytesIO()
    w = PureWriter(fo, parsed, codec=codec)
    w.write({"u": valid})
    with pytest.raises(ValueError):
        w.write({"u": invalid})
    # The same marker instance that failed can still be written where valid.
    w.write({"u": make_union_value("long", 5)})
    w.flush()

    records = read_bytes(fo, return_union_branch=True)
    assert records[0]["u"].branch == "Named"
    assert records[0]["u"].value == {"x": "OK"}
    assert records[1]["u"].branch == "long"
    assert records[1]["u"].value == 5


# ---------------------------------------------------------------------------
# Validation settings and input consumption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("validator", [False, True])
def test_strict_and_strict_allow_default_unchanged(validator):
    schema = {
        "type": "record",
        "name": "Strict",
        "fields": [
            {"name": "a", "type": "long"},
            {"name": "b", "type": "long", "default": 0},
        ],
    }

    def write(opts, record):
        fo = BytesIO()
        w = PureWriter(fo, parse_schema(schema), validator=validator, options=opts)
        w.write(record)
        w.flush()
        return read_bytes(fo)

    assert write({"strict": True}, {"a": 1, "b": 2}) == [{"a": 1, "b": 2}]
    with pytest.raises((ValueError, ValidationError)):
        write({"strict": True}, {"a": 1})
    with pytest.raises((ValueError, ValidationError)):
        write({"strict": True}, {"a": 1, "b": 2, "c": 3})

    assert write({"strict_allow_default": True}, {"a": 1}) == [{"a": 1, "b": 0}]
    with pytest.raises((ValueError, ValidationError)):
        write({"strict_allow_default": True}, {"b": 2})


class OneShotArray:
    """An array-like object with a length that can only be iterated once."""

    def __init__(self, items):
        self._items = list(items)
        self.iterations = 0

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise RuntimeError("array was consumed more than once")
        return iter(self._items)


@pytest.mark.parametrize("backend", ["pure", "cython"])
def test_one_shot_array_is_not_preconsumed_without_validator(backend):
    classes = dict(writer_classes())
    if backend not in classes:
        pytest.skip("Cython writer is not available in this run")
    Writer = classes[backend]

    schema = {
        "type": "record",
        "name": "OneShot",
        "fields": [{"name": "xs", "type": {"type": "array", "items": "long"}}],
    }
    source = OneShotArray([1, 2, 3])
    fo = BytesIO()
    w = Writer(fo, parse_schema(schema))
    w.write({"xs": source})
    w.flush()

    assert source.iterations == 1
    assert read_bytes(fo) == [{"xs": [1, 2, 3]}]


# ---------------------------------------------------------------------------
# Logical type conversions
# ---------------------------------------------------------------------------


def _register_logical(monkeypatch, name, fn):
    monkeypatch.setitem(fastavro.write.LOGICAL_WRITERS, name, fn)


@pytest.mark.parametrize("validator", [False, True])
def test_logical_conversion_runs_once_per_encoding(monkeypatch, validator):
    calls = []

    def prepare(data, schema):
        calls.append(data)
        return data + "!"

    _register_logical(monkeypatch, "string-counted", prepare)
    schema = {
        "type": "record",
        "name": "Logical",
        "fields": [
            {"name": "a", "type": "long"},
            {"name": "x", "type": {"type": "string", "logicalType": "counted"}},
        ],
    }
    fo = BytesIO()
    w = PureWriter(fo, parse_schema(schema), validator=validator)
    w.write({"a": 1, "x": "one"})
    assert calls == ["one"]

    calls.clear()
    w.write({"a": 2, "x": "two"})
    assert calls == ["two"]
    w.flush()

    assert read_bytes(fo) == [{"a": 1, "x": "one!"}, {"a": 2, "x": "two!"}]


@pytest.mark.parametrize("validator", [False, True])
def test_logical_conversion_in_union_runs_once_per_encoding(monkeypatch, validator):
    calls = []

    def prepare(data, schema):
        calls.append(data)
        return data + "!"

    _register_logical(monkeypatch, "string-counted", prepare)
    schema = {
        "type": "record",
        "name": "LogicalUnion",
        "fields": [
            {
                "name": "x",
                "type": [
                    "null",
                    {"type": "string", "logicalType": "counted"},
                ],
            }
        ],
    }
    fo = BytesIO()
    w = PureWriter(fo, parse_schema(schema), validator=validator)
    w.write({"x": "v"})
    assert calls == ["v"]
    w.flush()

    assert read_bytes(fo) == [{"x": "v!"}]


def test_logical_conversion_failure_recovers_and_is_not_memoized_across_attempts(
    monkeypatch,
):
    calls = []

    def prepare(data, schema):
        calls.append(data)
        if data == "bad":
            raise ValueError("logical boom")
        return data + "!"

    _register_logical(monkeypatch, "string-counted", prepare)
    schema = {
        "type": "record",
        "name": "LogicalFail",
        "fields": [
            {"name": "a", "type": "long"},
            {"name": "x", "type": {"type": "string", "logicalType": "counted"}},
        ],
    }
    fo = BytesIO()
    w = PureWriter(fo, parse_schema(schema))
    w.write({"a": 1, "x": "one"})

    calls.clear()
    for _ in range(2):
        with pytest.raises(ValueError, match="logical boom"):
            w.write({"a": 2, "x": "bad"})
    # Each failed attempt runs its own conversion once; the side effect is not
    # rolled back, but the failure is not swallowed either.
    assert calls == ["bad", "bad"]

    calls.clear()
    w.write({"a": 3, "x": "three"})
    assert calls == ["three"]
    w.flush()

    assert read_bytes(fo) == [
        {"a": 1, "x": "one!"},
        {"a": 3, "x": "three!"},
    ]


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def test_batch_writer_still_surfaces_the_error():
    fo = BytesIO()
    with pytest.raises(ValueError):
        fastavro.writer(
            fo,
            parse_schema(RECORD_SCHEMA),
            [{"id": 1, "v": "OK"}, {"id": 99, "v": "BAD"}, {"id": 2, "v": "OK"}],
        )
    # The batch writer stops at the error; the accepted record was still
    # published, and the rejected record was not.
    fo.seek(0)
    assert list(fastavro.reader(fo)) == [{"id": 1, "v": "OK"}]


def test_schemaless_writer_leaves_stream_untouched_on_failure():
    names = ["id", "v"]
    schema = {
        "type": "record",
        "name": "Schemaless",
        "fields": [
            {"name": names[0], "type": "long"},
            {"name": names[1], "type": ENUM},
        ],
    }
    parsed = parse_schema(schema)

    fo = BytesIO()
    with pytest.raises(ValueError):
        fastavro.schemaless_writer(fo, parsed, {"id": 99, "v": "BAD"})
    assert fo.getvalue() == b""

    # The success path is unchanged and round-trips through schemaless_reader.
    fo = BytesIO()
    fastavro.schemaless_writer(fo, parsed, {"id": 1, "v": "OK"})
    fo.seek(0)
    assert fastavro.schemaless_reader(fo, parsed) == {"id": 1, "v": "OK"}


def test_json_writer_recovers_for_non_record_root_schema():
    cases = [
        ("union", ["long", {"type": "enum", "name": "RootEnum", "symbols": ["OK"]}]),
        (
            "array",
            {
                "type": "array",
                "items": {"type": "enum", "name": "ArrayEnum", "symbols": ["OK"]},
            },
        ),
        (
            "map",
            {
                "type": "map",
                "values": {"type": "enum", "name": "MapEnum", "symbols": ["OK"]},
            },
        ),
    ]
    bad_values = {
        "union": "BAD",
        "array": ["OK", "BAD"],
        "map": {"a": "BAD"},
    }
    good_values = {"union": 7, "array": ["OK"], "map": {"a": "OK"}}

    for name, schema in cases:
        parsed = parse_schema(schema)
        fo = StringIO()
        w = PureJSONWriter(AvroJSONEncoder(fo), parsed)
        w.write(good_values[name])
        with pytest.raises(ValueError):
            w.write(bad_values[name])
        w.write(good_values[name])
        w.flush()
        assert read_json(fo, schema) == [good_values[name], good_values[name]]


def test_backend_is_reported():
    assert backend_name() in ("pure-python", "cython")
    assert os.environ.get("TEST_BACKEND", backend_name()) == backend_name()
