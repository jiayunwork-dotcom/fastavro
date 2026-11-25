"""Caching and conversion helpers for Avro field defaults.

A schema default is stored in its JSON/Avro form.  The three Python APIs need
different forms:

* reading returns logical Python values (for example ``datetime.date``);
* writing uses the underlying Avro values after logical-type preparation;
* validation checks that same prepared form.

The conversions are memoized on the parsed field.  Mutable logical values are
copied before being returned to a record so that changing one record cannot
change the schema cache or another record.
"""

import datetime
import decimal
from typing import Any, Dict

from .logical_readers import LOGICAL_READERS
from .logical_writers import LOGICAL_WRITERS
from .schema import extract_logical_type, extract_record_type

DEFAULT_CACHE_KEY = "__fastavro_default_cache__"
_LOGICAL_KEY = "logical"
_WRITER_KEY = "writer"
_UNSET = object()


class PreparedDefault:
    """Marker for a default that has already passed through LOGICAL_WRITERS."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


def _cache(field: Any) -> Dict[str, Any]:
    cache = getattr(field, "default_cache", None)
    if isinstance(cache, dict):
        return cache

    cache = {}
    try:
        field.default_cache = cache
    except AttributeError:
        # Direct callers may pass a plain, unparsed field dictionary.
        existing = field.get(DEFAULT_CACHE_KEY)
        if isinstance(existing, dict):
            return existing
        field[DEFAULT_CACHE_KEY] = cache
    return cache


def _resolve_schema(schema: Any, named_schemas: Any) -> Any:
    """Resolve a named schema when a mapping of named schemas is available."""
    seen = set()
    while isinstance(schema, str) and schema not in seen:
        seen.add(schema)
        found = None
        if isinstance(named_schemas, dict):
            # read_record uses {"writer": ..., "reader": ...}; write and
            # validation use a flat mapping.
            reader_schemas = named_schemas.get("reader")
            if isinstance(reader_schemas, dict):
                found = reader_schemas.get(schema)
            if found is None:
                writer_schemas = named_schemas.get("writer")
                if isinstance(writer_schemas, dict):
                    found = writer_schemas.get(schema)
            if found is None:
                found = named_schemas.get(schema)
        if found is None:
            break
        schema = found
    return schema


def _json_default_to_logical(default: Any, schema: Any, named_schemas: Any) -> Any:
    schema = _resolve_schema(schema, named_schemas)
    record_type = extract_record_type(schema)

    if record_type == "array":
        return [
            _json_default_to_logical(item, schema["items"], named_schemas)
            for item in default
        ]

    if record_type == "map":
        return {
            key: _json_default_to_logical(value, schema["values"], named_schemas)
            for key, value in default.items()
        }

    if record_type in ("record", "error", "request"):
        field_types = {field["name"]: field["type"] for field in schema["fields"]}
        return {
            key: _json_default_to_logical(
                value, field_types.get(key, value), named_schemas
            )
            for key, value in default.items()
        }

    # Union defaults are handled by the union resolution code in the other
    # paths.  Keep their raw form here, but copy_container() still isolates
    # mutable defaults for readers.
    if record_type == "union":
        return default

    value = default

    # Avro JSON represents bytes/fixed as strings whose Unicode code points
    # are the unsigned byte values.
    if record_type in ("bytes", "fixed") and isinstance(value, str):
        value = value.encode("iso-8859-1")

    logical_type = extract_logical_type(schema)
    if logical_type:
        # Some schemas written by hand use the decimal's text form rather
        # than the spec-defined latin-1 encoding of its two's-complement bytes.
        # Accept that extension when the spec interpretation does not fit.
        if (
            logical_type in ("bytes-decimal", "fixed-decimal")
            and isinstance(default, str)
        ):
            physical = default.encode("iso-8859-1")
            reader = LOGICAL_READERS.get(logical_type)
            candidate = (
                reader(physical, schema, schema) if reader is not None else physical
            )
            if _decimal_fits_schema(candidate, schema):
                return candidate
            try:
                return decimal.Decimal(default)
            except decimal.InvalidOperation:
                return candidate

        # If the schema was supplied with an already-Python logical default,
        # do not send it through a reader expecting the physical Avro value.
        if not _is_logical_python_value(value, logical_type):
            reader = LOGICAL_READERS.get(logical_type)
            if reader is not None:
                value = reader(value, schema, schema)

    return value


def _decimal_fits_schema(value: Any, schema: Dict[str, Any]) -> bool:
    if not isinstance(value, decimal.Decimal):
        return False
    precision = schema["precision"]
    scale = schema.get("scale", 0)
    _, digits, exponent = value.as_tuple()
    return exponent == -scale and len(digits) <= precision


def _is_logical_python_value(value: Any, logical_type: str) -> bool:
    if logical_type == "int-date":
        return isinstance(value, datetime.date)
    if logical_type in ("bytes-decimal", "fixed-decimal"):
        return isinstance(value, decimal.Decimal)
    if logical_type == "string-uuid":
        import uuid

        return isinstance(value, uuid.UUID)
    if logical_type in (
        "long-timestamp-millis",
        "long-timestamp-micros",
        "long-local-timestamp-millis",
        "long-local-timestamp-micros",
    ):
        return isinstance(value, datetime.datetime)
    if logical_type in ("int-time-millis", "long-time-micros"):
        return isinstance(value, datetime.time)
    return False


def _raw_default_to_writer(default: Any, schema: Any, named_schemas: Any) -> Any:
    """Prepare a stored Avro/JSON default for the writer/validator paths."""
    schema = _resolve_schema(schema, named_schemas)
    record_type = extract_record_type(schema)

    if record_type == "array":
        return [
            _raw_default_to_writer(item, schema["items"], named_schemas)
            for item in default
        ]

    if record_type == "map":
        return {
            key: _raw_default_to_writer(item, schema["values"], named_schemas)
            for key, item in default.items()
        }

    if record_type in ("record", "error", "request"):
        field_types = {field["name"]: field["type"] for field in schema["fields"]}
        return {
            key: _raw_default_to_writer(
                item, field_types.get(key, item), named_schemas
            )
            for key, item in default.items()
        }

    if record_type == "union":
        return default

    value = default

    # Avro JSON bytes/fixed defaults use latin-1 code points for byte values.
    if record_type in ("bytes", "fixed") and isinstance(value, str):
        value = value.encode("iso-8859-1")

    logical_type = extract_logical_type(schema)
    if logical_type:
        # Hand-written decimal defaults are occasionally given as decimal text
        # rather than the specification's latin-1 encoded two's-complement bytes.
        if (
            logical_type in ("bytes-decimal", "fixed-decimal")
            and isinstance(default, str)
        ):
            reader = LOGICAL_READERS.get(logical_type)
            candidate = reader(value, schema, schema) if reader is not None else value
            if not _decimal_fits_schema(candidate, schema):
                try:
                    value = decimal.Decimal(default)
                except decimal.InvalidOperation:
                    value = candidate

        writer = LOGICAL_WRITERS.get(logical_type)
        if writer is not None:
            value = writer(value, schema)
        return PreparedDefault(value)

    return value


def get_logical_default(field: Dict[str, Any], named_schemas: Any = None) -> Any:
    """Return the cached logical Python form of a field's default."""
    cache = _cache(field)
    value = cache.get(_LOGICAL_KEY, _UNSET)
    if value is _UNSET:
        value = _json_default_to_logical(
            field["default"], field["type"], named_schemas
        )
        cache[_LOGICAL_KEY] = value
    return value


def get_writer_default(field: Dict[str, Any], named_schemas: Any = None) -> Any:
    """Return the cached prepared form used by writing and validation."""
    cache = _cache(field)
    value = cache.get(_WRITER_KEY, _UNSET)
    if value is _UNSET:
        value = _raw_default_to_writer(
            field["default"], field["type"], named_schemas
        )
        cache[_WRITER_KEY] = value
    return value


def copy_default_value(value: Any) -> Any:
    """Recursively copy Avro containers while sharing immutable scalars."""
    if isinstance(value, list):
        return [copy_default_value(item) for item in value]
    if isinstance(value, dict):
        return {key: copy_default_value(item) for key, item in value.items()}
    return value
