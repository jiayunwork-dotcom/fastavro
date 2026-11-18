"""Union branch identity helpers.

The regular readers throw away which branch of a union was used when a value
can be represented by more than one branch.  That is fine for most consumers
but it is lossy when the record is written back out: ``1.25`` fits both
``float`` and ``double`` and fastavro's normal union selection deliberately
prefers ``double``.

This module implements the opt-in ``return_union_branch`` read mode.  In that
mode every union position is represented by a :class:`UnionValue`, which keeps
the branch together with the value.  The marker travels with the record, so it
survives editing unrelated fields and can be consumed by ``writer``,
``schemaless_writer`` and ``validate``.

The helpers in this module are intentionally free of imports from the reader,
writer or validation modules.  They are shared by the pure Python and Cython
implementations.
"""

from typing import Any, Dict, Optional

from .const import AVRO_TYPES, NAMED_TYPES


class UnionValue:
    """A value together with the union branch that produced it.

    Parameters
    ----------
    branch
        Stable branch identity.  For primitive branches this is the Avro type
        name (for example ``"float"`` or ``"double"``).  For named branches
        (records, enums and fixed) this is the full schema name (for example
        ``"example.Measurement"``).
    value
        The plain Python value of the branch.
    index
        The zero-based position of the branch in the union schema that produced
        this marker.  It is optional for markers created by hand; the writer
        can resolve the branch from ``branch`` alone.

    The marker is deliberately not a tuple and not a subclass of the plain
    value.  This keeps it independent of ``disable_tuple_notation`` and avoids
    the ambiguity that ``1.25`` (or a tuple used as array data) would otherwise
    introduce.
    """

    __slots__ = ("branch", "value", "index")

    def __init__(self, branch: str, value: Any, index: Optional[int] = None):
        if not isinstance(branch, str) or not branch:
            raise ValueError("union branch must be a non-empty string")

        if index is not None:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("union branch index must be an integer or None")
            if index < 0:
                raise ValueError("union branch index must not be negative")

        self.branch = branch
        self.value = value
        self.index = index

    @property
    def name(self) -> str:
        """Alias for :attr:`branch`, useful for code familiar with tuples."""
        return self.branch

    @property
    def branch_index(self) -> Optional[int]:
        """Alias for :attr:`index`."""
        return self.index

    def __repr__(self) -> str:
        return (
            f"UnionValue(branch={self.branch!r}, value={self.value!r}, "
            f"index={self.index!r})"
        )

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, UnionValue):
            # The index is only a hint about the schema that produced the
            # marker.  Branch identity plus the payload is what matters.
            return self.branch == other.branch and self.value == other.value
        return NotImplemented

    __hash__ = None


class _UnionReadResult:
    """Internal carrier returned by ``read_union`` in branch read mode."""

    __slots__ = ("index", "writer_schema", "reader_schema", "value")

    def __init__(self, index, writer_schema, reader_schema, value):
        self.index = index
        self.writer_schema = writer_schema
        self.reader_schema = reader_schema
        self.value = value


_NAMED_RETURN_OPTIONS = (
    "return_record_name",
    "return_record_name_override",
    "return_named_type",
    "return_named_type_override",
)


def check_union_branch_options(return_union_branch: bool, **other_options: Any) -> None:
    """Reject combinations with the tuple-returning reader options.

    ``return_union_branch`` already carries the branch in a ``UnionValue``.
    Combining it with the older options would either nest tuples around the
    marker or hide it, so the combination is rejected explicitly instead of
    producing a surprising shape.
    """
    if not return_union_branch:
        return

    conflicts = sorted(
        name for name in _NAMED_RETURN_OPTIONS if other_options.get(name)
    )
    if conflicts:
        raise ValueError(
            "return_union_branch=True cannot be combined with "
            + ", ".join(f"{name}=True" for name in conflicts)
            + ". Branch mode returns union positions as fastavro.UnionValue "
            + "objects (branch plus value) instead of tuples."
        )


def _named_schema_halves(named_schemas):
    if (
        isinstance(named_schemas, dict)
        and isinstance(named_schemas.get("writer"), dict)
        and isinstance(named_schemas.get("reader"), dict)
    ):
        return named_schemas["writer"], named_schemas["reader"]
    named_schemas = named_schemas or {}
    return named_schemas, named_schemas


def finish_union_read(
    data: Any,
    writer_schema: Any,
    original_reader_schema: Any,
    resolved_reader_schema: Any,
    named_schemas: Dict[str, Any],
) -> Any:
    """Convert an internal union read result into the public read form.

    ``read_union`` returns a :class:`_UnionReadResult`.  This function decides
    whether the caller should see a ``UnionValue`` (the location is a union in
    the applicable schema) or a plain value (schema resolution collapsed the
    location to a single type).
    """
    writer_ns, reader_ns = _named_schema_halves(named_schemas)

    if isinstance(data, UnionValue):
        # Already marked at an inner (named schema) resolution step.
        return data

    if isinstance(data, _UnionReadResult):
        value = data.value
        if original_reader_schema is None:
            return UnionValue(
                branch_name(data.writer_schema, writer_ns),
                value,
                data.index,
            )
        if isinstance(original_reader_schema, list):
            resolved = (
                data.reader_schema
                if data.reader_schema is not None
                else resolved_reader_schema
            )
            index = branch_index_for_schema(original_reader_schema, resolved, reader_ns)
            return UnionValue(branch_name(resolved, reader_ns), value, index)
        # The reader schema collapsed this location to a single type.
        return value

    # The writer schema is not a union.  A union reader schema still selects a
    # branch during resolution and that target branch is what we report.
    if isinstance(original_reader_schema, list):
        index = branch_index_for_schema(
            original_reader_schema, resolved_reader_schema, reader_ns
        )
        return UnionValue(branch_name(resolved_reader_schema, reader_ns), data, index)

    return data


def is_union_value(datum: Any) -> bool:
    """Return True if *datum* is a :class:`UnionValue` marker."""
    return isinstance(datum, UnionValue)


def _as_union_value(datum: Any) -> UnionValue:
    if not isinstance(datum, UnionValue):
        raise TypeError(f"expected a UnionValue marker, got {type(datum)!r}")
    return datum


def get_union_branch(datum: Any) -> str:
    """Return the branch identity stored on *datum*."""
    return _as_union_value(datum).branch


def get_union_value(datum: Any) -> Any:
    """Return the plain value stored inside *datum*."""
    return _as_union_value(datum).value


def make_union_value(
    branch: str, value: Any, index: Optional[int] = None
) -> UnionValue:
    """Create a marker suitable for ``writer`` and ``validate``."""
    return UnionValue(branch, value, index)


def _resolve_named(schema: Any, named_schemas: Optional[Dict[str, Any]]) -> Any:
    """Resolve a named schema reference to its parsed definition, if possible."""
    seen = set()
    while isinstance(schema, str) and schema not in AVRO_TYPES:
        if not named_schemas or schema in seen:
            break
        seen.add(schema)
        resolved = named_schemas.get(schema)
        if resolved is None or resolved is schema:
            break
        schema = resolved
    return schema


def branch_name(schema: Any, named_schemas: Optional[Dict[str, Any]] = None) -> str:
    """Return the stable identity of a single union branch schema.

    Primitive types use their Avro type name.  Named types use their full name.
    ``named_schemas`` is a flat mapping as used by the writer/validator (or one
    of the ``writer``/``reader`` halves from the reader's nested mapping).
    """
    if isinstance(schema, str):
        if schema in AVRO_TYPES:
            return schema
        resolved = _resolve_named(schema, named_schemas)
        if isinstance(resolved, dict) and "name" in resolved:
            return resolved["name"]
        return schema

    if isinstance(schema, list):
        # Unions cannot be nested inside unions, but be defensive.
        return "union"

    schema_type = schema.get("type")
    if schema_type in NAMED_TYPES or "name" in schema:
        return schema.get("name", schema_type)
    return schema_type


def _alias_names(schema: Any, named_schemas: Optional[Dict[str, Any]]) -> list:
    resolved = _resolve_named(schema, named_schemas)
    if isinstance(resolved, dict):
        aliases = resolved.get("aliases", [])
        if isinstance(aliases, list):
            return [alias for alias in aliases if isinstance(alias, str)]
    return []


def find_branch_index(
    union_schema: list,
    branch: str,
    named_schemas: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Find the index of *branch* in *union_schema*, or return None.

    An exact branch name wins.  If there is no exact match, a branch whose
    schema aliases *branch* (or the unqualified form of it) is accepted, so a
    record produced from an older schema can be written with the renamed
    (aliased) schema.
    """
    found = [
        index
        for index, candidate in enumerate(union_schema)
        if branch_name(candidate, named_schemas) == branch
    ]
    if not found:
        unqualified = branch.rsplit(".", 1)[-1]
        found = [
            index
            for index, candidate in enumerate(union_schema)
            if branch in _alias_names(candidate, named_schemas)
            or unqualified in _alias_names(candidate, named_schemas)
        ]

    if len(found) > 1:
        raise ValueError(
            f"union branch {branch!r} is ambiguous in schema {union_schema}"
        )
    return found[0] if found else None


def branch_index_for_schema(
    union_schema: list,
    resolved_schema: Any,
    named_schemas: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Locate the reader-union branch chosen by schema resolution."""
    for index, candidate in enumerate(union_schema):
        if candidate is resolved_schema:
            return index

    # Parsed schemas are usually the very same objects.  The fallback covers
    # callers that pass a copy (or a named schema reference that resolution
    # returned as a full name).
    for index, candidate in enumerate(union_schema):
        try:
            if candidate == resolved_schema:
                return index
        except Exception:
            pass

    branch = branch_name(resolved_schema, named_schemas)
    return find_branch_index(union_schema, branch, named_schemas)


def resolve_marker_index(
    union_schema: list,
    marker: UnionValue,
    named_schemas: Optional[Dict[str, Any]] = None,
) -> int:
    """Return the target union index for *marker*.

    The branch name is authoritative: it lets a record survive reordering of
    the target union.  The stored index is used as a fast path only when it
    still points at the same branch.
    """
    if not isinstance(marker, UnionValue):
        raise TypeError(f"expected a UnionValue marker, got {type(marker)!r}")

    index = marker.index
    if index is not None and 0 <= index < len(union_schema):
        if branch_name(union_schema[index], named_schemas) == marker.branch:
            return index

    found = find_branch_index(union_schema, marker.branch, named_schemas)
    if found is not None:
        return found

    raise ValueError(
        f"provided union branch {marker.branch!r} not found in schema "
        f"{union_schema}"
    )


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def _default_matches(schema: Any, default: Any, named_schemas: Dict[str, Any]) -> bool:
    """Best-effort check that *default* can inhabit *schema*.

    This mirrors the lenient interpretation used by ``parse_schema`` for
    defaults.  It is only used to pick the branch of a union default, never to
    reject a schema.
    """
    schema = _resolve_named(schema, named_schemas)

    if isinstance(schema, list):
        return any(
            _default_matches(candidate, default, named_schemas) for candidate in schema
        )

    if isinstance(schema, str):
        if schema == "null":
            return default is None
        if schema == "boolean":
            return isinstance(default, bool)
        if schema == "string":
            return isinstance(default, str)
        if schema == "bytes":
            return isinstance(default, (str, bytes, bytearray))
        if schema in ("int", "long"):
            return isinstance(default, int) and not isinstance(default, bool)
        if schema in ("float", "double"):
            return isinstance(default, (int, float)) and not isinstance(default, bool)
        # A named reference that could not be resolved.
        return True

    if not isinstance(schema, dict):
        return True

    schema_type = schema["type"]
    if schema_type == "enum":
        return isinstance(default, str) and default in schema.get("symbols", [])
    if schema_type == "fixed":
        return isinstance(default, (str, bytes, bytearray))
    if schema_type in ("record", "error", "request"):
        return isinstance(default, dict)
    if schema_type == "array":
        return isinstance(default, (list, tuple)) and all(
            _default_matches(schema["items"], item, named_schemas) for item in default
        )
    if schema_type == "map":
        return (
            isinstance(default, dict)
            and all(isinstance(key, str) for key in default)
            and all(
                _default_matches(schema["values"], value, named_schemas)
                for value in default.values()
            )
        )
    return _default_matches(schema_type, default, named_schemas)


def _coerce_binary_default(
    schema: Any, value: Any, named_schemas: Dict[str, Any]
) -> Any:
    """Encode Avro bytes/fixed defaults, which are JSON strings (ISO-8859-1).

    ``parse_schema`` accepts such defaults as strings.  The branch form is meant
    to be validated and written like a value read from the file, so convert it
    to the ``bytes`` payload the reader would have produced.
    """
    resolved = _resolve_named(schema, named_schemas)
    if isinstance(resolved, dict) and resolved.get("type") in ("bytes", "fixed"):
        if isinstance(value, str):
            return value.encode("iso-8859-1")
    elif resolved == "bytes" and isinstance(value, str):
        return value.encode("iso-8859-1")
    return value


def materialize_default(
    schema: Any,
    default: Any,
    named_schemas: Dict[str, Any],
) -> Any:
    """Turn a reader-schema default into a value containing branch markers.

    Union defaults resolve to the first branch that can hold the default.  Per
    the Avro specification that is normally the first branch anyway.  Nested
    containers are walked so that union positions inside record, array and map
    defaults are marked too.
    """
    schema = _resolve_named(schema, named_schemas)

    if isinstance(schema, list):
        selected_index = None
        for index, candidate in enumerate(schema):
            if _default_matches(candidate, default, named_schemas):
                selected_index = index
                break

        if selected_index is None:
            # Keep the read permissive, as older callers may have schemas whose
            # defaults parse_schema accepted leniently.  The first branch is
            # the Avro-specified location for a union default.
            selected_index = 0

        selected = schema[selected_index]
        value = materialize_default(selected, default, named_schemas)
        value = _coerce_binary_default(selected, value, named_schemas)
        return UnionValue(
            branch_name(selected, named_schemas),
            value,
            selected_index,
        )

    if not isinstance(schema, dict):
        return default

    schema_type = schema["type"]
    if schema_type in ("record", "error", "request") and isinstance(default, dict):
        fields = {field["name"]: field for field in schema.get("fields", [])}
        result = dict(default)
        for name, value in default.items():
            if name in fields:
                result[name] = materialize_default(
                    fields[name]["type"], value, named_schemas
                )
        return result

    if schema_type == "array" and isinstance(default, (list, tuple)):
        return [
            _coerce_binary_default(
                schema["items"],
                materialize_default(schema["items"], item, named_schemas),
                named_schemas,
            )
            for item in default
        ]

    if schema_type == "map" and isinstance(default, dict):
        return {
            key: _coerce_binary_default(
                schema["values"],
                materialize_default(schema["values"], value, named_schemas),
                named_schemas,
            )
            for key, value in default.items()
        }

    return _coerce_binary_default(schema, default, named_schemas)
