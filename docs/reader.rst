fastavro.read
=============

.. autoclass:: fastavro._read_py.reader

.. autoclass:: fastavro._read_py.block_reader

.. autoclass:: fastavro._read_py.Block

.. autofunction:: fastavro._read_py.schemaless_reader

.. autofunction:: fastavro._read_py.is_avro

.. autoclass:: fastavro._union.UnionValue

Keeping the union branch that was read
--------------------------------------

A plain Python value can be valid in more than one branch of a union.  The
canonical example is a record containing either a ``float`` or a ``double``
field: once the bytes have been decoded, ``1.25`` looks the same no matter
which branch produced it, and writing the record again applies fastavro's
normal union selection (which prefers ``double``).  If downstream consumers
distinguish report types by union branch, that rewrite silently changes the
report type.

Pass ``return_union_branch=True`` to ``reader``, ``block_reader`` or
``schemaless_reader`` to keep the branch::

    from fastavro import reader, UnionValue

    with open("measurements.avro", "rb") as fo:
        for record in reader(fo, return_union_branch=True):
            marker = record["value"]
            if isinstance(marker, UnionValue):
                print(marker.branch, marker.value)

Every union position is returned as a ``fastavro.UnionValue`` with:

``branch``
    The Avro type name for a primitive branch (``"float"``, ``"double"``,
    ``"null"``, ...) or the full schema name for a named branch
    (``"example.Measurement"``, enums and fixed included).
``value``
    The ordinary Python value of that branch.  Nested union positions inside
    ``value`` carry their own markers.
``index``
    The zero-based index of the branch in the union that produced the value.
    It is only a hint; ``branch`` is what identifies the branch when writing.

The same representation is produced in nested positions: array items, map
values and recursive records all keep their markers.  When a ``reader_schema``
is supplied the branch reported is the one selected by schema resolution, even
if the *writer* field was not a union; if the reader schema replaced the union
at that position with a plain type, the plain value is returned there instead.
Fields filled from reader-schema defaults are marked in the same way (a union
default uses the first branch that can hold it) and are validated and written
like any other field.

The markers can be handed straight back to ``writer`` or ``schemaless_writer``.
Business code can inspect or replace them without touching the rest of the
record::

    from fastavro import get_union_branch, get_union_value, make_union_value

    record = next(reader(fo, return_union_branch=True))
    branch = get_union_branch(record["value"])
    raw = get_union_value(record["value"])
    record["label"] = "corrected"          # unrelated plain field
    record["value"] = make_union_value("float", raw)

``validate`` (and the writer's internal validation) understands the markers:
the payload must be valid for the branch named by the marker.  A marker for a
branch that is not in the target union is rejected, and so is a payload that
would only be valid in a different branch.  This branch check also runs in
``writer`` even when the ``validator`` option is disabled.  With
``disable_tuple_notation=True``, plain tuples keep their usual meaning while
markers are still honoured.

``return_union_branch`` cannot be combined with ``return_record_name``,
``return_record_name_override``, ``return_named_type`` or
``return_named_type_override``.  Those options wrap values in tuples, which
would either nest the marker or hide it, so the combination raises
``ValueError`` at construction time rather than returning an ambiguous shape.

What it costs
~~~~~~~~~~~~~

For values
    Each union value costs one small ``UnionValue`` object per union position
    that is read, plus its payload.  Reading without the option is unchanged
    and pays nothing.  Since the marker owns its payload, code that expects a
    bare ``float``/``dict`` must unwrap it with ``get_union_value`` or
    ``.value``.  Equality is by ``(branch, value)`` so two markers compare equal
    exactly when they name the same branch and carry equal payloads.

For streaming
    The representation is inline in the record; there is no side table keyed by
    record identity, so ``reader`` still yields one record at a time and
    ``block_reader`` still yields one block at a time.  Neither retains records
    or blocks after they have been yielded.  The only extra allocation is the
    marker objects belonging to the record you are currently processing, which
    are released with that record.
