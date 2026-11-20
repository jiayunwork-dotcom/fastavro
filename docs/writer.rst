fastavro.write
==============

.. autofunction:: fastavro._write_py.writer

.. autofunction:: fastavro._write_py.schemaless_writer

Using the tuple notation to specify which branch of a union to take
-------------------------------------------------------------------

Since this library uses plain dictionaries to represent a record, it is
possible for that dictionary to fit the definition of two different records.

For example, given a dictionary like this::

    {"name": "My Name"}

It would be valid against both of these records::

    child_schema = {
        "name": "Child",
        "type": "record",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "favorite_color", "type": ["null", "string"]},
        ]
    }

    pet_schema = {
        "name": "Pet",
        "type": "record",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "favorite_toy", "type": ["null", "string"]},
        ]
    }

This becomes a problem when a schema contains a union of these two similar
records as it is not clear which record the dictionary represents. For example,
if you used the previous dictionary with the following schema, it wouldn't be
clear if the record should be serialized as a `Child` or a `Pet`::

    household_schema = {
        "name": "Household",
        "type": "record",
        "fields": [
            {"name": "address", "type": "string"},
            {
                "name": "family_members",
                "type": {
                    "type": "array", "items": [
                        {
                            "name": "Child",
                            "type": "record",
                            "fields": [
                                {"name": "name", "type": "string"},
                                {"name": "favorite_color", "type": ["null", "string"]},
                            ]
                        }, {
                            "name": "Pet",
                            "type": "record",
                            "fields": [
                                {"name": "name", "type": "string"},
                                {"name": "favorite_toy", "type": ["null", "string"]},
                            ]
                        }
                    ]
                }
            },
        ]
    }

To resolve this, you can use a tuple notation where the first value of the
tuple is the fully namespaced record name and the second value is the
dictionary. For example::

    records = [
        {
            "address": "123 Drive Street",
            "family_members": [
                ("Child", {"name": "Son"}),
                ("Child", {"name": "Daughter"}),
                ("Pet", {"name": "Dog"}),
            ]
        }
    ]

Using the record hint to specify which branch of a union to take
----------------------------------------------------------------

In addition to the tuple notation for specifying the name of a record, you can
also include a special `-type` attribute (note that this attribute is `-type`,
not `type`) on a record to do the same thing. So the example above which looked
like this::

    records = [
        {
            "address": "123 Drive Street",
            "family_members": [
                ("Child", {"name": "Son"}),
                ("Child", {"name": "Daughter"}),
                ("Pet", {"name": "Dog"}),
            ]
        }
    ]

Would now look like this::

    records = [
        {
            "address": "123 Drive Street",
            "family_members": [
                {"-type": "Child", "name": "Son"},
                {"-type": "Child", "name": "Daughter"},
                {"-type": "Pet", "name": "Dog"},
            ]
        }
    ]

Unlike the tuple notation which can be used with any avro type in a union, this
`-type` hint can only be used with records. However, this can be useful if you
want to make a single record dictionary that can be used both in and out of
unions.

Recovering from a rejected record
---------------------------------

``fastavro.write.Writer`` and the ``json_writer`` encoder both treat one
``write()`` call as an indivisible unit. If anything while encoding a record
raises -- an illegal enum symbol deep in a nested record, array or map, a value
the strict validator rejects, a logical type conversion that fails, or a union
branch marker that does not match its schema -- the exception propagates to the
caller and none of that record's bytes are kept. Records accepted before it are
preserved, and the same writer can then encode the next record.

For the binary ``Writer`` the rejected record is truncated out of the pending
in-memory block, so ``flush()`` (or an automatic block flush) after a failure
never publishes it and never rewrites blocks that were already written. For the
JSON encoder the syntax/parser state and the pending output are restored to the
boundary that preceded the failed record, so a later record starts from a clean
record boundary.

Logical type conversions are user supplied and may have side effects. Within a
single record encoding, a given logical value is converted at most once, even
when union branch selection inspects it before the branch is chosen. Side
effects of a conversion that runs and then fails are not undone; only the
writer's buffered output and syntax state are rolled back.

The batch ``writer(fo, schema, records)`` entry point still stops at the first
rejected record and re-raises. Before doing so it flushes the records that were
already accepted, so the file contains a valid prefix and the rejected record is
absent.
