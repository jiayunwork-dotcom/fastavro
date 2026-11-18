from typing import Any, Iterable

from .types import Schema

def validate(
    datum: Any,
    schema: Schema,
    field: str = ...,
    raise_errors: bool = ...,
    strict: bool = ...,
    disable_tuple_notation: bool = ...,
) -> bool: ...
def validate_many(
    records: Iterable[Any],
    schema: Schema,
    raise_errors: bool = ...,
    strict: bool = ...,
    disable_tuple_notation: bool = ...,
) -> bool: ...
