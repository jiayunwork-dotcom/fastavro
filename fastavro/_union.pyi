from typing import Any, Dict, Optional

from .types import NamedSchemas, Schema

class UnionValue:
    branch: str
    value: Any
    index: Optional[int]
    def __init__(self, branch: str, value: Any, index: Optional[int] = ...) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def branch_index(self) -> Optional[int]: ...

class _UnionReadResult:
    index: int
    writer_schema: Schema
    reader_schema: Optional[Schema]
    value: Any
    def __init__(
        self,
        index: int,
        writer_schema: Schema,
        reader_schema: Optional[Schema],
        value: Any,
    ) -> None: ...

def check_union_branch_options(
    return_union_branch: bool, **other_options: Any
) -> None: ...
def finish_union_read(
    data: Any,
    writer_schema: Schema,
    original_reader_schema: Optional[Schema],
    resolved_reader_schema: Optional[Schema],
    named_schemas: Dict[str, Any],
) -> Any: ...
def is_union_value(datum: Any) -> bool: ...
def get_union_branch(datum: Any) -> str: ...
def get_union_value(datum: Any) -> Any: ...
def make_union_value(
    branch: str, value: Any, index: Optional[int] = ...
) -> UnionValue: ...
def branch_name(schema: Schema, named_schemas: Optional[NamedSchemas] = ...) -> str: ...
def find_branch_index(
    union_schema: list,
    branch: str,
    named_schemas: Optional[NamedSchemas] = ...,
) -> Optional[int]: ...
def resolve_marker_index(
    union_schema: list,
    marker: UnionValue,
    named_schemas: Optional[NamedSchemas] = ...,
) -> int: ...
def materialize_default(
    schema: Schema, default: Any, named_schemas: Dict[str, Schema]
) -> Any: ...
