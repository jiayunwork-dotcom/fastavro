"""Helpers shared by the pure Python and Cython write backends."""

import threading

_STATE = threading.local()


def _is_appendable(file_like):
    if file_like.seekable() and file_like.tell() != 0:
        if "<stdout>" == getattr(file_like, "name", ""):
            # In OSX, sys.stdout is seekable and has a non-zero tell() but
            # we wouldn't want to append to a stdout. In the python REPL,
            # sys.stdout is named `<stdout>`
            return False
        if file_like.readable():
            return True
        else:
            raise ValueError(
                "When appending to an avro file you must use the "
                + "'a+' mode, not just 'a'"
            )
    else:
        return False


def _begin_write_attempt():
    """Start a scope in which each logical value is prepared at most once.

    A single call to ``write()`` validates and then encodes one record.  Both
    phases can encounter the same logical type position, and union branch
    selection inspects logical record branches before one is chosen, so the
    user supplied logical writer would otherwise run several times per
    encoding.  Logical writers may have side effects, so the result is memoized
    for the duration of the attempt.  The memo is dropped afterwards; side
    effects of a failed attempt are deliberately not undone.
    """
    depth = getattr(_STATE, "depth", 0) + 1
    _STATE.depth = depth
    if depth == 1:
        _STATE.memo = {}
        _STATE.produced = {}


def _end_write_attempt():
    depth = getattr(_STATE, "depth", 1) - 1
    _STATE.depth = depth
    if depth <= 0:
        _STATE.depth = 0
        _STATE.memo = None
        _STATE.produced = None


def _prepare_logical(datum, schema, prepare):
    """Apply ``prepare`` to ``datum`` at most once per write attempt.

    Outside a write attempt (for example ``fastavro.validation.validate``)
    the logical writer is simply called, matching the historical behaviour.
    """
    memo = getattr(_STATE, "memo", None)
    if memo is None:
        return prepare(datum, schema)

    position = (id(prepare), id(schema))
    produced = _STATE.produced.get(position)
    # Union branch selection hands the value a logical writer produced back to
    # write_data for the selected branch.  That value is already encoded and
    # must not be converted a second time.
    if produced is not None and id(datum) in produced:
        return datum

    key = (id(prepare), id(schema), id(datum))
    entry = memo.get(key)
    if entry is not None and entry[0] is datum:
        prepared = entry[1]
    else:
        prepared = prepare(datum, schema)
        # Keep a reference to the original datum so its id cannot be reused by
        # an unrelated object during this attempt.
        memo[key] = (datum, prepared)

    if prepared is not datum:
        # Keep the produced value alive so its id stays unique for the rest of
        # the attempt.
        _STATE.produced.setdefault(position, {})[id(prepared)] = prepared
    return prepared


def _write_data_with_rollback(write_once, rollback):
    """Run one record attempt and restore the encoder state on failure.

    ``write_once`` performs validation and encoding of a single record.
    ``rollback`` restores the encoder to the state it had before the attempt;
    the original exception is always re-raised.  This is the shared acceptance
    boundary for both the binary and the JSON writer, so a record is either
    encoded completely or not at all.
    """
    _begin_write_attempt()
    try:
        write_once()
    except BaseException:
        try:
            rollback()
        except BaseException:
            # Never mask the original failure with a rollback problem.
            pass
        raise
    finally:
        _end_write_attempt()
