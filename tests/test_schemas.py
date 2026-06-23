"""Validation gates for the packaged input schemas.

Each dataset's ``input_schema.json`` must be a well-formed Draft-07 JSON
Schema and must not contain a "vacuous ``if``" on an optional property.

An ``if`` that constrains a property through ``properties`` but does not
list that property in the ``if``'s own ``required`` matches *vacuously* when
the property is absent (``properties`` only constrains properties that are
present). The clause then misfires for assets that simply omit the property:
a ``then``-only clause imposes a spurious requirement, and an ``if``/``else``
clause routes the asset to the wrong branch.

A clause is flagged unless the constrained property is *guaranteed present*
wherever the clause is evaluated -- that is, the property appears in:

- the schema's top-level ``required``;
- a ``required`` sibling of the clause holding the ``if``; or
- the ``required`` of an enclosing ``if`` whose ``then`` contains the clause
  -- the "skip when absent" guard ``{"if": {"required": ["X"]}, "then":
  {<clause on X>}}``, which reaches the inner clause only when ``X`` is
  present.

The fix for a flagged clause is correspondingly one of: add the property to
the ``if``'s own ``required`` (fine for a ``then``-only clause), or wrap the
clause in such an outer guard (the right fix for an ``if``/``else`` clause,
which keeps its branch routing intact).

These tests are pelicun-free and read the packaged data directly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jsonschema
import pytest

from dlml._catalog import data_root

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_SCHEMA_COUNT = 2


def _discover_schemas() -> list[Path]:
    """Find every ``input_schema.json`` under the data tree, sorted."""
    root = data_root()
    return sorted(
        root.rglob('input_schema.json'),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def _vacuous_if_violations(schema: dict) -> list[str]:
    """
    Return descriptions of vacuous ``if`` clauses on optional properties.

    Walks the whole schema, threading the set of properties *guaranteed
    present* at each point. For every ``if`` subschema, any property it
    constrains via ``properties`` but neither lists in the ``if``'s
    ``required`` nor has guaranteed present is reported. A property is
    guaranteed present when it is in the top-level ``required``, in a
    ``required`` sibling of the clause, or in the ``required`` of an
    enclosing ``if`` whose ``then`` we have descended into (see the module
    docstring). Both ``then``-only and ``if``/``else`` clauses are checked.

    Only ``properties`` is treated as constraining a present-only property;
    ``patternProperties``/``dependentRequired``/``dependencies`` are not
    inspected (no packaged schema uses them inside an ``if``).

    Parameters
    ----------
    schema : dict
        The parsed JSON schema.

    Returns
    -------
    list of str
        One message per offending property, empty when the schema is clean.

    """
    violations: list[str] = []

    def walk(node: object, path: str, guaranteed: set[str]) -> None:
        if isinstance(node, dict):
            here = guaranteed | set(node.get('required', []))
            sub_if = node.get('if')
            if_required: set[str] = set()
            if isinstance(sub_if, dict):
                constrained = set(sub_if.get('properties', {}))
                if_required = set(sub_if.get('required', []))
                violations.extend(
                    f"{path}/if constrains optional property '{prop}' "
                    f'without guaranteeing its presence'
                    for prop in sorted(constrained - if_required - here)
                )
            # Descending into ``then`` means the sibling ``if`` held, so its
            # required properties are guaranteed present there; ``else`` and
            # every other keyword inherit only ``here``.
            for key, value in node.items():
                child = here | if_required if key == 'then' else here
                walk(value, f'{path}/{key}', child)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f'{path}[{index}]', guaranteed)

    walk(schema, '$', set())
    return violations


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Real schemas
# ---------------------------------------------------------------------------


def test_discovery_finds_the_input_schemas():
    """Guard the parametrized tests against empty discovery."""
    assert len(_discover_schemas()) == _EXPECTED_SCHEMA_COUNT


@pytest.mark.parametrize(
    'schema_path',
    _discover_schemas(),
    ids=lambda p: p.relative_to(data_root()).as_posix(),
)
def test_input_schema_is_valid_draft7(schema_path: Path):
    """Each input schema is a well-formed Draft-07 JSON Schema."""
    jsonschema.Draft7Validator.check_schema(_load_schema(schema_path))


@pytest.mark.parametrize(
    'schema_path',
    _discover_schemas(),
    ids=lambda p: p.relative_to(data_root()).as_posix(),
)
def test_input_schema_has_no_vacuous_if(schema_path: Path):
    """No ``if`` conditions a requirement on an optional, unrequired property."""
    assert _vacuous_if_violations(_load_schema(schema_path)) == []


# ---------------------------------------------------------------------------
# The lint actually works (self-tests on synthetic schemas)
# ---------------------------------------------------------------------------


def test_vacuous_if_flags_optional_property():
    """An ``if`` on a property that can be absent is flagged."""
    schema = {
        'required': ['A'],
        'allOf': [
            {'if': {'properties': {'B': {'const': True}}}, 'then': {}},
        ],
    }
    violations = _vacuous_if_violations(schema)
    assert len(violations) == 1
    assert "'B'" in violations[0]


def test_vacuous_if_ignores_top_level_required_property():
    """An ``if`` on an always-present (top-level required) property is benign."""
    schema = {
        'required': ['A'],
        'allOf': [
            {'if': {'properties': {'A': {'const': True}}}, 'then': {}},
        ],
    }
    assert _vacuous_if_violations(schema) == []


def test_vacuous_if_ignores_clause_that_guards_the_property():
    """An ``if`` that lists the property in its own ``required`` is fine."""
    schema = {
        'required': [],
        'allOf': [
            {
                'if': {'properties': {'B': {'const': True}}, 'required': ['B']},
                'then': {},
            },
        ],
    }
    assert _vacuous_if_violations(schema) == []


def test_unguarded_if_else_on_optional_property_is_flagged():
    """An ``if``/``else`` whose trigger can be absent misroutes -- flagged."""
    schema = {
        'required': [],
        'allOf': [
            {
                'if': {'properties': {'B': {'const': True}}},
                'then': {},
                'else': {},
            },
        ],
    }
    violations = _vacuous_if_violations(schema)
    assert len(violations) == 1
    assert "'B'" in violations[0]


def test_if_else_guarded_by_outer_required_is_not_flagged():
    """The skip-when-absent guard around an ``if``/``else`` clears it."""
    schema = {
        'required': [],
        'allOf': [
            {
                'if': {'required': ['B']},
                'then': {
                    'if': {'properties': {'B': {'const': True}}},
                    'then': {},
                    'else': {},
                },
            },
        ],
    }
    assert _vacuous_if_violations(schema) == []


def test_vacuous_if_flags_each_offending_property_separately():
    """Every offending property in a multi-property ``if`` is reported once."""
    schema = {
        'required': ['A'],
        'allOf': [
            {
                'if': {
                    'properties': {
                        'A': {'const': 1},
                        'B': {'const': 1},
                        'C': {'const': 1},
                    }
                },
                'then': {},
            },
        ],
    }
    violations = _vacuous_if_violations(schema)
    assert len(violations) == 2  # A is top-level required -> excluded
    assert any("'B'" in message for message in violations)
    assert any("'C'" in message for message in violations)


def test_vacuous_if_ignored_when_sibling_required_forces_presence():
    """A ``required`` sibling that forces the property present is benign."""
    schema = {
        'required': [],
        'allOf': [
            {
                'required': ['B'],
                'if': {'properties': {'B': {'const': True}}},
                'then': {},
            },
        ],
    }
    assert _vacuous_if_violations(schema) == []


def test_else_branch_does_not_inherit_the_if_guard():
    """The ``if``'s ``required`` guards only ``then`` -- a vacuous clause in the
    ``else`` (where the ``if`` was false, so the property may be absent) is
    still flagged."""
    schema = {
        'required': [],
        'allOf': [
            {
                'if': {'required': ['B']},
                'then': {},
                'else': {
                    'if': {'properties': {'B': {'const': True}}},
                    'then': {},
                },
            },
        ],
    }
    violations = _vacuous_if_violations(schema)
    assert len(violations) == 1
    assert "'B'" in violations[0]


def test_then_guard_credits_only_if_required_not_if_properties():
    """Descending into ``then`` credits the ``if``'s ``required`` only; a
    ``properties`` constraint on the trigger does not guarantee presence, so the
    outer and the nested vacuous ``if`` on ``B`` are both flagged."""
    schema = {
        'required': [],
        'allOf': [
            {
                'if': {'properties': {'B': {'const': True}}},
                'then': {
                    'if': {'properties': {'B': {'const': True}}},
                    'then': {},
                },
            },
        ],
    }
    assert len(_vacuous_if_violations(schema)) == 2
