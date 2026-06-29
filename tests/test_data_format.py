"""Format and value gates for the packaged model parameters and metadata.

For every dataset and every collection it provides, the parsed parameters must
load into a well-formed tidy table whose columns are all recognized, whose
required structure is present, and whose every cell carries a valid value.
Metadata that ships alongside must document every model. These checks are
pelicun-free and read the packaged data through the public API.

The contract below is a verified floor: it captures the structure and value
domains every packaged dataset satisfies, locked to the vocabulary observed in
the data (broader pelicun support is enabled deliberately, by extending these
rules, rather than left open). Demand-type and distribution-family membership
is enforced in ``test_vocabulary`` (every value in the data is in
:mod:`dlml.vocabulary`); this module enforces the remaining value domains.
"""

from __future__ import annotations

import re

import jsonschema
import pandas as pd
import pytest

import dlml
from dlml import _catalog

# Every (dataset, collection) pair that provides parameters, and the subset
# that also ships metadata. Built once so a parametrized test cannot pass
# vacuously by discovering nothing (guarded by the counts below).
_PAIRS = [
    (dataset, collection)
    for dataset in dlml.list_datasets()
    for collection in dlml.available_collections(dataset)
]
_PAIR_IDS = [f'{collection}::{dataset}' for dataset, collection in _PAIRS]


def _has_metadata(dataset: str, collection: str) -> bool:
    """Return whether a collection ships a metadata JSON (no parse)."""
    try:
        _catalog.metadata_path(dataset, collection)
    except dlml.DatasetFileNotFoundError:
        return False
    return True


_METADATA_PAIRS = [pair for pair in _PAIRS if _has_metadata(*pair)]
_METADATA_PAIR_IDS = [
    f'{collection}::{dataset}' for dataset, collection in _METADATA_PAIRS
]

# Pairs whose models carry nested limit/damage states (fragility LimitStates,
# consequence DamageStates) -- loss models have none. The value is
# ``(metadata key, CSV group prefix)``. Used by the metadata/CSV cross-check.
_NESTED_STATES = {
    'fragility': ('LimitStates', 'LS'),
    'consequence_repair': ('DamageStates', 'DS'),
}
_NESTED_PAIRS = [pair for pair in _METADATA_PAIRS if pair[1] in _NESTED_STATES]
_NESTED_PAIR_IDS = [
    f'{collection}::{dataset}' for dataset, collection in _NESTED_PAIRS
]

# Tripwires: if the packaged data changes shape, these flag it for review.
_EXPECTED_PAIR_COUNT = 18
_EXPECTED_METADATA_PAIR_COUNT = 16
_EXPECTED_NESTED_PAIR_COUNT = 15

# Distribution families that additionally require a dispersion parameter
# (Theta_1); the rest (deterministic, multilinear_CDF, empirical, ...) define
# their parameter through Theta_0 alone. Drives the completeness rule.
_TWO_PARAMETER_FAMILIES = frozenset(
    {'normal', 'normal_cov', 'normal_std', 'lognormal', 'weibull', 'uniform'}
)

# Per-collection column contract.
# ``fixed_columns`` maps a constant group to the sub-columns it may expose;
# ``numbered_prefix``/``numbered_subcolumns`` describe the repeated groups
# (``LS<n>`` / ``DS<n>``).
# ``required_subcolumns`` is the must-be-present floor;
# ``requires_numbered_group`` demands at least one repeated group.
# ``theta0_strictly_positive`` is True where a scalar Theta_0 is a
# capacity/median (must be > 0) and False where it is a cost/time consequence
# (0 is allowed).
_COLLECTION_RULES = {
    'fragility': {
        'index_names': ('ID',),
        'fixed_columns': {
            'Demand': frozenset({'Directional', 'Offset', 'Type', 'Unit'}),
            'Incomplete': frozenset({''}),
        },
        'numbered_prefix': 'LS',
        'numbered_subcolumns': frozenset(
            {'Family', 'Theta_0', 'Theta_1', 'DamageStateWeights'}
        ),
        'required_subcolumns': {'Demand': frozenset({'Type', 'Unit'})},
        'requires_numbered_group': True,
        'theta0_strictly_positive': True,
    },
    'consequence_repair': {
        'index_names': (None, None),
        'fixed_columns': {
            'Quantity': frozenset({'Unit'}),
            'DV': frozenset({'Unit'}),
            'Incomplete': frozenset({''}),
        },
        'numbered_prefix': 'DS',
        'numbered_subcolumns': frozenset(
            {'Family', 'Theta_0', 'Theta_1', 'LongLeadTime'}
        ),
        'required_subcolumns': {
            'Quantity': frozenset({'Unit'}),
            'DV': frozenset({'Unit'}),
        },
        'requires_numbered_group': True,
        'theta0_strictly_positive': False,
    },
    'loss_repair': {
        'index_names': (None, None),
        'fixed_columns': {
            'Demand': frozenset({'Directional', 'Offset', 'Type', 'Unit'}),
            'DV': frozenset({'Unit'}),
            'Incomplete': frozenset({''}),
            'LossFunction': frozenset({'Theta_0'}),
        },
        'numbered_prefix': None,
        'numbered_subcolumns': frozenset(),
        'required_subcolumns': {
            'Demand': frozenset({'Type', 'Unit'}),
            'DV': frozenset({'Unit'}),
            'LossFunction': frozenset({'Theta_0'}),
        },
        'requires_numbered_group': False,
        'theta0_strictly_positive': True,
    },
}


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _is_float(value: object) -> bool:
    """Whether ``value`` parses as a float."""
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_binary(value: object) -> bool:
    """Whether ``value`` is the flag 0 or 1."""
    return _is_float(value) and float(value) in (0.0, 1.0)


def _is_whole_number(value: object) -> bool:
    """Whether ``value`` is an integer-valued number (e.g. a story offset)."""
    return _is_float(value) and float(value).is_integer()


def _is_valid_theta0(value: object, *, strictly_positive: bool) -> bool:
    """
    Whether a ``Theta_0`` cell is valid.

    A scalar ``Theta_0`` must be a positive (capacity/median) or non-negative
    (cost/time) float. A string ``Theta_0`` -- a multilinear-CDF spec or a
    quantity-/demand-dependent curve -- must decompose, on ``|`` and ``,``,
    into a non-empty list of floats.
    """
    if _is_float(value):
        number = float(value)
        return number > 0 if strictly_positive else number >= 0
    if isinstance(value, str):
        tokens = [t.strip() for t in re.split(r'[|,]', value) if t.strip()]
        return bool(tokens) and all(_is_float(t) for t in tokens)
    return False


def _is_normalized_weights(value: object) -> bool:
    """Whether ``DamageStateWeights`` is ``|``-separated floats summing to 1."""
    tokens = [t.strip() for t in str(value).split('|') if t.strip()]
    return (
        bool(tokens)
        and all(_is_float(t) for t in tokens)
        and abs(sum(float(t) for t in tokens) - 1.0) <= 0.02
    )


def _numbered_groups(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return the ``<prefix><n>`` group names present, in column order."""
    seen = dict.fromkeys(top for top, _ in frame.columns)
    return [top for top in seen if re.fullmatch(rf'{prefix}\d+', top)]


def _cells(frame: pd.DataFrame, subcolumn: str):
    """
    Yield ``(index_key, group, value)`` for non-null cells of a sub-column.

    ``index_key`` is the frame's index entry -- the model ID for fragility and
    the ``(model, DV)`` tuple for the repair collections -- used only to locate
    a bad cell in a failure message.
    """
    for top, sub in frame.columns:
        if sub != subcolumn:
            continue
        for index_key, value in frame[(top, sub)].dropna().items():
            yield index_key, top, value


def _model_ids(frame: pd.DataFrame) -> set[str]:
    """Return the set of model IDs (the outermost index level)."""
    return set(frame.index.get_level_values(0))


# ---------------------------------------------------------------------------
# Discovery is non-vacuous
# ---------------------------------------------------------------------------


def test_discovery_finds_every_parameter_table():
    """The work lists match the known dataset/collection inventory."""
    assert len(_PAIRS) == _EXPECTED_PAIR_COUNT
    assert len(_METADATA_PAIRS) == _EXPECTED_METADATA_PAIR_COUNT
    assert len(_NESTED_PAIRS) == _EXPECTED_NESTED_PAIR_COUNT


# ---------------------------------------------------------------------------
# Tier A -- loadability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_parameters_load_into_a_tidy_table(dataset: str, collection: str):
    """Every collection's parameters load non-empty with MultiIndex columns
    and a unique, non-positional model index."""
    frame = dlml.get_parameters(dataset, collection)
    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert isinstance(frame.columns, pd.MultiIndex)
    assert frame.columns.nlevels == 2
    assert frame.index.is_unique
    assert not isinstance(frame.index, pd.RangeIndex)


# ---------------------------------------------------------------------------
# Tier B -- structure (closed-world columns + required floor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_columns_are_all_recognized(dataset: str, collection: str):
    """Every column is a known fixed group or a ``<prefix><n>`` group, with a
    sub-column valid for that group -- no stray or misspelled columns."""
    rule = _COLLECTION_RULES[collection]
    prefix = rule['numbered_prefix']
    frame = dlml.get_parameters(dataset, collection)
    for top, sub in frame.columns:
        if top in rule['fixed_columns']:
            assert (
                sub in rule['fixed_columns'][top]
            ), f'bad sub-column ({top!r}, {sub!r})'
        elif prefix is not None and re.fullmatch(rf'{prefix}\d+', top):
            assert (
                sub in rule['numbered_subcolumns']
            ), f'bad sub-column ({top!r}, {sub!r})'
        else:
            pytest.fail(f'unrecognized column group {top!r}')


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_required_groups_and_index_shape(dataset: str, collection: str):
    """The required groups/sub-columns and the index shape are present. At
    least one numbered group must exist, but not a specific one (a model may
    define only ``LS2``, leaving ``LS1`` empty)."""
    rule = _COLLECTION_RULES[collection]
    frame = dlml.get_parameters(dataset, collection)
    assert tuple(frame.index.names) == rule['index_names']
    for group, required_subs in rule['required_subcolumns'].items():
        present = {sub for top, sub in frame.columns if top == group}
        assert present, f'missing required group {group!r}'
        assert (
            required_subs <= present
        ), f'{group} missing {sorted(required_subs - present)}'
    if rule['requires_numbered_group']:
        assert _numbered_groups(frame, rule['numbered_prefix']), 'no numbered group'


# ---------------------------------------------------------------------------
# Tier B -- value domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_flag_columns_are_binary(dataset: str, collection: str):
    """``Directional``, ``LongLeadTime`` and ``Incomplete`` carry only 0 or 1."""
    frame = dlml.get_parameters(dataset, collection)
    # The Incomplete flag is the single column ('Incomplete', ''), so it is
    # reached via the empty sub-column; the guard keeps any other empty-sub
    # column (none today) from being treated as a flag.
    bad = [
        (index_key, group, value)
        for subcolumn in ('Directional', 'LongLeadTime', '')
        for index_key, group, value in _cells(frame, subcolumn)
        if (subcolumn != '' or group == 'Incomplete') and not _is_binary(value)
    ]
    assert not bad, f'non-binary flag values: {bad[:3]}'


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_offsets_are_whole_numbers(dataset: str, collection: str):
    """``Demand``/``Offset`` is an integer-valued story offset."""
    frame = dlml.get_parameters(dataset, collection)
    bad = [c for c in _cells(frame, 'Offset') if not _is_whole_number(c[2])]
    assert not bad, f'non-integer offsets: {bad[:3]}'


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_dispersion_is_positive(dataset: str, collection: str):
    """Every ``Theta_1`` dispersion is a strictly positive float."""
    frame = dlml.get_parameters(dataset, collection)
    bad = [
        c
        for c in _cells(frame, 'Theta_1')
        if not (_is_float(c[2]) and float(c[2]) > 0)
    ]
    assert not bad, f'non-positive dispersion: {bad[:3]}'


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_theta0_values_are_well_formed(dataset: str, collection: str):
    """Every ``Theta_0`` is a valid scalar or a valid numeric-string curve."""
    positive = _COLLECTION_RULES[collection]['theta0_strictly_positive']
    frame = dlml.get_parameters(dataset, collection)
    bad = [
        c
        for c in _cells(frame, 'Theta_0')
        if not _is_valid_theta0(c[2], strictly_positive=positive)
    ]
    assert not bad, f'malformed Theta_0: {bad[:3]}'


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_damage_state_weights_are_normalized(dataset: str, collection: str):
    """Every ``DamageStateWeights`` is ``|``-separated floats summing to ~1."""
    frame = dlml.get_parameters(dataset, collection)
    bad = [
        c
        for c in _cells(frame, 'DamageStateWeights')
        if not _is_normalized_weights(c[2])
    ]
    assert not bad, f'malformed DamageStateWeights: {bad[:3]}'


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_numbered_groups_complete_unless_incomplete(dataset: str, collection: str):
    """A ``LS<n>``/``DS<n>`` group that names a ``Family`` must supply that
    family's parameters (``Theta_0``, plus ``Theta_1`` for a two-parameter
    family) -- unless the model is flagged ``Incomplete``."""
    rule = _COLLECTION_RULES[collection]
    prefix = rule['numbered_prefix']
    if prefix is None:
        return
    frame = dlml.get_parameters(dataset, collection)
    if ('Incomplete', '') in frame.columns:
        incomplete = (
            frame[('Incomplete', '')]
            .map(lambda v: _is_binary(v) and float(v) == 1.0)
            .astype(bool)
        )
    else:
        incomplete = pd.Series(data=False, index=frame.index)

    def column_or_na(group: str, sub: str) -> pd.Series:
        if (group, sub) in frame.columns:
            return frame[(group, sub)]
        return pd.Series(data=pd.NA, index=frame.index)

    violations: list[tuple] = []
    for group in _numbered_groups(frame, prefix):
        family = column_or_na(group, 'Family')
        active = family.notna() & ~incomplete
        missing_theta0 = active & column_or_na(group, 'Theta_0').isna()
        needs_theta1 = active & family.isin(_TWO_PARAMETER_FAMILIES)
        missing_theta1 = needs_theta1 & column_or_na(group, 'Theta_1').isna()
        violations += [
            (mid, group, 'Theta_0') for mid in frame.index[missing_theta0]
        ]
        violations += [
            (mid, group, 'Theta_1') for mid in frame.index[missing_theta1]
        ]
    assert not violations, f'under-specified complete models: {violations[:3]}'


# ---------------------------------------------------------------------------
# Tier C -- metadata schema
# ---------------------------------------------------------------------------

# A damage/repair-state entry inside LimitStates / DamageStates.
_REPAIR_STATE = {
    'type': 'object',
    'required': ['Description'],
    'additionalProperties': False,
    'properties': {
        'Description': {'type': 'string'},
        'RepairAction': {'type': 'string'},
    },
}
# fragility: LimitStates -> LS<n> -> DS<n> -> repair state.
_LIMIT_STATES = {
    'type': 'object',
    'additionalProperties': False,
    'patternProperties': {
        r'^LS\d+$': {
            'type': 'object',
            'additionalProperties': False,
            'patternProperties': {r'^DS\d+$': _REPAIR_STATE},
        }
    },
}
# consequence: DamageStates -> DS<n> -> repair state.
_DAMAGE_STATES = {
    'type': 'object',
    'additionalProperties': False,
    'patternProperties': {r'^DS\d+$': _REPAIR_STATE},
}
_GENERAL_INFORMATION = {
    'type': 'object',
    'required': ['ShortName', 'Description', 'Version', 'ComponentGroups'],
    'additionalProperties': False,
    'properties': {
        'ShortName': {'type': 'string'},
        'Description': {'type': 'string'},
        'Version': {'type': 'string'},
        'ComponentGroups': {'type': ['object', 'array']},
        'DecisionVariables': {'type': 'object'},
    },
}
_REFERENCES = {'type': 'object', 'additionalProperties': {'type': 'string'}}

# Top-level metadata keys that are not models.
_RESERVED_KEYS = frozenset({'_GeneralInformation', 'References'})

# Fields every model carries, with the closed-world property set extended per
# collection (RoundUpToIntegerQuantity is a boolean flag stored canonically as
# the string 'True' or 'False').
_COMMON_REQUIRED = [
    'Description',
    'SuggestedComponentBlockSize',
    'RoundUpToIntegerQuantity',
]
_COMMON_PROPERTIES = {
    'Description': {'type': 'string'},
    'SuggestedComponentBlockSize': {'type': 'string'},
    'RoundUpToIntegerQuantity': {'enum': ['True', 'False']},
}


def _model_schema(required: list, properties: dict) -> dict:
    """A closed-world model schema: common fields plus per-collection extras."""
    return {
        'type': 'object',
        'additionalProperties': False,
        'required': [*_COMMON_REQUIRED, *required],
        'properties': {**_COMMON_PROPERTIES, **properties},
    }


def _file_schema(model_schema: dict) -> dict:
    """A whole metadata file: ``_GeneralInformation`` (required), optional
    ``References``, and every other key a model."""
    return {
        'type': 'object',
        'required': ['_GeneralInformation'],
        'properties': {
            '_GeneralInformation': _GENERAL_INFORMATION,
            'References': _REFERENCES,
        },
        'additionalProperties': model_schema,
    }


_METADATA_SCHEMAS = {
    'fragility': _file_schema(
        _model_schema(
            ['LimitStates'],
            {
                'Comments': {'type': 'string'},
                'Reference': {'type': 'array', 'items': {'type': 'string'}},
                'LimitStates': _LIMIT_STATES,
            },
        )
    ),
    'consequence_repair': _file_schema(
        _model_schema(
            ['DamageStates'],
            {
                'Comments': {'type': 'string'},
                'ControllingDemand': {'type': 'string'},
                'DamageStates': _DAMAGE_STATES,
            },
        )
    ),
    'loss_repair': _file_schema(_model_schema([], {})),
}


@pytest.mark.parametrize(
    ('dataset', 'collection'), _METADATA_PAIRS, ids=_METADATA_PAIR_IDS
)
def test_metadata_matches_its_schema(dataset: str, collection: str):
    """The whole metadata file conforms to its collection's schema: a valid
    ``_GeneralInformation``, an optional ``References`` bibliography, and every
    model entry carrying exactly its required and optional fields -- including
    the nested ``LimitStates`` / ``DamageStates`` structure."""
    metadata = dlml.get_metadata(dataset, collection)
    validator = jsonschema.Draft7Validator(_METADATA_SCHEMAS[collection])
    errors = sorted(
        f'{error.json_path}: {error.message}'
        for error in validator.iter_errors(metadata)
    )
    assert not errors, f'{len(errors)} schema violation(s), e.g. {errors[:3]}'


@pytest.mark.parametrize(
    ('dataset', 'collection'), _METADATA_PAIRS, ids=_METADATA_PAIR_IDS
)
def test_metadata_keys_are_exactly_the_models(dataset: str, collection: str):
    """Top-level keys are precisely the model IDs plus ``_GeneralInformation``
    and the optional ``References`` -- every model documented, nothing stray."""
    metadata = dlml.get_metadata(dataset, collection)
    documented = set(metadata) - _RESERVED_KEYS
    model_ids = _model_ids(dlml.get_parameters(dataset, collection))
    assert documented == model_ids, (
        f'extra={sorted(documented - model_ids)[:3]} '
        f'missing={sorted(model_ids - documented)[:3]}'
    )


@pytest.mark.parametrize(
    ('dataset', 'collection'), _METADATA_PAIRS, ids=_METADATA_PAIR_IDS
)
def test_model_citations_resolve(dataset: str, collection: str):
    """Every per-model ``Reference`` citation key resolves to an entry in the
    file's ``References`` bibliography."""
    metadata = dlml.get_metadata(dataset, collection)
    bibliography = set(metadata.get('References', {}))
    dangling = [
        (model_id, citation)
        for model_id, entry in metadata.items()
        if model_id not in _RESERVED_KEYS and isinstance(entry, dict)
        for citation in entry.get('Reference', [])
        if citation not in bibliography
    ]
    assert not dangling, f'dangling citations: {dangling[:3]}'


def _states_in_parameters(frame: pd.DataFrame, prefix: str) -> dict[str, set[str]]:
    """Map each model to the ``<prefix><n>`` groups it populates in the CSV."""
    states: dict[str, set[str]] = {}
    models = frame.index.get_level_values(0)
    for top, sub in frame.columns:
        if re.fullmatch(rf'{prefix}\d+', top):
            for model in models[frame[(top, sub)].notna().to_numpy()]:
                states.setdefault(model, set()).add(top)
    return states


@pytest.mark.parametrize(
    ('dataset', 'collection'), _NESTED_PAIRS, ids=_NESTED_PAIR_IDS
)
def test_metadata_states_match_parameters(dataset: str, collection: str):
    """Each model's documented limit/damage states (minus the ``LS0``/``DS0``
    undamaged baseline the CSV omits) are exactly the ``LS<n>``/``DS<n>``
    groups it defines in the parameters table."""
    nested_key, prefix = _NESTED_STATES[collection]
    baseline = f'{prefix}0'
    metadata = dlml.get_metadata(dataset, collection)
    in_parameters = _states_in_parameters(
        dlml.get_parameters(dataset, collection), prefix
    )
    mismatches = []
    for model_id, entry in metadata.items():
        if model_id in _RESERVED_KEYS or not isinstance(entry, dict):
            continue
        documented = set(entry.get(nested_key, {})) - {baseline}
        defined = in_parameters.get(model_id, set())
        if documented != defined:
            mismatches.append((model_id, sorted(documented), sorted(defined)))
    assert not mismatches, f'metadata vs CSV {prefix} mismatch: {mismatches[:3]}'
