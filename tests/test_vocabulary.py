"""Tests for the canonical controlled vocabularies in :mod:`dlml.vocabulary`.

Beyond the structural invariants of the enumerations, these tests prove the
vocabularies actually cover every demand type and distribution family used in
the packaged model parameters -- so the library's data can never reference a
value the vocabulary does not define.
"""

from __future__ import annotations

import dlml
from dlml import vocabulary

_PAIRS = [
    (dataset, collection)
    for dataset in dlml.list_datasets()
    for collection in dlml.available_collections(dataset)
]


def _demand_type_base(value: str) -> str:
    """Strip a spectral ``|<period>`` qualifier from a demand-type value."""
    return value.split('|', 1)[0]


def _values_under(subcolumn: str) -> set[str]:
    """Collect the distinct non-null values across every ``*/subcolumn`` column."""
    found: set[str] = set()
    for dataset, collection in _PAIRS:
        frame = dlml.get_parameters(dataset, collection)
        for top, sub in frame.columns:
            if sub == subcolumn:
                found.update(frame[(top, sub)].dropna().astype(str))
    return found


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_demand_types_are_the_mapping_keys():
    """DEMAND_TYPES is exactly the set of demand-type names in the mapping."""
    assert frozenset(vocabulary.EDP_to_demand_type) == vocabulary.DEMAND_TYPES
    assert vocabulary.DEMAND_TYPES
    # Fidelity tripwire: an accidental edit to the pelicun-mirrored mapping
    # must be conscious. Update this count deliberately when pelicun changes.
    assert len(vocabulary.EDP_to_demand_type) == 31


def test_demand_type_codes_are_nonempty_strings():
    """Every demand-type name maps to a non-empty short code."""
    assert all(
        isinstance(code, str) and code
        for code in vocabulary.EDP_to_demand_type.values()
    )


def test_unit_types_are_the_expected_set():
    """UNIT_TYPES carries exactly the five valid unit types."""
    assert (
        frozenset(
            {
                'acceleration',
                'speed',
                'displacement',
                'unitless',
                'rotation',
                'force',
                'force_per_length',
                'pressure',
            }
        )
        == vocabulary.UNIT_TYPES
    )


def test_edp_types_entries_are_well_formed():
    """Every EDP_TYPES entry has exactly an Acronym and a valid UnitType."""
    for name, info in vocabulary.EDP_TYPES.items():
        assert set(info) == {'Acronym', 'UnitType'}, name
        assert isinstance(info['Acronym'], str)
        assert info['Acronym']
        assert info['UnitType'] in vocabulary.UNIT_TYPES, name


def test_demand_type_mapping_is_derived_from_edp_types():
    """EDP_to_demand_type is the name-to-acronym projection of EDP_TYPES."""
    assert vocabulary.EDP_to_demand_type == {
        name: info['Acronym'] for name, info in vocabulary.EDP_TYPES.items()
    }
    assert list(vocabulary.EDP_to_demand_type) == list(vocabulary.EDP_TYPES)


def test_names_sharing_an_acronym_share_a_unit_type():
    """Two names with the same acronym agree on the unit type, so a consumer
    may safely look up units at the acronym level."""
    by_acronym: dict[str, str] = {}
    for name, info in vocabulary.EDP_TYPES.items():
        expected = by_acronym.setdefault(info['Acronym'], info['UnitType'])
        assert info['UnitType'] == expected, name


def test_distribution_families_include_the_core_set():
    """The family set carries pelicun's core distributions, incl. the
    ``deterministic`` default used for an absent family."""
    core = {'normal', 'lognormal', 'uniform', 'weibull', 'deterministic'}
    assert core <= vocabulary.DISTRIBUTION_FAMILIES
    assert all(isinstance(name, str) for name in vocabulary.DISTRIBUTION_FAMILIES)
    # Fidelity tripwire against pelicun's rv_class_map registry (11 families).
    assert len(vocabulary.DISTRIBUTION_FAMILIES) == 11


def test_vocabulary_is_reexported_from_package():
    """The canonical names are reachable directly from the package root."""
    assert dlml.DEMAND_TYPES is vocabulary.DEMAND_TYPES
    assert dlml.DISTRIBUTION_FAMILIES is vocabulary.DISTRIBUTION_FAMILIES
    assert dlml.EDP_TYPES is vocabulary.EDP_TYPES
    assert dlml.EDP_to_demand_type is vocabulary.EDP_to_demand_type
    assert dlml.UNIT_TYPES is vocabulary.UNIT_TYPES


# ---------------------------------------------------------------------------
# The vocabulary covers the packaged data
# ---------------------------------------------------------------------------


def test_every_demand_type_in_the_data_is_known():
    """Every ``Demand``/``Type`` value (sans spectral qualifier) is defined."""
    used = {_demand_type_base(value) for value in _values_under('Type')}
    unknown = used - vocabulary.DEMAND_TYPES
    assert not unknown, f'undefined demand types in data: {sorted(unknown)}'


def test_every_distribution_family_in_the_data_is_known():
    """Every non-null ``Family`` value across the data is a defined family."""
    used = _values_under('Family')
    unknown = used - vocabulary.DISTRIBUTION_FAMILIES
    assert not unknown, f'undefined families in data: {sorted(unknown)}'
