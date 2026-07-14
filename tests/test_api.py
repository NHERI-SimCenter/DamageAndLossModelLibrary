"""Standalone unit tests for :mod:`dlml.api`, the public API.

These tests are pelicun-free and do not access the network. They exercise
the public accessors against the real packaged data tree, the defensive-copy
contract that protects the internal caches, and asset validation against a
dataset's input schema.

Datasets are used as fixtures for the structural property each one
exercises (noted at the constants below). A dataset can be swapped (value +
comment) without renaming a test.
"""

import pandas as pd
import pytest

from dlml import (
    DatasetFileNotFoundError,
    available_collections,
    convert_to_MultiIndex,
    get_consequence_repair,
    get_file,
    get_fragility,
    get_loss_repair,
    get_metadata,
    get_parameters,
    get_schema,
    list_datasets,
    validate_asset,
    validate_assets,
)

EXPECTED_DATASET_COUNT = 11
EXPECTED_DATASET_IDS = [
    'flood/building/portfolio/Hazus v6.1',
    'hurricane/building/component/SimCenter Wind Component Library',
    'hurricane/building/portfolio/Hazus v5.1 coupled',
    'hurricane/building/portfolio/Hazus v5.1 original',
    'seismic/building/component/FEMA P-58 2nd Edition',
    'seismic/building/portfolio/Hazus v5.1',
    'seismic/building/portfolio/Hazus v6.1',
    'seismic/building/subassembly/Hazus v5.1',
    'seismic/power_network/portfolio/Hazus v5.1',
    'seismic/transportation_network/portfolio/Hazus v5.1',
    'seismic/water_network/portfolio/Hazus v6.1',
]

# Dataset fixtures, each chosen for the structural property noted.

# Has fragility + consequence_repair but no loss_repair; no schema.
DATASET_TWO_COLLECTIONS = 'seismic/building/component/FEMA P-58 2nd Edition'
# Has fragility parameters AND metadata, plus an input schema.
DATASET_WITH_SCHEMA = 'seismic/building/portfolio/Hazus v6.1'
# Provides loss_repair parameters but no loss_repair metadata JSON.
DATASET_LOSS_NO_METADATA = 'flood/building/portfolio/Hazus v6.1'
# Carries the extra (non-convention) file combine_wind_flood.csv.
DATASET_WITH_EXTRA_FILE = 'hurricane/building/portfolio/Hazus v5.1 coupled'
EXTRA_FILE = 'combine_wind_flood.csv'


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_datasets_returns_expected_ids():
    assert list_datasets() == EXPECTED_DATASET_IDS
    assert len(list_datasets()) == EXPECTED_DATASET_COUNT


def test_available_collections_reexport_matches_catalog():
    assert available_collections(DATASET_TWO_COLLECTIONS) == [
        'consequence_repair',
        'fragility',
    ]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('accessor', 'dataset'),
    [
        (get_fragility, DATASET_TWO_COLLECTIONS),
        (get_consequence_repair, DATASET_TWO_COLLECTIONS),
        (get_loss_repair, DATASET_LOSS_NO_METADATA),
    ],
    ids=['fragility', 'consequence_repair', 'loss_repair'],
)
def test_collection_accessor_returns_tidy_dataframe(accessor, dataset):
    """Each collection accessor returns a MultiIndex-column, ID-indexed frame."""
    frame = accessor(dataset)
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.columns, pd.MultiIndex)
    assert not isinstance(frame.index, pd.RangeIndex)
    assert not frame.empty


def test_get_parameters_matches_named_accessor():
    """get_parameters and the named accessor return identical data."""
    via_generic = get_parameters(DATASET_TWO_COLLECTIONS, 'fragility')
    via_named = get_fragility(DATASET_TWO_COLLECTIONS)
    pd.testing.assert_frame_equal(via_generic, via_named)


def test_get_loss_repair_raises_when_collection_absent():
    """A dataset lacking loss_repair raises rather than returning empty data."""
    with pytest.raises(DatasetFileNotFoundError, match='no loss_repair parameters'):
        get_loss_repair(DATASET_TWO_COLLECTIONS)


# ---------------------------------------------------------------------------
# Defensive-copy contract (cache integrity)
# ---------------------------------------------------------------------------


def test_get_fragility_returns_independent_copies():
    """Mutating a returned frame must not leak into the next call's result."""
    first = get_fragility(DATASET_TWO_COLLECTIONS)
    expected_shape = first.shape
    first.iloc[:, :] = 0
    first['injected_column'] = 1
    second = get_fragility(DATASET_TWO_COLLECTIONS)
    assert second.shape == expected_shape
    assert 'injected_column' not in second.columns
    assert not (second == 0).all().all()


def test_get_metadata_returns_independent_copies():
    """Mutating returned metadata must not corrupt the cache."""
    first = get_metadata(DATASET_WITH_SCHEMA, 'fragility')
    first['__injected__'] = 'tampered'
    second = get_metadata(DATASET_WITH_SCHEMA, 'fragility')
    assert '__injected__' not in second


def test_get_schema_returns_independent_copies():
    """Mutating a returned schema must not corrupt the cache."""
    first = get_schema(DATASET_WITH_SCHEMA)
    first['properties'].clear()
    first['__injected__'] = True
    second = get_schema(DATASET_WITH_SCHEMA)
    assert '__injected__' not in second
    assert second['properties']


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_get_metadata_returns_dict_when_present():
    metadata = get_metadata(DATASET_WITH_SCHEMA, 'fragility')
    assert isinstance(metadata, dict)
    assert metadata


def test_get_metadata_raises_when_absent():
    """loss_repair here has parameters but no metadata JSON."""
    with pytest.raises(DatasetFileNotFoundError, match='no loss_repair metadata'):
        get_metadata(DATASET_LOSS_NO_METADATA, 'loss_repair')


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_get_schema_returns_dict_with_properties():
    schema = get_schema(DATASET_WITH_SCHEMA)
    assert isinstance(schema, dict)
    assert 'properties' in schema


def test_get_schema_raises_when_absent():
    with pytest.raises(DatasetFileNotFoundError, match='no input schema'):
        get_schema(DATASET_TWO_COLLECTIONS)


# ---------------------------------------------------------------------------
# Extra files
# ---------------------------------------------------------------------------


def test_get_file_resolves_existing_extra_file():
    path = get_file(DATASET_WITH_EXTRA_FILE, EXTRA_FILE)
    assert path.is_file()
    assert path.name == EXTRA_FILE


def test_get_file_raises_for_bogus_filename():
    with pytest.raises(DatasetFileNotFoundError):
        get_file(DATASET_WITH_EXTRA_FILE, 'does_not_exist.csv')


# ---------------------------------------------------------------------------
# Single-asset validation
# ---------------------------------------------------------------------------


def test_validate_asset_accepts_conforming_features():
    """A fully conforming asset produces no messages."""
    features = {
        'StructureType': 'W1',
        'DesignLevel': 'High-Code',
        'FoundationType': 'Shallow',
    }
    assert validate_asset(DATASET_WITH_SCHEMA, features) == []


def test_validate_asset_reports_missing_required_properties():
    """An empty asset is missing required properties; messages are sorted."""
    errors = validate_asset(DATASET_WITH_SCHEMA, {})
    assert errors
    assert errors == sorted(errors)
    assert any('StructureType' in message for message in errors)


def test_validate_asset_reports_every_violation():
    """Two independent problems yield at least two distinct messages."""
    features = {'StructureType': 'NOT_A_TYPE', 'DesignLevel': 'NOT_A_LEVEL'}
    errors = validate_asset(DATASET_WITH_SCHEMA, features)
    assert len(errors) >= 2
    assert any('StructureType' in message for message in errors)
    assert any('DesignLevel' in message for message in errors)


def test_validate_asset_raises_without_schema():
    with pytest.raises(DatasetFileNotFoundError, match='no input schema'):
        validate_asset(DATASET_TWO_COLLECTIONS, {'StructureType': 'W1'})


# ---------------------------------------------------------------------------
# Multi-asset validation
# ---------------------------------------------------------------------------


def test_validate_assets_reports_only_failures():
    """Only failing assets appear in the report, keyed by asset ID."""
    assets = {
        'good': {
            'StructureType': 'W1',
            'DesignLevel': 'High-Code',
            'FoundationType': 'Shallow',
        },
        'bad_enum': {
            'StructureType': 'NOPE',
            'DesignLevel': 'High-Code',
            'FoundationType': 'Shallow',
        },
        'missing': {},
    }
    report = validate_assets(DATASET_WITH_SCHEMA, assets)
    assert set(report) == {'bad_enum', 'missing'}
    assert all(report[asset_id] for asset_id in report)


def test_validate_assets_empty_when_all_pass():
    assets = {
        'a': {
            'StructureType': 'W1',
            'DesignLevel': 'High-Code',
            'FoundationType': 'Shallow',
        },
    }
    assert validate_assets(DATASET_WITH_SCHEMA, assets) == {}


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_convert_to_multiindex_is_importable_and_splits_index():
    frame = pd.DataFrame({'value': [1, 2]}, index=['A-1', 'B-2'])
    converted = convert_to_MultiIndex(frame, axis=0)
    assert isinstance(converted.index, pd.MultiIndex)
    assert converted.index.nlevels == 2
    assert list(converted.index) == [('A', '1'), ('B', '2')]
