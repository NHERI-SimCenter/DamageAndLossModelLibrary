"""Standalone unit tests for :mod:`dlml._catalog`.

These tests are pelicun-free and do not access the network. They cover
convention-based dataset discovery against the real library data tree and
the public resolver functions, including their error paths.

Datasets are used as fixtures for the structural property each one
exercises (noted at the constants below). A dataset can be swapped (value
+ comment) without renaming a test.
"""

import pytest

from dlml._catalog import (
    DatasetFileNotFoundError,
    UnknownDatasetError,
    available_collections,
    config_script_path,
    data_root,
    dataset_ids,
    file_path,
    metadata_path,
    parameters_path,
    schema_path,
)

# Invariants of the packaged data tree.
EXPECTED_DATASET_COUNT = 11
EXPECTED_FILE_COUNT = 46

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

# Dataset fixtures, each chosen for the structural property noted. To use a
# different dataset, change the value and the comment.

# Two collections (fragility + consequence_repair); no schema or config script.
DATASET_TWO_COLLECTIONS = 'seismic/building/component/FEMA P-58 2nd Edition'
# Two collections plus an input schema and a config script.
DATASET_WITH_SCHEMA_AND_CONFIG = 'seismic/building/portfolio/Hazus v6.1'
# Carries the extra (non-convention) file combine_wind_flood.csv.
DATASET_WITH_EXTRA_FILE = 'hurricane/building/portfolio/Hazus v5.1 coupled'
# Carries the extra (non-convention) file FloodRulesets.py.
DATASET_WITH_EXTRA_SCRIPT = 'flood/building/portfolio/Hazus v6.1'
# Not present in the data tree.
DATASET_UNKNOWN = 'not/a/real/dataset'


def _count_disk_files():
    """Independently count every file under the data tree."""
    count = 0
    stack = [data_root()]
    while stack:
        node = stack.pop()
        for child in node.iterdir():
            if child.is_dir():
                stack.append(child)
            else:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Discovery and tree invariants
# ---------------------------------------------------------------------------


def test_dataset_ids_returns_expected_sorted_list():
    assert dataset_ids() == EXPECTED_DATASET_IDS


def test_total_dataset_count_invariant():
    assert len(dataset_ids()) == EXPECTED_DATASET_COUNT


def test_total_file_count_invariant():
    # Independent walk guards against accidental data add/remove.
    assert _count_disk_files() == EXPECTED_FILE_COUNT


def test_dataset_ids_are_well_formed():
    root = data_root()
    for dataset_id in dataset_ids():
        assert dataset_id
        assert not dataset_id.startswith('/')
        assert not dataset_id.endswith('/')
        node = root
        for part in dataset_id.split('/'):
            node = node / part
        assert node.is_dir()


def test_data_root_points_to_data_dir():
    assert data_root().is_dir()
    assert (data_root() / 'seismic').is_dir()


# ---------------------------------------------------------------------------
# available_collections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('dataset', 'expected'),
    [
        # two collections present
        (
            'seismic/building/component/FEMA P-58 2nd Edition',
            ['consequence_repair', 'fragility'],
        ),
        # a single fragility collection
        ('seismic/water_network/portfolio/Hazus v6.1', ['fragility']),
        # a single loss_repair collection
        ('flood/building/portfolio/Hazus v6.1', ['loss_repair']),
    ],
)
def test_available_collections_reflect_present_parameter_csvs(dataset, expected):
    """A dataset reports exactly the collections whose parameters CSV exists."""
    assert available_collections(dataset) == expected


# ---------------------------------------------------------------------------
# parameters_path / metadata_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('resolver', 'expected_name'),
    [(parameters_path, 'fragility.csv'), (metadata_path, 'fragility.json')],
    ids=['parameters', 'metadata'],
)
def test_collection_resolver_returns_conventional_file(resolver, expected_name):
    """A collection resolves to its conventional CSV/JSON file when present."""
    path = resolver(DATASET_WITH_SCHEMA_AND_CONFIG, 'fragility')
    assert path.is_file()
    assert path.name == expected_name


@pytest.mark.parametrize(
    'resolver', [parameters_path, metadata_path], ids=['parameters', 'metadata']
)
def test_collection_resolver_rejects_unrecognized_collection(resolver):
    """An unrecognized collection name is rejected before any file lookup."""
    with pytest.raises(DatasetFileNotFoundError, match='not a known collection'):
        resolver(DATASET_WITH_SCHEMA_AND_CONFIG, 'nonsense')


def test_parameters_path_error_distinguishes_missing_from_available():
    """The error names the requested (missing) collection and lists present ones."""
    # DATASET_TWO_COLLECTIONS has fragility + consequence_repair but no loss_repair.
    with pytest.raises(DatasetFileNotFoundError) as excinfo:
        parameters_path(DATASET_TWO_COLLECTIONS, 'loss_repair')
    requested, _, available = str(excinfo.value).partition('available collections:')
    # The missing requested collection is named in the leading clause only...
    assert 'no loss_repair parameters' in requested
    assert 'loss_repair' not in available
    # ...and the collections the dataset DOES have are listed afterward.
    assert 'fragility' in available
    assert 'consequence_repair' in available


@pytest.mark.parametrize(
    ('dataset', 'collection'),
    [
        # loss_repair has parameters but no metadata JSON
        ('flood/building/portfolio/Hazus v6.1', 'loss_repair'),
        # fragility has parameters but no metadata JSON
        ('seismic/water_network/portfolio/Hazus v6.1', 'fragility'),
    ],
)
def test_metadata_path_raises_when_metadata_file_absent(dataset, collection):
    """Metadata can be missing even when the collection's parameters exist."""
    with pytest.raises(DatasetFileNotFoundError, match=f'no {collection} metadata'):
        metadata_path(dataset, collection)


# ---------------------------------------------------------------------------
# schema_path / config_script_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('resolver', 'expected_name'),
    [
        (schema_path, 'input_schema.json'),
        (config_script_path, 'pelicun_config.py'),
    ],
    ids=['schema', 'config_script'],
)
def test_optional_file_resolver_returns_conventional_file(resolver, expected_name):
    """The schema and config-script resolvers return their file when present."""
    path = resolver(DATASET_WITH_SCHEMA_AND_CONFIG)
    assert path.is_file()
    assert path.name == expected_name


@pytest.mark.parametrize(
    ('resolver', 'expected_message'),
    [
        (schema_path, 'no input schema'),
        (config_script_path, 'no config script'),
    ],
    ids=['schema', 'config_script'],
)
def test_optional_file_resolver_raises_when_file_absent(resolver, expected_message):
    """A dataset lacking the optional file gets a clear error."""
    # DATASET_TWO_COLLECTIONS has neither an input schema nor a config script.
    with pytest.raises(DatasetFileNotFoundError, match=expected_message):
        resolver(DATASET_TWO_COLLECTIONS)


# ---------------------------------------------------------------------------
# file_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('dataset', 'filename'),
    [
        # an extra (non-convention) Python file
        (DATASET_WITH_EXTRA_SCRIPT, 'FloodRulesets.py'),
        # an extra (non-convention) CSV file
        (DATASET_WITH_EXTRA_FILE, 'combine_wind_flood.csv'),
        # a convention file requested by its exact name
        (DATASET_WITH_SCHEMA_AND_CONFIG, 'fragility.csv'),
    ],
)
def test_file_path_resolves_any_existing_file(dataset, filename):
    """file_path resolves any existing file by exact name, extras included."""
    path = file_path(dataset, filename)
    assert path.is_file()
    assert path.name == filename


def test_file_path_lists_real_files_for_unknown_filename():
    """An unknown filename raises, and the error lists the dataset's real files."""
    with pytest.raises(DatasetFileNotFoundError) as excinfo:
        file_path(DATASET_WITH_SCHEMA_AND_CONFIG, 'does_not_exist.csv')
    message = str(excinfo.value)
    assert 'does_not_exist.csv' in message
    assert 'fragility.csv' in message  # a real file present in the dataset


# ---------------------------------------------------------------------------
# unknown-dataset handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'call',
    [
        lambda: available_collections(DATASET_UNKNOWN),
        lambda: parameters_path(DATASET_UNKNOWN, 'fragility'),
        lambda: metadata_path(DATASET_UNKNOWN, 'fragility'),
        lambda: schema_path(DATASET_UNKNOWN),
        lambda: config_script_path(DATASET_UNKNOWN),
        lambda: file_path(DATASET_UNKNOWN, 'fragility.csv'),
    ],
    ids=[
        'available_collections',
        'parameters_path',
        'metadata_path',
        'schema_path',
        'config_script_path',
        'file_path',
    ],
)
def test_resolvers_raise_for_unknown_dataset(call):
    """Every resolver validates the dataset ID before touching the filesystem."""
    with pytest.raises(UnknownDatasetError):
        call()


def test_unknown_dataset_error_suggests_close_match():
    # A near-miss of an existing ID should surface a suggestion.
    with pytest.raises(UnknownDatasetError) as excinfo:
        parameters_path('seismic/building/portfolio/Hazus v6.0', 'fragility')
    assert 'Hazus v6.1' in str(excinfo.value)


def test_unknown_dataset_error_omits_suggestion_when_none_is_close():
    with pytest.raises(UnknownDatasetError) as excinfo:
        parameters_path('completely/unrelated/identifier/xyz', 'fragility')
    message = str(excinfo.value)
    assert 'unknown dataset ID' in message
    assert 'Did you mean' not in message
