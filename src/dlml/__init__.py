"""Damage and Loss Model Library (dlml)."""

from importlib import metadata

from dlml._catalog import DatasetFileNotFoundError, UnknownDatasetError
from dlml.api import (
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
from dlml.vocabulary import (
    DEMAND_TYPES,
    DISTRIBUTION_FAMILIES,
    EDP_to_demand_type,
)

try:
    __version__ = metadata.version('dlml')
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0+local'

__all__ = [
    'DEMAND_TYPES',
    'DISTRIBUTION_FAMILIES',
    'DatasetFileNotFoundError',
    'EDP_to_demand_type',
    'UnknownDatasetError',
    '__version__',
    'available_collections',
    'convert_to_MultiIndex',
    'get_consequence_repair',
    'get_file',
    'get_fragility',
    'get_loss_repair',
    'get_metadata',
    'get_parameters',
    'get_schema',
    'list_datasets',
    'validate_asset',
    'validate_assets',
]
