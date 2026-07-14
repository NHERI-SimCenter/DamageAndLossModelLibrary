"""
Public API of the Damage and Loss Model Library.

This module is the supported entry point for reading packaged datasets. It
exposes dataset and collection discovery, parsed model parameters and
metadata, input schemas, an escape hatch for extra files, and asset
validation against a dataset's input schema.

Parsed objects (DataFrames, dicts, validators) are cached so repeated calls
are cheap. The public accessors return defensive copies, so callers may
freely mutate what they receive without corrupting the shared cache.

The module is pelicun-free: importing it never imports pelicun.
"""

from __future__ import annotations

import copy
import functools
import json
from typing import TYPE_CHECKING

import jsonschema

from dlml import _catalog
from dlml._tabular import convert_to_MultiIndex, load_tabular

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

__all__ = [
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


# ---------------------------------------------------------------------------
# Dataset and collection discovery
# ---------------------------------------------------------------------------


def list_datasets() -> list[str]:
    """
    List the IDs of every packaged dataset.

    Returns
    -------
    list of str
        The dataset IDs (folder paths relative to ``data/``), sorted.

    """
    return _catalog.dataset_ids()


def available_collections(dataset_id: str) -> list[str]:
    """
    List the collections a dataset provides parameters for.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    list of str
        The sorted collections whose parameters CSV exists, a subset of
        ``('fragility', 'consequence_repair', 'loss_repair')``.

    """
    return _catalog.available_collections(dataset_id)


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


@functools.cache
def _load_parameters(dataset_id: str, collection: str) -> pd.DataFrame:
    """Resolve and parse a collection's parameters CSV (cached)."""
    return load_tabular(_catalog.parameters_path(dataset_id, collection))


def get_parameters(dataset_id: str, collection: str) -> pd.DataFrame:
    """
    Return the parsed model parameters for a collection.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    collection: str
        One of ``'fragility'``, ``'consequence_repair'`` or
        ``'loss_repair'``.

    Returns
    -------
    pandas.DataFrame
        The parsed parameters, with MultiIndex columns and a
        component-ID index. A defensive copy is returned on each call.

    """
    return _load_parameters(dataset_id, collection).copy()


def get_fragility(dataset_id: str) -> pd.DataFrame:
    """
    Return the fragility model parameters for a dataset.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    pandas.DataFrame
        The fragility parameters. A defensive copy is returned on each
        call.

    """
    return get_parameters(dataset_id, 'fragility')


def get_consequence_repair(dataset_id: str) -> pd.DataFrame:
    """
    Return the consequence-repair model parameters for a dataset.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    pandas.DataFrame
        The consequence-repair parameters. A defensive copy is returned
        on each call.

    """
    return get_parameters(dataset_id, 'consequence_repair')


def get_loss_repair(dataset_id: str) -> pd.DataFrame:
    """
    Return the loss-repair model parameters for a dataset.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    pandas.DataFrame
        The loss-repair parameters. A defensive copy is returned on each
        call.

    """
    return get_parameters(dataset_id, 'loss_repair')


# ---------------------------------------------------------------------------
# Metadata and schema
# ---------------------------------------------------------------------------


@functools.cache
def _load_metadata(dataset_id: str, collection: str) -> dict:
    """Resolve and parse a collection's metadata JSON (cached)."""
    path = _catalog.metadata_path(dataset_id, collection)
    return json.loads(path.read_text(encoding='utf-8'))


def get_metadata(dataset_id: str, collection: str) -> dict:
    """
    Return the metadata for a collection.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    collection: str
        One of ``'fragility'``, ``'consequence_repair'`` or
        ``'loss_repair'``.

    Returns
    -------
    dict
        The parsed metadata. A defensive deep copy is returned on each
        call.

    """
    return copy.deepcopy(_load_metadata(dataset_id, collection))


@functools.cache
def _load_schema(dataset_id: str) -> dict:
    """Resolve and parse a dataset's input schema (cached)."""
    path = _catalog.schema_path(dataset_id)
    return json.loads(path.read_text(encoding='utf-8'))


def get_schema(dataset_id: str) -> dict:
    """
    Return the input-validation schema for a dataset.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    dict
        The parsed JSON schema. A defensive deep copy is returned on each
        call.

    """
    return copy.deepcopy(_load_schema(dataset_id))


# ---------------------------------------------------------------------------
# Extra files
# ---------------------------------------------------------------------------


def get_file(dataset_id: str, filename: str) -> Path:
    """
    Resolve any file in a dataset's folder by its exact name.

    This is the escape hatch for extra files (e.g.
    ``combine_wind_flood.csv``) that fall outside the naming convention.
    The file is not read; a path is returned.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    filename: str
        The exact name of a file in the dataset's folder.

    Returns
    -------
    Path
        The path to the file.

    """
    return _catalog.file_path(dataset_id, filename)


# ---------------------------------------------------------------------------
# Asset validation
# ---------------------------------------------------------------------------


@functools.cache
def _load_validator(dataset_id: str) -> jsonschema.Draft7Validator:
    """Build a Draft7 validator from a dataset's cached schema (cached)."""
    return jsonschema.Draft7Validator(_load_schema(dataset_id))


def _format_errors(
    validator: jsonschema.Draft7Validator, features: dict
) -> list[str]:
    """Return sorted ``'<json_path>: <message>'`` strings for all violations."""
    return sorted(
        f'{error.json_path}: {error.message}'
        for error in validator.iter_errors(features)
    )


def validate_asset(dataset_id: str, features: dict) -> list[str]:
    """
    Validate a single asset's features against a dataset's input schema.

    Every schema violation is reported, not just the first.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    features: dict
        The asset's feature values, keyed by feature name.

    Returns
    -------
    list of str
        Sorted human-readable messages, one per violation, each formatted
        as ``'<json_path>: <message>'``. An empty list means the asset is
        valid.

    """
    return _format_errors(_load_validator(dataset_id), features)


def validate_assets(dataset_id: str, assets: dict) -> dict:
    """
    Validate many assets against a dataset's input schema.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    assets: dict
        A mapping of asset ID to that asset's feature dict.

    Returns
    -------
    dict
        A report of failures: a mapping of asset ID to its sorted list of
        violation messages, including only assets that have at least one
        violation. An empty dict means every asset is valid.

    """
    validator = _load_validator(dataset_id)
    report: dict = {}
    for asset_id, features in assets.items():
        errors = _format_errors(validator, features)
        if errors:
            report[asset_id] = errors
    return report
