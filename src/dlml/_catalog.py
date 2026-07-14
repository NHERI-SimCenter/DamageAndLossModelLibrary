"""
Convention-based discovery and resolution of dlml datasets.

Datasets are discovered by walking the packaged ``data/`` tree, and their
files are resolved by an enforced naming convention: a file is found only
if it is named exactly as the convention dictates.

A **dataset** is a leaf folder under ``data/`` that directly contains at
least one collection's parameters CSV; its ID is the folder path relative
to ``data/``, e.g. ``seismic/building/portfolio/Hazus v6.1``. Within a
dataset, models are grouped into **collections** -- ``fragility``,
``consequence_repair`` and ``loss_repair`` -- and each model is one row of
a collection's table. The naming conventions are:

- collection parameters: ``<collection>.csv``
- collection metadata: ``<collection>.json`` (may be absent)
- input validation schema: ``input_schema.json`` (optional)
- Pelicun auto-population script: ``pelicun_config.py`` (optional)
- any other file is an extra file, reachable through :func:`file_path`

This module is pelicun-free and only resolves paths; it does not read file
contents.
"""

from __future__ import annotations

import difflib
import functools
from importlib import resources
from pathlib import Path
from typing import cast

#: Collections a dataset may provide, each stored as a parameters CSV and
#: an optional metadata JSON.
_COLLECTIONS = ('fragility', 'consequence_repair', 'loss_repair')
#: Conventional name of the input-validation schema file.
_SCHEMA_FILE = 'input_schema.json'
#: Conventional name of the Pelicun auto-population script.
_CONFIG_SCRIPT_FILE = 'pelicun_config.py'


class UnknownDatasetError(KeyError):
    """Raised when a dataset ID does not exist in the data tree."""


class DatasetFileNotFoundError(KeyError):
    """Raised when a requested file is absent for a dataset."""


def data_root() -> Path:
    """
    Return the root of the packaged library data tree.

    Returns
    -------
    Path
        A path-like handle to the ``data`` directory inside the
        installed ``dlml`` package.

    """
    # importlib.resources types this as a Traversable; for a normal
    # (unpacked) installation it is a concrete pathlib.Path.
    return cast(Path, resources.files('dlml') / 'data')


@functools.lru_cache(maxsize=1)
def _discover_dataset_ids() -> tuple[str, ...]:
    """
    Discover dataset IDs by walking the packaged data tree.

    A folder is a dataset if it directly contains at least one
    collection's parameters CSV. Dataset folders are not descended into.
    The result is cached as an immutable tuple.

    Returns
    -------
    tuple of str
        The dataset IDs (folder paths relative to ``data/``), sorted.

    """
    found: list[str] = []
    stack: list[tuple[Path, str]] = [(data_root(), '')]
    while stack:
        node, prefix = stack.pop()
        children = list(node.iterdir())
        names = {child.name for child in children if child.is_file()}
        if any(f'{collection}.csv' in names for collection in _COLLECTIONS):
            found.append(prefix)
            continue
        for child in children:
            # Intermediate folders hold only subfolders, so the false
            # branch (a loose file) never fires for real data.
            if child.is_dir():  # pragma: no branch
                rel = f'{prefix}/{child.name}' if prefix else child.name
                stack.append((child, rel))
    return tuple(sorted(found))


def dataset_ids() -> list[str]:
    """
    Return the discovered dataset IDs.

    Returns
    -------
    list of str
        The dataset IDs (folder paths relative to ``data/``), sorted. A
        new list is returned on each call.

    """
    return list(_discover_dataset_ids())


def _validate_dataset_id(dataset_id: str) -> None:
    """
    Check that a dataset ID is known.

    Raises
    ------
    UnknownDatasetError
        If the dataset ID does not exist. The error message lists the
        closest known IDs as suggestions.

    """
    known = dataset_ids()
    if dataset_id in known:
        return
    suggestions = difflib.get_close_matches(dataset_id, known, n=3)
    msg = f'unknown dataset ID: {dataset_id!r}'
    if suggestions:
        hint = ', '.join(repr(item) for item in suggestions)
        msg = f'{msg}. Did you mean: {hint}?'
    raise UnknownDatasetError(msg)


def _dataset_root(dataset_id: str) -> Path:
    """Return the resolved data folder for a validated dataset."""
    _validate_dataset_id(dataset_id)
    node = data_root()
    for part in dataset_id.split('/'):
        node = node / part
    return node


def available_collections(dataset_id: str) -> list[str]:
    """
    Return the collections a dataset provides parameters for.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    list of str
        The sorted collections whose parameters CSV exists.

    """
    root = _dataset_root(dataset_id)
    return sorted(c for c in _COLLECTIONS if (root / f'{c}.csv').is_file())


def _reject_unknown_collection(collection: str) -> None:
    """
    Check that a collection name is recognized.

    Raises
    ------
    DatasetFileNotFoundError
        If ``collection`` is not one of the known collections.

    """
    if collection not in _COLLECTIONS:
        msg = (
            f'{collection!r} is not a known collection; '
            f'expected one of {list(_COLLECTIONS)}'
        )
        raise DatasetFileNotFoundError(msg)


def parameters_path(dataset_id: str, collection: str) -> Path:
    """
    Resolve the parameters CSV for a collection.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    collection: str
        One of ``'fragility'``, ``'consequence_repair'`` or
        ``'loss_repair'``.

    Returns
    -------
    Path
        A path-like handle to ``<collection>.csv``.

    Raises
    ------
    DatasetFileNotFoundError
        If the dataset provides no parameters for the collection. The
        error lists the available collections.

    """
    root = _dataset_root(dataset_id)
    _reject_unknown_collection(collection)
    path = root / f'{collection}.csv'
    if not path.is_file():
        msg = (
            f"dataset '{dataset_id}' has no {collection} parameters; "
            f'available collections: {available_collections(dataset_id)}'
        )
        raise DatasetFileNotFoundError(msg)
    return path


def metadata_path(dataset_id: str, collection: str) -> Path:
    """
    Resolve the metadata JSON for a collection.

    Metadata may be absent even when the parameters CSV is present.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    collection: str
        One of ``'fragility'``, ``'consequence_repair'`` or
        ``'loss_repair'``.

    Returns
    -------
    Path
        A path-like handle to ``<collection>.json``.

    Raises
    ------
    DatasetFileNotFoundError
        If the dataset provides no metadata for the collection. The
        error lists the available collections.

    """
    root = _dataset_root(dataset_id)
    _reject_unknown_collection(collection)
    path = root / f'{collection}.json'
    if not path.is_file():
        msg = (
            f"dataset '{dataset_id}' has no {collection} metadata; "
            f'available collections: {available_collections(dataset_id)}'
        )
        raise DatasetFileNotFoundError(msg)
    return path


def schema_path(dataset_id: str) -> Path:
    """
    Resolve the input-validation schema for a dataset.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    Path
        A path-like handle to ``input_schema.json``.

    Raises
    ------
    DatasetFileNotFoundError
        If the dataset has no input schema.

    """
    root = _dataset_root(dataset_id)
    path = root / _SCHEMA_FILE
    if not path.is_file():
        msg = f"dataset '{dataset_id}' has no input schema"
        raise DatasetFileNotFoundError(msg)
    return path


def config_script_path(dataset_id: str) -> Path:
    """
    Resolve the Pelicun auto-population script for a dataset.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.

    Returns
    -------
    Path
        A path-like handle to ``pelicun_config.py``.

    Raises
    ------
    DatasetFileNotFoundError
        If the dataset has no config script.

    """
    root = _dataset_root(dataset_id)
    path = root / _CONFIG_SCRIPT_FILE
    if not path.is_file():
        msg = f"dataset '{dataset_id}' has no config script"
        raise DatasetFileNotFoundError(msg)
    return path


def file_path(dataset_id: str, filename: str) -> Path:
    """
    Resolve any file in a dataset's folder by its exact name.

    This is the escape hatch for extra files (e.g.
    ``combine_wind_flood.csv``) that fall outside the naming convention.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier.
    filename: str
        The exact name of a file in the dataset's folder.

    Returns
    -------
    Path
        A path-like handle to the file.

    Raises
    ------
    DatasetFileNotFoundError
        If the folder has no such file. The error lists the folder's
        files.

    """
    root = _dataset_root(dataset_id)
    path = root / filename
    if not path.is_file():
        present = sorted(c.name for c in root.iterdir() if c.is_file())
        msg = f"dataset '{dataset_id}' has no file {filename!r}; files: {present}"
        raise DatasetFileNotFoundError(msg)
    return path
