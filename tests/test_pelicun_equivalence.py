"""Behavioral-equivalence gate: dlml reads its data exactly as pelicun does.

dlml's tabular deserializer (:mod:`dlml._tabular`) is a port of pelicun's
``load_data``. This test proves the port stays faithful: for every packaged
parameters table, ``dlml.get_parameters`` must equal
``pelicun.file_io.load_data(path, None, orientation=1, reindex=False)`` -- the
exact call dlml mirrors -- and the re-exported ``convert_to_MultiIndex`` must
match pelicun's on both axes.

This is the one test that needs pelicun (the ``test`` extra); it is skipped
where pelicun is unavailable. Importing pelicun currently triggers a one-time
DLML data download, so *before* the import we point ``DLML_DATA_DIR`` at a
throwaway directory holding the manifest file pelicun probes for plus a fresh
version-check cache -- making the import offline and side-effect-free.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import dlml
from dlml import _catalog

# Skip the whole module only when pelicun (the ``test`` extra) is genuinely not
# installed -- checked via find_spec, which locates the top-level package
# WITHOUT executing its __init__ (and thus without triggering the DLML
# download).
if importlib.util.find_spec('pelicun') is None:
    pytest.skip('pelicun (the test extra) is not installed', allow_module_level=True)

# --- Offline pelicun import (configured before pelicun is imported) ----------
_STUB_DATA_DIR = Path(tempfile.mkdtemp(prefix='dlml-pelicun-stub-'))
atexit.register(shutil.rmtree, _STUB_DATA_DIR, ignore_errors=True)
# The manifest's presence tells pelicun the data is installed (no download).
(_STUB_DATA_DIR / 'model_files.txt').write_text('', encoding='utf-8')
# A recent check timestamp short-circuits pelicun's daily network version
# check.
(_STUB_DATA_DIR / '.dlml_cache.json').write_text(
    json.dumps(
        {
            'last_version_check': datetime.now().isoformat(),  # noqa: DTZ005
            'update_available': False,
        }
    ),
    encoding='utf-8',
)
os.environ['DLML_DATA_DIR'] = str(_STUB_DATA_DIR)

# pelicun is installed, so with the guard above the import must succeed
# offline. Importing directly (not importorskip) makes a future guard
# regression a hard error here rather than a silently skipped gate.
import pelicun.base as pelicun_base  # noqa: E402
import pelicun.file_io as pelicun_file_io  # noqa: E402

_PAIRS = [
    (dataset, collection)
    for dataset in dlml.list_datasets()
    for collection in dlml.available_collections(dataset)
]
_PAIR_IDS = [f'{collection}::{dataset}' for dataset, collection in _PAIRS]
_EXPECTED_PAIR_COUNT = 18


def test_discovery_is_non_vacuous():
    """Guard the parametrized equivalence test against empty discovery."""
    assert len(_PAIRS) == _EXPECTED_PAIR_COUNT


@pytest.mark.parametrize(('dataset', 'collection'), _PAIRS, ids=_PAIR_IDS)
def test_get_parameters_matches_pelicun_load_data(dataset: str, collection: str):
    """``dlml.get_parameters`` returns exactly what pelicun's ``load_data``
    produces from the same parameters file."""
    path = _catalog.parameters_path(dataset, collection)
    expected = pelicun_file_io.load_data(
        str(path), None, orientation=1, reindex=False
    )
    actual = dlml.get_parameters(dataset, collection)
    pd.testing.assert_frame_equal(actual, expected)


def test_convert_to_multiindex_matches_pelicun():
    """dlml's re-exported ``convert_to_MultiIndex`` behaves identically to
    pelicun's when splitting dash-joined labels on either axis."""
    on_columns = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        columns=['Demand-Type', 'LS1-Theta_0', 'Incomplete'],
    )
    on_rows = pd.DataFrame({'value': [1, 2]}, index=['A-1', 'B-2'])
    for frame, axis in ((on_columns, 1), (on_rows, 0)):
        pd.testing.assert_frame_equal(
            dlml.convert_to_MultiIndex(frame.copy(), axis=axis),
            pelicun_base.convert_to_MultiIndex(frame.copy(), axis=axis),
        )
