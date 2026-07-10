"""Packaging gate: the built wheel ships exactly the library, nothing more.

Building the wheel and inspecting it guards against two failure modes the
other tests cannot see: a library data file silently dropped from the
package, and authoring material (the ``data_sources/`` mirror, spreadsheets,
notebooks, scratch files) leaking in and ballooning the download past PyPI's
limits.

The wheel is built once per session with ``uv`` -- the project's build tool,
and what ``scripts/check.sh`` already runs under -- so these tests are skipped
only where ``uv`` is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

from dlml._catalog import data_root

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_PREFIX = 'dlml/data/'

# The wheel must stay under PyPI's per-file limit; this also trips long
# before any authoring leak (the spreadsheet sources alone are ~80 MB).
_MAX_WHEEL_MB = 25

# Data-tree files ship as plain text; nothing else belongs in the package.
_ALLOWED_DATA_SUFFIXES = frozenset({'.py', '.csv', '.json'})

# Substrings that must never appear in a wheel entry -- authoring material and
# build droppings that live outside the shipped library.
_FORBIDDEN_TOKENS = (
    'data_sources',
    '__pycache__',
    '.pyc',
    '.ipynb',
    '.xls',
    '.dill',
    'source_',
    '_header',
    '_filtered',
    'scratch',
)


@pytest.fixture(scope='session')
def wheel() -> tuple[list[str], int]:
    """Build the wheel once and return its (entry names, size in bytes)."""
    uv = shutil.which('uv')
    if uv is None:
        pytest.skip('building the wheel requires uv')
    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run(  # noqa: S603  (fixed command, resolved uv path)
            [uv, 'build', '--wheel', '--out-dir', out_dir],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        built = list(Path(out_dir).glob('*.whl'))
        assert len(built) == 1, f'expected exactly one wheel, got {built}'
        names = zipfile.ZipFile(built[0]).namelist()
        size = built[0].stat().st_size
    return names, size


def _data_entries(names: list[str]) -> set[str]:
    """The data-tree paths in the wheel, relative to ``dlml/data/``."""
    return {n[len(_DATA_PREFIX) :] for n in names if n.startswith(_DATA_PREFIX)}


def test_wheel_ships_every_library_data_file(wheel: tuple[list[str], int]):
    """The wheel's data tree is exactly the packaged data on disk -- no file
    dropped, none added."""
    names, _ = wheel
    in_wheel = _data_entries(names)
    on_disk = {
        path.relative_to(data_root()).as_posix()
        for path in data_root().rglob('*')
        if path.is_file()
        and '__pycache__' not in path.parts
        and path.suffix != '.pyc'
    }
    assert in_wheel == on_disk, (
        f'missing={sorted(on_disk - in_wheel)[:3]} '
        f'extra={sorted(in_wheel - on_disk)[:3]}'
    )


def test_wheel_ships_the_package_modules(wheel: tuple[list[str], int]):
    """The public package modules are present in the wheel."""
    names = set(wheel[0])
    for module in (
        'dlml/__init__.py',
        'dlml/api.py',
        'dlml/vocabulary.py',
        'dlml/_catalog.py',
        'dlml/_tabular.py',
    ):
        assert module in names, f'missing module {module}'


def test_wheel_excludes_authoring_material(wheel: tuple[list[str], int]):
    """No authoring file leaks in, and every data-tree file is a known text
    format."""
    names, _ = wheel
    for name in names:
        lowered = name.lower()
        present = [token for token in _FORBIDDEN_TOKENS if token in lowered]
        assert not present, f'forbidden {present} in wheel entry {name!r}'
    for entry in _data_entries(names):
        assert (
            Path(entry).suffix in _ALLOWED_DATA_SUFFIXES
        ), f'unexpected file type in data tree: {entry!r}'


def test_wheel_contains_only_library_and_metadata(wheel: tuple[list[str], int]):
    """Every entry is a data file, a package module, or wheel metadata -- so a
    stray non-code file left under ``src/dlml/`` (outside ``data/``) cannot
    ride along unnoticed, which the per-tree checks above would miss."""
    names, _ = wheel
    for name in names:
        if name.startswith(_DATA_PREFIX):
            continue  # data tree -- validated by the suffix allow-list above
        if name.startswith('dlml/'):
            assert name.endswith('.py'), f'non-code file in package: {name!r}'
        elif '.dist-info/' in name:
            continue  # wheel metadata
        else:
            pytest.fail(f'unexpected wheel entry: {name!r}')


def test_wheel_size_under_threshold(wheel: tuple[list[str], int]):
    """The wheel stays well under PyPI's limit (a leak would balloon it)."""
    megabytes = wheel[1] / (1024 * 1024)  # binary MB (2**20 bytes)
    assert megabytes < _MAX_WHEEL_MB, f'wheel is {megabytes:.1f} MB'
