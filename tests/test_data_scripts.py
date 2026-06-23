"""
Static import audit for the Python scripts shipped in the data tree.

The library ships executable Python beside its data: the per-dataset
``pelicun_config.py`` auto-population scripts and their sibling helper
modules (e.g. ``FloodRulesets.py``). Pelicun runs these at simulation
time, so each may import only from a tightly controlled set -- otherwise a
``dlml`` user could hit a surprise missing dependency mid-run.

This module enforces that policy *statically*. The audit parses every
script with :mod:`ast` and never imports or executes it. The point is
to catch future drift -- a new script quietly adding a forbidden import.

A data-tree script may import only (by top-level module name):

- any Python standard-library module (via :data:`sys.stdlib_module_names`,
  which also covers ``__future__``);
- ``pelicun`` (and any ``pelicun.*`` submodule);
- ``pandas``, ``numpy`` or ``jsonschema``;
- a sibling module -- a name matching another ``.py`` file's stem in the
  same folder;
- any name declared in a module-level ``REQUIRES = [...]`` list of string
  literals in that script (an escape valve for a genuine extra dependency).

:data:`sys.stdlib_module_names` exists only on Python 3.10+, so the whole
module is skipped on older interpreters; running the static audit on the
3.12 dev interpreter is sufficient.

The audit is static: it sees ``import`` / ``from`` statements at any
nesting depth, but not dynamic imports (``importlib.import_module``,
``__import__``, ``exec``). That is an accepted limitation -- these scripts
are short, authored in-repo, and use plain imports, and a genuine extra
dependency has the ``REQUIRES`` valve. A script that fails to parse
surfaces as a parse error, the intended loud failure for malformed code.
"""

from __future__ import annotations

import ast
import sys
from typing import TYPE_CHECKING

import pytest

from dlml._catalog import data_root

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason='sys.stdlib_module_names requires Python 3.10+',
)

# Third-party modules every data-tree script may always import. The
# standard library and per-folder siblings/REQUIRES are added on top.
_ALWAYS_ALLOWED = frozenset({'pelicun', 'pandas', 'numpy', 'jsonschema'})

# Sentinel surfaced for a relative import (``from . import x``). None exist
# today; flagging it keeps such an import from slipping through unaudited.
_RELATIVE_IMPORT_MARKER = '<relative import: needs review, not allowed by policy>'

# Scripts the audit must discover. Guards the parametrized test against
# silently passing because discovery found nothing.
_EXPECTED_SCRIPTS = [
    'flood/building/portfolio/Hazus v6.1/FloodRulesets.py',
    'flood/building/portfolio/Hazus v6.1/pelicun_config.py',
    'hurricane/building/portfolio/Hazus v5.1 coupled/pelicun_config.py',
    'seismic/building/portfolio/Hazus v6.1/pelicun_config.py',
    'seismic/building/subassembly/Hazus v5.1/pelicun_config.py',
    'seismic/power_network/portfolio/Hazus v5.1/pelicun_config.py',
    'seismic/transportation_network/portfolio/Hazus v5.1/pelicun_config.py',
    'seismic/water_network/portfolio/Hazus v6.1/pelicun_config.py',
]


def _discover_scripts() -> list[Path]:
    """
    Find every Python script shipped in the data tree.

    Returns
    -------
    list of Path
        Absolute paths to all ``.py`` files under the data root, sorted
        deterministically by their path relative to that root.

    """
    root = data_root()
    return sorted(root.rglob('*.py'), key=lambda p: p.relative_to(root).as_posix())


def _imported_top_level_modules(source: str) -> set[str]:
    """
    Collect the top-level module names imported by some Python source.

    The source is parsed with :mod:`ast` and walked in full, so imports
    nested inside functions or conditionals are captured too. For
    ``import a.b.c`` the result holds ``'a'``; for ``from a.b import c`` it
    holds ``'a'``. A relative import (``from . import x``, ``node.level >
    0``) has no top-level module name and is not allowed by policy, so it
    is surfaced as :data:`_RELATIVE_IMPORT_MARKER` for review.

    Parameters
    ----------
    source : str
        The Python source code to analyze.

    Returns
    -------
    set of str
        The imported top-level module names, plus the relative-import
        marker if any relative import is present.

    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split('.', 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                modules.add(_RELATIVE_IMPORT_MARKER)
            elif node.module is not None:
                modules.add(node.module.split('.', 1)[0])
    return modules


def _declared_requires(source: str) -> set[str]:
    """
    Read a script's module-level ``REQUIRES`` allow-list, if any.

    Locates a top-level assignment (``REQUIRES = [...]`` or
    ``REQUIRES: list[str] = [...]``) whose value is a list of string
    literals and returns those strings. Anything else (no ``REQUIRES``, a
    non-list value, or non-string elements) yields an empty set.

    Parameters
    ----------
    source : str
        The Python source code to analyze.

    Returns
    -------
    set of str
        The module names whitelisted by the script's ``REQUIRES`` list.

    """
    requires: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        targets: Iterable[ast.expr]
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if 'REQUIRES' not in names or not isinstance(node.value, ast.List):
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                requires.add(element.value)
    return requires


def _allowed_modules(py_path: Path) -> set[str]:
    """
    Compute the set of modules a given data-tree script may import.

    The allowed set is the standard library, the always-allowed
    third-party modules, the stems of the sibling ``.py`` files in the
    same folder, and any names declared in the script's ``REQUIRES`` list.

    Parameters
    ----------
    py_path : Path
        Path to the data-tree script being audited.

    Returns
    -------
    set of str
        Every top-level module name the script is permitted to import.

    """
    siblings = {
        sibling.stem for sibling in py_path.parent.glob('*.py') if sibling != py_path
    }
    return (
        set(sys.stdlib_module_names)
        | set(_ALWAYS_ALLOWED)
        | siblings
        | _declared_requires(py_path.read_text(encoding='utf-8'))
    )


def _disallowed_imports(py_path: Path) -> list[str]:
    """
    Return the imports of a data-tree script that policy forbids.

    Parameters
    ----------
    py_path : Path
        Path to the data-tree script to audit.

    Returns
    -------
    list of str
        The sorted top-level module names the script imports that are not
        in its allowed set. Empty when the script is compliant. A relative
        import surfaces as :data:`_RELATIVE_IMPORT_MARKER`.

    """
    source = py_path.read_text(encoding='utf-8')
    imported = _imported_top_level_modules(source)
    allowed = _allowed_modules(py_path)
    return sorted(imported - allowed)


# ---------------------------------------------------------------------------
# Discovery is non-vacuous
# ---------------------------------------------------------------------------


def test_discovery_finds_the_known_scripts():
    """The audit must find at least every known data-tree script."""
    root = data_root()
    discovered = {path.relative_to(root).as_posix() for path in _discover_scripts()}
    assert set(_EXPECTED_SCRIPTS) <= discovered


# ---------------------------------------------------------------------------
# Every real script passes the audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'script',
    _discover_scripts(),
    ids=lambda p: p.relative_to(data_root()).as_posix(),
)
def test_data_script_imports_are_allowed(script: Path):
    """Each shipped data-tree script imports only permitted modules."""
    assert _disallowed_imports(script) == []


# ---------------------------------------------------------------------------
# The audit actually works (self-tests on synthetic scripts)
# ---------------------------------------------------------------------------


def test_forbidden_import_is_flagged(tmp_path: Path):
    """A forbidden import is reported; an allowed one is not."""
    script = tmp_path / 'pelicun_config.py'
    script.write_text(
        'import json\nimport pandas as pd\nimport shapely\n',
        encoding='utf-8',
    )
    assert _disallowed_imports(script) == ['shapely']


def test_requires_whitelists_an_extra_dependency(tmp_path: Path):
    """A module named in ``REQUIRES`` is no longer flagged."""
    script = tmp_path / 'pelicun_config.py'
    script.write_text(
        "REQUIRES = ['shapely']\nimport shapely\n",
        encoding='utf-8',
    )
    assert _disallowed_imports(script) == []


def test_sibling_module_import_is_allowed(tmp_path: Path):
    """A name matching a sibling ``.py`` file's stem is permitted."""
    (tmp_path / 'Helpers.py').write_text('x = 1\n', encoding='utf-8')
    script = tmp_path / 'pelicun_config.py'
    script.write_text(
        'from Helpers import x\n',
        encoding='utf-8',
    )
    assert _disallowed_imports(script) == []


def test_import_nested_in_function_is_audited(tmp_path: Path):
    """Imports inside functions are audited, not just module-level ones."""
    script = tmp_path / 'pelicun_config.py'
    script.write_text(
        'def run():\n    import shapely\n    return shapely\n',
        encoding='utf-8',
    )
    assert _disallowed_imports(script) == ['shapely']


def test_relative_import_is_flagged_for_review(tmp_path: Path):
    """A relative import is surfaced as disallowed for review."""
    script = tmp_path / 'pelicun_config.py'
    script.write_text(
        'from . import helpers\n',
        encoding='utf-8',
    )
    assert _disallowed_imports(script) == [_RELATIVE_IMPORT_MARKER]
