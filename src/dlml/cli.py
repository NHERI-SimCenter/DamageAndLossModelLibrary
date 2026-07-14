"""
Command-line interface for the Damage and Loss Model Library.

The console-script entry point (``dlml``, registered in ``pyproject.toml``)
exposes three subcommands:

* ``dlml explorer`` -- launch the DLML Explorer (needs the ``[explorer]`` extra).
* ``dlml list`` -- list the packaged datasets and their collections.
* ``dlml info <dataset>`` -- show a dataset's collections and model counts.

``list`` and ``info`` use only the core dependencies, so they work from a bare
``pip install simcenter-dlml``; ``explorer`` checks for the optional
dependencies and prints an install hint if they are missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from importlib import resources
from pathlib import Path

from dlml import api

#: The app's base theme, mirrored from ``.streamlit/config.toml``. Streamlit
#: reads that file only from the working directory or the user's home, never
#: from inside an installed package, so ``dlml explorer`` passes these as
#: ``--theme.*`` flags to give pip users the intended base palette. The in-app
#: light/dark toggle is a CSS overlay and works regardless of these values.
_THEME = {
    'base': 'light',
    'primaryColor': '#C8382E',
    'backgroundColor': '#F4F6FB',
    'secondaryBackgroundColor': '#FFFFFF',
    'textColor': '#171C26',
    'font': 'sans serif',
}


def _streamlit_installed() -> bool:
    """Return whether the optional ``streamlit`` dependency is importable."""
    return importlib.util.find_spec('streamlit') is not None


def _locate_app() -> Path:
    """Return the path to the packaged Streamlit entry script."""
    return Path(str(resources.files('dlml.web'))) / 'app.py'


def _theme_args() -> list[str]:
    """Return the Streamlit ``--theme.*`` flags carrying the base palette."""
    args: list[str] = []
    for key, value in _THEME.items():
        args += [f'--theme.{key}', value]
    return args


def _explorer_command(app_path: Path, port: int | None) -> list[str]:
    """Assemble the headless, themed ``streamlit run`` command line.

    ``--server.port`` is added only when *port* is given; otherwise Streamlit
    uses its own default (typically 8501) and prints the URL itself.
    """
    command = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        str(app_path),
        '--server.headless',
        'true',
        '--browser.gatherUsageStats',
        'false',
        *_theme_args(),
    ]
    if port is not None:
        command += ['--server.port', str(port)]
    return command


def _prefetch_search_models() -> None:
    """Download the search models up front so first search isn't a silent wait.

    Best-effort: if the search extra is absent, or the download fails (e.g.
    offline), the explorer still launches and degrades gracefully in the
    browser.
    """
    if importlib.util.find_spec('fastembed') is None:
        return
    print(
        'Preparing the search models '
        '(first run downloads them from HuggingFace; cached afterwards) …',
        flush=True,
    )
    try:
        module = importlib.import_module('dlml.web.st_search.semantic_index')
        module.prefetch_embedding_models()
    except Exception as exc:  # noqa: BLE001 -- non-fatal; search retries on demand
        print(
            f'  note: could not preload the search models ({exc}); '
            'they will download on first search instead',
            file=sys.stderr,
        )


def _cmd_explorer(port: int | None) -> int:
    """Launch the DLML Explorer web app; return the process exit code."""
    if not _streamlit_installed():
        print(
            "The 'dlml explorer' command needs the optional explorer "
            "dependencies.\nInstall them with:  pip install 'simcenter-dlml[explorer]'",
            file=sys.stderr,
        )
        return 1
    app_path = _locate_app()
    if not app_path.is_file():
        print(f'could not locate the packaged app at: {app_path}', file=sys.stderr)
        return 1
    _prefetch_search_models()
    if port is None:
        print(
            'Launching the DLML Explorer — Streamlit will print the URL below.',
            flush=True,
        )
    else:
        print(f'Launching the DLML Explorer at http://localhost:{port}', flush=True)
    return subprocess.call(_explorer_command(app_path, port))  # noqa: S603


def _cmd_list() -> int:
    """Print every packaged dataset ID with its collections."""
    for dataset_id in api.list_datasets():
        collections = ', '.join(api.available_collections(dataset_id))
        print(f'{dataset_id}  [{collections}]')
    return 0


def _cmd_info(dataset_id: str) -> int:
    """Print a dataset's collections and per-collection model counts."""
    try:
        collections = api.available_collections(dataset_id)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(dataset_id)
    for collection in collections:
        count = len(api.get_parameters(dataset_id, collection))
        print(f'  {collection}: {count} models')
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``dlml`` console script."""
    parser = argparse.ArgumentParser(
        prog='dlml',
        description='Damage and Loss Model Library command-line interface.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    explorer = sub.add_parser('explorer', help='launch the DLML Explorer web app')
    explorer.add_argument(
        '--port',
        type=int,
        default=None,
        help='port to serve on (default: the Streamlit default, typically 8501)',
    )

    sub.add_parser('list', help='list the packaged datasets and their collections')

    info = sub.add_parser(
        'info', help="show a dataset's collections and model counts"
    )
    info.add_argument('dataset', help="dataset ID (see 'dlml list')")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    args = _build_parser().parse_args(argv)
    if args.command == 'list':
        return _cmd_list()
    if args.command == 'info':
        return _cmd_info(args.dataset)
    return _cmd_explorer(args.port)


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
