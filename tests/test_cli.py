"""Tests for the ``dlml`` command-line interface.

``list`` and ``info`` are exercised against the real packaged data; ``web`` is
tested with the Streamlit launch mocked out, so no server is started and the
optional web dependencies are not required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dlml import api, cli


def test_list_prints_every_dataset(capsys):
    rc = cli.main(['list'])
    assert rc == 0
    out = capsys.readouterr().out
    datasets = api.list_datasets()
    assert datasets  # there is packaged data to list
    for dataset_id in datasets:
        assert dataset_id in out
    # Collections are shown in brackets next to each dataset.
    assert '[' in out


def test_info_reports_collections_and_counts(capsys):
    dataset_id = api.list_datasets()[0]
    rc = cli.main(['info', dataset_id])
    assert rc == 0
    out = capsys.readouterr().out
    assert dataset_id in out
    assert 'models' in out
    for collection in api.available_collections(dataset_id):
        assert collection in out


def test_info_unknown_dataset_errors(capsys):
    rc = cli.main(['info', 'no/such/dataset'])
    assert rc == 1
    err = capsys.readouterr().err
    assert 'unknown dataset' in err.lower()


def test_explorer_without_streamlit_prints_install_hint(monkeypatch, capsys):
    monkeypatch.setattr(cli, '_streamlit_installed', lambda: False)
    rc = cli.main(['explorer'])
    assert rc == 1
    assert 'dlml[explorer]' in capsys.readouterr().err


def test_explorer_missing_app_errors(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, '_streamlit_installed', lambda: True)
    monkeypatch.setattr(cli, '_locate_app', lambda: tmp_path / 'nope.py')
    rc = cli.main(['explorer'])
    assert rc == 1
    assert 'could not locate' in capsys.readouterr().err


def test_explorer_launches_streamlit(monkeypatch, tmp_path):
    app = tmp_path / 'app.py'
    app.write_text('')
    monkeypatch.setattr(cli, '_streamlit_installed', lambda: True)
    monkeypatch.setattr(cli, '_locate_app', lambda: app)
    monkeypatch.setattr(cli, '_prefetch_search_models', lambda: None)
    captured: dict = {}

    def fake_call(cmd):
        captured['cmd'] = cmd
        return 0

    monkeypatch.setattr(cli.subprocess, 'call', fake_call)
    rc = cli.main(['explorer', '--port', '1234'])
    assert rc == 0
    cmd = captured['cmd']
    assert str(app) in cmd
    assert '1234' in cmd
    assert '--server.headless' in cmd
    assert '--theme.primaryColor' in cmd


def test_explorer_uses_streamlit_default_port(monkeypatch, tmp_path, capsys):
    app = tmp_path / 'app.py'
    app.write_text('')
    monkeypatch.setattr(cli, '_streamlit_installed', lambda: True)
    monkeypatch.setattr(cli, '_locate_app', lambda: app)
    monkeypatch.setattr(cli, '_prefetch_search_models', lambda: None)
    captured: dict = {}

    def fake_call(cmd):
        captured['cmd'] = cmd
        return 0

    monkeypatch.setattr(cli.subprocess, 'call', fake_call)
    rc = cli.main(['explorer'])
    assert rc == 0
    assert '--server.port' not in captured['cmd']
    assert 'Streamlit will print the URL' in capsys.readouterr().out


def test_explorer_command_is_headless_and_themed():
    cmd = cli._explorer_command(Path('app.py'), 8502)
    assert cmd[:4] == [sys.executable, '-m', 'streamlit', 'run']
    assert '--server.headless' in cmd
    port_idx = cmd.index('--server.port')
    assert cmd[port_idx + 1] == '8502'
    assert '--theme.base' in cmd


def test_explorer_command_omits_port_when_none():
    cmd = cli._explorer_command(Path('app.py'), None)
    assert '--server.port' not in cmd


def test_prefetch_skips_when_fastembed_absent(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda *_: None)
    cli._prefetch_search_models()
    assert capsys.readouterr().out == ''


def test_prefetch_runs_when_available(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda *_: object())
    fake = SimpleNamespace(prefetch_embedding_models=lambda: None)
    monkeypatch.setattr(cli.importlib, 'import_module', lambda *_: fake)
    cli._prefetch_search_models()
    assert 'Preparing the search models' in capsys.readouterr().out


def test_prefetch_reports_failure_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda *_: object())

    def boom(*_):
        raise RuntimeError

    monkeypatch.setattr(cli.importlib, 'import_module', boom)
    cli._prefetch_search_models()
    captured = capsys.readouterr()
    assert 'Preparing the search models' in captured.out
    assert 'could not preload' in captured.err


def test_theme_args_are_flag_value_pairs():
    args = cli._theme_args()
    assert len(args) % 2 == 0
    assert '--theme.base' in args
    assert 'light' in args


def test_locate_app_points_at_shipped_file():
    app = cli._locate_app()
    assert app.name == 'app.py'
    assert app.is_file()


def test_streamlit_installed_returns_bool():
    assert isinstance(cli._streamlit_installed(), bool)


def test_no_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main([])


def _read_config_theme() -> dict:
    """Parse the ``[theme]`` table from the repo's .streamlit/config.toml.

    A tiny hand parser (flat ``key = "value"`` lines) rather than tomllib, so
    the test runs on Python 3.9/3.10 too, where tomllib is unavailable.
    """
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / '.streamlit' / 'config.toml'
    assert config.is_file(), f'expected {config} to exist'
    theme: dict = {}
    in_theme = False
    for raw in config.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            in_theme = line == '[theme]'
            continue
        if in_theme and '=' in line:
            key, value = line.split('=', 1)
            theme[key.strip()] = value.strip().strip('"')
    return theme


def test_theme_matches_streamlit_config():
    # cli._THEME is the single source of truth for the base theme, passed as
    # --theme.* flags by `dlml explorer`. The repo's .streamlit/config.toml
    # (read by `streamlit run` directly, in dev) must mirror it exactly so both
    # launch paths look identical. This guards against the two drifting.
    assert _read_config_theme() == cli._THEME
