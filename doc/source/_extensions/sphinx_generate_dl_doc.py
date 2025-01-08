"""Configure Sphinx to run a custom script to generate files."""

import os
import subprocess
from pathlib import Path

from sphinx.application import Sphinx


def run_script(app: Sphinx):
    """
    Generate files before building docs.

    Run a custom Python script to generate files before Sphinx builds
    the documentation.

    Parameters
    ----------
    app: Sphinx
        The Sphinx application instance.
    """
    script_path = str(Path(app.srcdir) / '_extensions' / 'generate_dl_doc.py')

    result = subprocess.run(['python', script_path], check=True)  # noqa: S603, S607

    if result.returncode != 0:
        msg = 'Script execution failed'
        raise RuntimeError(msg)


def setup(app: Sphinx):
    """
    Set up the custom Sphinx extension.

    Parameters
    ----------
    app: Sphinx
        The Sphinx application instance.
    """
    app.connect('builder-inited', run_script)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
