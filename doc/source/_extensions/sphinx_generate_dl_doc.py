import os
import subprocess

from sphinx.application import Sphinx


def run_script(app: Sphinx):
    """
    Run a custom Python script to generate files before Sphinx builds
    the documentation.

    Parameters
    ----------
    app: Sphinx
        The Sphinx application instance.
    """
    script_path = os.path.join(app.srcdir, '_extensions', 'generate_dl_doc.py')

    result = subprocess.run(['python', script_path], check=True)

    if result.returncode != 0:
        raise RuntimeError('Script execution failed')


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
