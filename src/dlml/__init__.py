"""Damage and Loss Model Library (dlml)."""

from importlib import metadata

try:
    __version__ = metadata.version('dlml')
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0+local'
