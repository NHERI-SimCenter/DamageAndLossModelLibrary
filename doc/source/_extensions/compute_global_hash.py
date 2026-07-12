"""Compute a global hash of the packaged `.csv` files (the doc-cache key)."""

from pathlib import Path

from dlml._catalog import data_root
from generate_dl_doc import combine_md5_hashes, generate_md5

# Hash exactly the packaged data tree (sorted for a stable order), so the doc
# cache key changes only when a shipped model CSV changes.
dlmls = sorted(Path(str(data_root())).rglob('*.csv'))
hashes = [generate_md5(dlml) for dlml in dlmls]
print(combine_md5_hashes(hashes))  # noqa: T201
