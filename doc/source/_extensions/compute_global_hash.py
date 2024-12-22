"""Compute a global hash considering all `.csv` files."""

from pathlib import Path

from generate_dl_doc import combine_md5_hashes, generate_md5

resource_folder = Path('../../../')
dlmls = resource_folder.rglob('*.csv')
hashes = [generate_md5(dlml) for dlml in dlmls]
print(combine_md5_hashes(hashes))  # noqa: T201
