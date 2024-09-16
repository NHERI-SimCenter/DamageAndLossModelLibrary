from generate_dl_doc import generate_md5, combine_md5_hashes
from pathlib import Path

resource_folder = Path('../../../')
dlmls = resource_folder.rglob('*.csv')
hashes = [generate_md5(dlml) for dlml in dlmls]
print(combine_md5_hashes(hashes))
