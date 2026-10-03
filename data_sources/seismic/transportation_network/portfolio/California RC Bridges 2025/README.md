# California RC Bridges 2025: bridge system models

The packaged dataset `src/dlml/data/seismic/transportation_network/portfolio/California RC Bridges 2025`
holds the bridge system fragilities of the Chen et al. (2025) database. It is generated together
with the component dataset by one script, which lives in the component authoring folder:

    data_sources/seismic/transportation_network/component/California RC Bridges 2025/

That folder holds the generator (`generate_library_files.py`), its inputs, and
`id_crosswalk.csv`, which maps every model ID of both datasets to its row in the source file.
The inputs are the source table (`source/`), the citations (`references.json`), the dataset
descriptions (`general_information.json`), and the model-level text (`metadata_text.json`:
phrase tables, source-study paragraphs, correction sentences, and damage-state descriptions).
Edit the inputs there and rerun the generator to update this dataset; do not edit the packaged
files by hand.
