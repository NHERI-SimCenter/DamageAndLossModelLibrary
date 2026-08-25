# Damage and Loss Model Library (DLML)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A curated, open-source, version-controlled library of standardized damage and
loss models for natural-hazard risk assessment.

---

## About the library

The Damage and Loss Model Library (DLML) is a project from the
[NHERI SimCenter](https://simcenter.designsafe-ci.org/) that addresses a
persistent gap in natural-hazards engineering: the lack of a centralized,
standardized, and easy-to-use home for damage and loss models. It gathers
performance models from established sources — the component library from
**FEMA P-58** and building- and infrastructure-level models from **FEMA
Hazus** — alongside peer-reviewed models from the research community, and
publishes them in one consistent, machine-readable form.

The library is built on three principles:

- **Version-controlled.** Every model belongs to a released snapshot, so an
  analysis can be tied permanently to a specific version of the data —
  supporting reproducible research and defensible professional practice.
- **Standardized and machine-readable.** Models are converted to a single
  tabular schema (CSV parameters plus JSON metadata), ready for computational
  workflows with no manual transcription.
- **Documented provenance.** Every model carries rich metadata and a citation
  to its source, so users can trace it to its origin and credit the original
  developers.

**Hazard coverage:** seismic, hurricane, and flood — spanning building
components, whole buildings, and infrastructure network assets.

## Data Organization

The data is a tree of **datasets**. A *dataset* is a leaf folder identified by
its path (`hazard/asset_type/resolution/methodology`), for example
`seismic/building/component/FEMA P-58 2nd Edition`. Each dataset provides one or
more **collections** — `fragility`, `consequence_repair`, and/or `loss_repair` —
and each collection is a table of **models**, one per row (typically one per
component/building/asset). Parameters are in `<collection>.csv`; the matching
`<collection>.json` holds the metadata (descriptions, limit and damage state
details, citations).

## How to use the DLML

Each of these is a different route to the same curated data.

### Install the Python package

```bash
pip install simcenter-dlml
```

The data ships inside the package (no runtime download), and you can read it 
with the `dlml` API:

```python
import dlml

dlml.list_datasets()
# ['flood/building/portfolio/Hazus v6.1',
#  'seismic/building/component/FEMA P-58 2nd Edition', ...]

fema_p58 = "seismic/building/component/FEMA P-58 2nd Edition"
dlml.available_collections(fema_p58)      # ['consequence_repair', 'fragility']
fragility = dlml.get_fragility(fema_p58)  # a pandas DataFrame of model parameters
metadata = dlml.get_metadata(fema_p58, "fragility")

# Validate asset features against a dataset's input schema:
seismic = "seismic/building/portfolio/Hazus v6.1"
dlml.validate_asset(seismic, {"StructureType": "W1", "DesignLevel": "Pre-Code"})
# []  -> valid
dlml.validate_asset(seismic, {"StructureType": "W1"})
# ["$: 'DesignLevel' is a required property"]
```

The core package needs only Python 3.9+, pandas, numpy, and jsonschema.

### Explore the models with the DLML Explorer

The **DLML Explorer** is a web app for discovering, visualizing, and assembling
models: search and filter the library, inspect interactive fragility curves and
consequence functions, add components to a selection, and download a
project-ready bundle. The easiest way to try it is the hosted version:

**[Open the DLML Explorer »](https://dlml-explorer.streamlit.app)**

You can also run it locally — for example, to explore alongside your own private
models — by installing the explorer extra:

```bash
pip install "simcenter-dlml[explorer]"
dlml explorer
```

(The explorer extra targets Python 3.10+.) And you can inspect the library
straight from the command line:

```bash
dlml list                 # every dataset and its collections
dlml info <dataset-id>    # a dataset's collections and model counts
```

### Use with Pelicun and the SimCenter tools

The library is the data backbone for
[Pelicun](https://github.com/NHERI-SimCenter/Pelicun), SimCenter's open-source
damage-and-loss engine, and is available automatically through the **PBE** and
**R2D** desktop applications that run Pelicun under the hood.

Pelicun 3.10 and later install this library from PyPI and need no further setup.
Pelicun 3.9 and earlier read the model data from this repository instead; see
[Release policy](#release-policy) if you are still on one of those versions.

### Work with the raw data

You can also read the CSV/JSON files directly from a clone of this repository:

```bash
git clone https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary.git
```

> A legacy [documentation website](https://nheri-simcenter.github.io/DamageAndLossModelLibrary/)
> with auto-generated model pages also remains available; it will be revised over
> time to complement the DLML Explorer.

## Release policy

Releases are published to PyPI as [`simcenter-dlml`](https://pypi.org/project/simcenter-dlml/)
and tagged here, with matching entries in [`CHANGELOG.md`](CHANGELOG.md). Version
numbers use two segments (`3.0`, `3.1`, ...), and every model in a release is a
fixed snapshot, so an analysis can cite the exact data it used.

### Migrating from Pelicun 3.9 and earlier

Pelicun 3.9 and earlier downloaded the model data from this repository's latest
GitHub release when you imported them. Since v3.0 the data lives under
`src/dlml/data/`, a layout those versions cannot read, so they now fail with a
download error.

We recommend upgrading. Pelicun 3.10 and later depend on `simcenter-dlml` directly
and download nothing:

```bash
pip install --upgrade pelicun
```

If you need to stay on Pelicun 3.9 or earlier, pin the model data to the last
release those versions can read:

```bash
pelicun dlml update v2.1.0
```

The `v2.1.0` release stays available for this purpose.

## Contributing

We welcome new and improved models from researchers and practitioners. A formal
contributor's guide is in progress; for now, the best way to start is to open a
GitHub Issue to discuss the model data you'd like to add. Our goal is a simple
pull-request workflow for data that conforms to the schema.

## License and acknowledgments

This library is distributed under the BSD 3-Clause license. See `LICENSE` for
details.

We gratefully acknowledge the generous in-kind contribution from Degenkolb
Engineers — and in particular Tshajlij Lee and Hannah Thompson — to the
development of the DLML Explorer.

This material is based upon work supported by the U.S. National Science
Foundation under Grants No. 1612843 and No. 2131111. Any opinions, findings, and
conclusions or recommendations expressed in this material are those of the
authors and do not necessarily reflect the views of the U.S. National Science
Foundation.

## Contact

For questions or support, please open an issue on the GitHub repository.

Adam Zsarnóczay, NHERI SimCenter, Stanford University — adamzs@stanford.edu
