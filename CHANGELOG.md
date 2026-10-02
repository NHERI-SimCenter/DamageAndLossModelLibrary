# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **California RC bridge fragilities (Chen et al. 2025):** two datasets from the Sa(1.0 s)-based fragility database of Chen et al. (2025, Earthquake Spectra 41(4)), published on DesignSafe as PRJ-5910. `seismic/transportation_network/portfolio/California RC Bridges 2025` holds 116 bridge system models and `seismic/transportation_network/component/California RC Bridges 2025` holds 801 bridge component models, covering 26 bridge groups by design era, span count, bent type, and abutment type. Each model's metadata cites its source study and records every correction made to the published values or labels.
- **DLML Explorer:** datasets that are not from FEMA, Hazus, or SimCenter show a "Research" badge in the model tree.
- **Bridge component demand types:** twelve new `EDP_TYPES` entries for fragility models conditioned on the response of a bridge component: Peak Column Curvature Ductility (`CCD`), Peak Column Drift Ratio (`CDR`), Peak Joint Opening (`JOP`), Peak Abutment Active Displacement (`ABA`), Peak Abutment Passive Displacement (`ABP`), Peak Abutment Transverse Displacement (`ABT`), Peak Bearing Deformation (`BRD`), Peak Deck Displacement (`DEC`), Peak Foundation Translation (`FNT`), Peak Foundation Rotation (`FNR`), Peak Shear Key Deformation (`KEY`), and Permanent Approach Settlement (`PAS`).
- **California RC bridge EDP-based fragilities (Chen et al. 2025):** `seismic/transportation_network/component/California RC Bridges 2025 EDP` holds 25 bridge component models from the EDP capacity table of the Chen et al. (2025) database. Each model is conditioned on the response of the component (peak values; the permanent settlement for approaches) instead of Sa(1.0 s): column curvature ductility or drift ratio; joint opening; abutment, deck, or foundation displacement; bearing or shear key deformation; foundation rotation; or approach settlement. Each model's metadata names the source of its capacities and records every change made to the published values.

### Changed
- **Distribution:** v3.0 and v3.1 are now published as GitHub Releases, and `releases/latest` resolves to the current version rather than to v2.1.0. Pelicun 3.10 and later are unaffected — they install `simcenter-dlml` from PyPI. Pelicun 3.9 and earlier, which read the model data from the latest GitHub release, no longer work against it; see the README's release policy for the upgrade path and for pinning the data to v2.1.0.

### Removed
- The root `model_files.txt` manifest, which described the pre-v3.0 top-level data layout and no longer matched the packaged data.

---

## [3.1] - 2026-07-22

This release extends the controlled vocabularies with unit-type information for demand types. The model data itself is unchanged.

### Added
- **Unit types in the demand-type vocabulary:** the new canonical `EDP_TYPES` mapping records each demand type's short acronym and the physical quantity it is measured in (`UnitType`), and the new `UNIT_TYPES` vocabulary enumerates the valid unit types: `acceleration`, `speed`, `displacement`, `unitless`, `rotation`, `force`, `force_per_length`, and `pressure`. Consumers (e.g., Pelicun 3.10+) can infer physical units for demand data from the vocabulary instead of maintaining hardcoded lists.

### Changed
- `EDP_to_demand_type` is now derived from `EDP_TYPES`. Its content is identical to before, so existing consumers are unaffected.

### Distribution
- Version 3.1 is published to PyPI as `simcenter-dlml`. The GitHub repository's `releases/latest` continues to point to v2.1.0 for legacy Pelicun (≤3.9) compatibility.

---

## [3.0] - 2026-07-13

This release turns the library into an installable Python package and adds a programmatic API, a command-line interface, and the DLML Explorer web app. The model data itself is unchanged and remains backward compatible.

### Added
- **Installable package:** the library is published to PyPI as `simcenter-dlml` (imported as `dlml`), with the model data bundled inside the wheel — no runtime download.
- **Python API** (`import dlml`) for discovering datasets and reading model parameters, metadata, and input schemas: `list_datasets`, `available_collections`, `get_fragility` / `get_consequence_repair` / `get_loss_repair` / `get_parameters`, `get_metadata`, `get_schema`, `get_file`, and `validate_asset` / `validate_assets`.
- **Command-line interface** (`dlml`): `dlml explorer` launches the web app; `dlml list` and `dlml info` inspect the packaged datasets.
- **DLML Explorer** web app (the `simcenter-dlml[explorer]` extra) for searching, visualizing, and assembling project-specific model collections.
- Test suite, continuous integration (lint, type-check, spell-check, a 3.9/3.12/3.13 × Linux/Windows/macOS test matrix, and a bare-install check), and PyPI trusted publishing.

### Changed
- **Distribution:** the project is now distributed as a pip-installable package in addition to being bundled with Pelicun; the source moved to a `src/dlml` layout with the data under `src/dlml/data`.
- **Self-contained:** the core library, its API, and the documentation build no longer require Pelicun. The package now bundles the CSV-parsing helper (`convert_to_MultiIndex`) and the controlled vocabularies it validates against (the recognized demand types and probability-distribution families), so the core install depends only on pandas, numpy, and jsonschema.

### Distribution
- Version 3.0 is distributed through PyPI as `simcenter-dlml`. The GitHub repository's `releases/latest` continues to point to v2.1.0, so existing Pelicun (≤3.9) installations — which download the model data from the latest GitHub release — keep working unchanged. Subsequent v3.x updates are published to PyPI.

---

## [2.1.0] - 2025-09-11

This release focuses on improving code quality and enhancing usability of Hazus assessments. The changes maintain backward compatibility while providing users with more flexibility in input specification and ensuring the codebase adheres to modern Python best practices.

### Changed
- **Input Validation:** Relaxed validation constraints for seismic and flood assessments to improve usability:
    - Allow HeightClass attribute for seismic structural systems (W1, W2, S3, PC1, MH) that don't require it in Hazus methodology
    - Remove PlanArea field from auto-populated seismic configuration as it's no longer needed
    - Allow RES1 occupancy buildings to have more than 3 stories in flood assessments, aligning with FEMA technical manual interpretation
- **Code Quality:** Comprehensive code formatting and linting improvements using Ruff across the entire codebase:
    - Applied consistent code formatting across 15 Python files
    - Fixed docstring formatting and missing docstring issues
    - Cleaned up import statements and unused code
    - Standardized quote usage and line spacing

### Fixed
- Resolved spelling issues in comments

---

## [2.0.0] - 2025-08-15

This release marks a major milestone for the Damage and Loss Model Library and the beginning of a more frequent and structured release schedule. After more than two years of continuous development, `v2.0.0` introduces a significantly improved data schema, a host of new models, and a documentation system for model discovery.

For a comprehensive overview of the library in its current state, please see our completely revamped README.md.

### Changed
- **BREAKING:** The core data schema for models has been updated and rationalized to better support a wider range of hazards and asset types. Older custom tools that relied on the `v1.0.0` schema will need to be updated.
    - Renamed `loss_repair.csv` files to `consequence_repair.csv` for consistency
    - Updated loss measure units to use ratios instead of percentages for Hazus models
    - Improved data validation and multilinear CDF handling in model generation scripts
- **Code Quality:** Comprehensive code formatting and linting improvements using Ruff, enhancing maintainability and consistency across the entire codebase.

### Added
- **Model Library Expansion:**
    - **Hurricane Models:**
        - Added complete building models for **Hazus Hurricane v5.1** (both original and coupled versions)
        - Added **SimCenter Hurricane Wind Component Library** with comprehensive wind pressure components from peer-reviewed research
    - **Flood Models:**
        - Added complete building models for **Hazus Flood v6.1**
    - **Seismic Infrastructure Models:**
        - Added **Hazus Seismic Power Network** models (v5.1)
        - Added **Hazus Seismic Water Network** models (v6.1) 
        - Added **Hazus Seismic Transportation Network** models (v5.1)
        - Updated **Hazus Seismic Building** models to v6.1 (in addition to existing v5.1)
        - Added **Hazus Seismic Building Subassembly** models (v5.1)
- **Documentation System:**
    - Added an automated documentation website built with Sphinx that allows for easy discovery and exploration of all models in the library
    - Implemented custom Sphinx extensions for automatic generation of model documentation with fragility and consequence curves
    - Added caching system for efficient documentation builds
- **Metadata Enhancements:**
    - Added rich metadata, including clear citations, for every model to ensure proper credit and academic integrity
    - Enhanced model descriptions and parameter documentation
- **Configuration Files:**
    - Added Pelicun configuration files for all model libraries to facilitate integration with the Pelicun damage and loss assessment framework

### Fixed
- Multiple bug fixes in Hazus model generation scripts and data processing
- Corrected fragility parameters and damage state definitions across various model libraries
- Fixed issues with multilinear CDF data validation and processing
- Resolved metadata formatting and JSON structure issues

---

## [1.0.0] - 2023-03-01

### Added
- Initial Release
- FEMA P-58 2nd edition
- Hazus Earthquake Model for Buildings