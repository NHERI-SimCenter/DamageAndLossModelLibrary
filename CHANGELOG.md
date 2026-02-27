# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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