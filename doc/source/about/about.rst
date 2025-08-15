.. _about:

*******
 About
*******

Overview
========

The Damage and Loss Model Library (DLML) is a project from the NHERI SimCenter designed to address a critical gap in natural hazards engineering: the lack of a centralized, standardized, and easy-to-use repository for damage and loss models. This library provides the essential data—model parameters, descriptive metadata, and configuration files—that power natural hazard risk assessment simulations.

This curated, open-source repository contains standardized model parameters and metadata for quantifying the impact of natural hazard events on the built environment. The library serves as a comprehensive resource for researchers, practitioners, and engineers working in natural hazards risk assessment.

Key Features
============

**Standardized Data Schema**
    A robust yet flexible schema for organizing models by hazard, asset type, and resolution, making it easy to use data and supporting new contributions.

**Rich Metadata & Clear Citations**
    Every model is paired with detailed metadata and a clear citation to the original source, ensuring proper academic integrity and traceability.

**Broad & Inclusive Collection**
    A growing collection of models, from large-scale industry standards like **FEMA P-58** and **FEMA Hazus** to peer-reviewed models from the broader research community. The goal is to provide a home for all high-quality damage and loss models.

**Automated Documentation**
    The metadata and parameters are used to automatically generate a user-friendly documentation website, complete with model descriptions, parameter tables, and plots of fragility and consequence functions.

Model Coverage
==============

The library includes comprehensive model collections across multiple hazards:

**Seismic Models**
    - FEMA P-58 2nd Edition building components
    - Hazus Seismic Building models (v5.1 and v6.1)
    - Hazus Seismic Infrastructure models (Power, Water, Transportation Networks)
    - Hazus Seismic Building Subassembly models

**Hurricane Models**
    - Hazus Hurricane building models (v5.1)
    - SimCenter Hurricane Wind Component Library with peer-reviewed research components

**Flood Models**
    - Hazus Flood building models (v6.1)

Usage
=====

There are three primary ways to interact with the DLML:

1. **Discover and Explore Models**: Use the documentation website to browse and understand available models
2. **Perform Calculations with Pelicun**: Integrate with the SimCenter's Pelicun simulation engine for damage and loss calculations
3. **Work with Raw Data**: Access the standardized CSV and JSON files directly from the repository

Data Organization
=================

The library is structured in a clear hierarchy of ``hazard/asset_type/resolution/methodology``. Within each methodology:

- **Model parameters** are stored in standardized CSV files (e.g., for fragility, loss, or consequences)
- **Metadata** is provided in corresponding JSON files
- **Configuration files** are included for integration with the Pelicun framework

License & Acknowledgments
=========================

This library is distributed under the BSD 3-Clause license.

This material is based upon work supported by the National Science Foundation under Grants No. 1612843 and 2131111. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
