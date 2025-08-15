:notoc:

===============================
 Damage and Loss Model Library
===============================

Welcome to the Damage and Loss Model Library (DLML), a curated, open-source repository of standardized model parameters and metadata for quantifying the impact of natural hazard events on the built environment.

This library is a project from the NHERI SimCenter designed to address a critical gap in natural hazards engineering: the lack of a centralized, standardized, and easy-to-use repository for damage and loss models. The DLML provides the essential data—model parameters, descriptive metadata, and configuration files—that power natural hazard risk assessment simulations.

**Key Features:**

* **Standardized Data Schema:** A robust yet flexible schema for organizing models by hazard, asset type, and resolution
* **Rich Metadata & Clear Citations:** Every model is paired with detailed metadata and clear citations to original sources
* **Broad & Inclusive Collection:** Growing collection from industry standards like FEMA P-58 and FEMA Hazus to peer-reviewed research models
* **Automated Documentation:** User-friendly documentation with model descriptions, parameter tables, and fragility/consequence function plots

**Model Coverage:**

The library includes comprehensive model collections for **seismic**, **hurricane**, and **flood** hazards, covering buildings, infrastructure networks, and individual components from sources including FEMA P-58, FEMA Hazus, and peer-reviewed academic research.

.. toctree::
   :caption: Documentation
   :maxdepth: 4

   about/about.rst
   release_notes/index.rst
   dl_doc/damage/index.rst
   dl_doc/repair/index.rst
