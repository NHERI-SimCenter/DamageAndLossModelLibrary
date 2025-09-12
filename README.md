
# Damage and Loss Model Library (DLML)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A curated, open-source repository of standardized model parameters and metadata for quantifying the impact of natural hazard events on the built environment.

---

### About This Library

The Damage and Loss Model Library (DLML) is a project from the NHERI SimCenter designed to address a critical gap in natural hazards engineering: the lack of a centralized, standardized, and easy-to-use repository for damage and loss models. This library provides the essential data—model parameters, descriptive metadata, and configuration files—that power natural hazard risk assessment simulations.

This `v2.1.0` release continues the evolution of the project, building on the improved data schema, extensive model collection, and documentation system introduced in v2.0.0, with enhanced code quality and improved usability for Hazus assessments.

**Key Features:**
* **Standardized Data Schema:** A robust yet flexible schema for organizing models by hazard, asset type, and resolution, making it easy to use data and supporting new contributions.
* **Rich Metadata & Clear Citations:** Every model is paired with detailed metadata and a clear citation to the original source.
* **A Broad & Inclusive Collection:** We feature a growing collection of models, from large-scale industry standards like **FEMA P-58** and **FEMA Hazus** to peer-reviewed models from the broader research community. Our goal is to provide a home for all high-quality damage and loss models.
* **Automated Documentation:** The metadata and parameters are used to automatically generate a user-friendly documentation website, complete with model descriptions, parameter tables, and plots of fragility and consequence functions.

---

### Three Ways to Use This Library

There are three primary ways to interact with the DLML, depending on your needs.

#### 1. Discover and Explore Models

The best way to get started is by exploring our documentation website. It provides a searchable, lightweight interface to discover all available models, view their parameters, and understand their assumptions.

[**➡️ Visit the Documentation Website**](https://nheri-simcenter.github.io/DamageAndLossModelLibrary/)

#### 2. Perform Calculations with Pelicun

For performing damage and loss calculations, we recommend using **Pelicun**, the SimCenter's open-source simulation engine or SimCenter's **PBE** and **R2D** desktop applications that utilize Pelicun in the background. Pelicun is designed to seamlessly use the models from this library as its inputs. This library is bundled with Pelicun, so all models are available automatically after installation.

[**➡️ Learn More About Pelicun**](https://github.com/NHERI-SimCenter/Pelicun)

#### 3. Work with the Raw Data

You can also work directly with the raw data files in this repository. The library is structured in a clear hierarchy of `hazard/asset_type/resolution/methodology`. Within each methodology, **model parameters are stored in standardized CSV files** (e.g., for fragility, loss, or consequences) and metadata in corresponding JSON files.

To get a local copy, clone the repository:
```bash
git clone [https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary.git](https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary.git)
````

-----

### Contributing

We welcome contributions of new and improved models from the researchers and practitioners. While we are in the process of creating a formal contributor's guide, the best way to start is by opening a GitHub Issue to discuss the model data you would like to add.

Our vision is to enable a simple workflow where contributors can submit pull requests with data that conforms to our schema. We appreciate your patience and collaboration as we work towards this goal.

-----

### License & Acknowledgments

This library is distributed under the BSD 3-Clause license. See `LICENSE` for more information.

This material is based upon work supported by the National Science Foundation under Grants No. 1612843 2131111. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

-----

### Contact

For questions or support, please open an issue on the GitHub repository.

Adam Zsarnóczay, NHERI SimCenter, Stanford University, adamzs@stanford.edu
