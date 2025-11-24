# Introduction

MolBlender is a Python toolkit designed for generating diverse molecular representations suitable for various machine learning tasks. It aims to provide a unified, user-friendly, and extensible framework for handling multi-modal molecular data.

## Motivation

Molecular representation learning is crucial for drug discovery, materials science, and chemical research. Numerous excellent packages like RDKit, DeepChem, and Chemprop exist, each with its strengths in specific representation types or modeling approaches. However, integrating features from different modalities (e.g., 2D topology, 3D conformation, sequence information, dynamic properties) often requires significant effort and boilerplate code.

MolBlender addresses this by:

* **Providing a unified API:** Accessing different types of fingerprints, graph embeddings, descriptors, and other representations follows a consistent pattern.
* **Supporting multi-modal inputs:** Handling SMILES, molecular files (SDF, PDB), and potentially sequence or trajectory data through a clear interface.
* **Leveraging existing libraries:** Building upon the robust functionalities of RDKit, DeepChem, CDK (via wrappers), and potentially others.
* **Focusing on good software engineering:** Employing clear abstractions, a central registry, parallel processing utilities, and standardized logging.
* **Facilitating dataset management:** Offering tools to load, process, featurize, analyze, and split molecular datasets.

## Core Concepts

* **Featurizers:** Classes responsible for calculating specific molecular representations.
* **Registry:** A central mechanism mapping string keys to featurizer implementations.
* **Input Types:** Declarations of the data format a featurizer expects.
* **Data Module:** Tools for managing molecular data collections.

This documentation will guide you through installation, basic usage, available representations, and the API details.