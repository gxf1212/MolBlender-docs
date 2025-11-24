# Fingerprints

Molecular fingerprints are one of the most widely used representations in cheminformatics and machine learning for molecules. They encode molecular structures into fixed-length vectors, often binary or count-based, that can be compared efficiently for similarity or fed into machine learning models.

MolBlender supports several widely-used fingerprinting toolkits, each with its own algorithm and applications. This guide introduces how to generate and use fingerprints from each supported backend. Whether you're working on molecular similarity search, QSAR modeling, or feature engineering for deep learning, these tools provide a unified and modular way to get started.

## Available Fingerprint Backends

::::{grid} 2
:class-container: sd-text-center
:gutter: 3

:::{grid-item-card} **RDKit Fingerprints**
:link: rdkit
:link-type: doc

Versatile and popular open-source cheminformatics fingerprints.
:::

:::{grid-item-card} **CDK Fingerprints**
:link: cdk
:link-type: doc

A Java-based toolkit offering a wide range of fingerprint types via CDK-pywrapper.
:::

:::{grid-item-card} **DeepChem Fingerprints**
:link: deepchem
:link-type: doc

Wrappers for fingerprint implementations within the DeepChem library.
:::

:::{grid-item-card} **Datamol Fingerprints**
:link: datamol
:link-type: doc

Efficient RDKit-based fingerprint generation provided by the Datamol library.
:::

::::

---

For detailed class and function references, see the full API documentation:
{doc}`/api/representations/fingerprints/index`.

```{toctree}
:maxdepth: 1
:hidden:

rdkit
cdk
deepchem
datamol
```