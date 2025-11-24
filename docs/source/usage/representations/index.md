# Representations

This section details how to generate various molecular representations using MolBlender.

Choose a representation type below:

::::{grid} 2
:class-container: sd-card
:gutter: 2

:::{grid-item-card} **Sequential**
:link: sequential/index
:link-type: doc

Handle sequence-based representations (SMILES, SELFIES, tokenizers, LMs).
:::

:::{grid-item-card} **Descriptors**
:link: descriptors/index
:link-type: doc

A series of simple physical/chemical properties.
:::

:::{grid-item-card} **Fingerprints**
:link: fingerprints/index
:link-type: doc

Generate 2D fingerprints (MACCS, Morgan, RDKit, CDK, etc.).
:::

:::{grid-item-card} **Graph**
:link: graph/index
:link-type: doc

Create graph-based representations for GNNs.
:::

:::{grid-item-card} **Image**
:link: image/index
:link-type: doc

Generate 2D or 3D image-based (or images) representations.
:::

:::{grid-item-card} **Spatial**
:link: spatial/index
:link-type: doc

Use 3D coordinate information for featurization. Matrix, 3D fingerprints, etc.
:::

:::{grid-item-card} **Temporal/Ensemble**
:link: temporal/index
:link-type: doc

Extract features from molecular dynamics trajectories/conformational ensembles.
:::

:::{grid-item-card} **Protein**
:link: protein/index
:link-type: doc

Work with protein-specific representations (sequences, structures).
:::

:::{grid-item-card} **Protein-ligand**
:link: protein-ligand/index
:link-type: doc

Work with protein-ligand interactions (contact graph, MaSIF, etc.).
:::

:::{grid-item-card} **Utilities**
:link: utils
:link-type: doc

Understand shared utilities for representation handling.
:::

::::

See {doc}`Featurizers <../basic/featurizers>` for general usage details.

See also the {doc}`API Reference <../../api/representations/index>` for technical implementation details.


```{toctree}
:maxdepth: 1
:hidden:

sequential/index
descriptors/index
fingerprints/index
graph/index
image/index
spatial/index
temporal/index
protein/index
protein-ligand/index
utils
```