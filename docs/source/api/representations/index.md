# Representations Module

At the heart of MolBlender is a powerful system for handling molecular representations across multiple modalities — chemical, structural, sequential, visual, and more.

This section introduces the different kinds of representations available in the toolkit, from traditional fingerprints (like RDKit or CDK) to deep learning-based encodings and even image-based representations. Each modality comes with its own interface, conversion utilities, and compatible downstream tools.

You'll also find helper functions and utilities that allow you to bridge different representations, chain transformations, and prepare features for modeling tasks.

```{toctree}
:maxdepth: 2

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
