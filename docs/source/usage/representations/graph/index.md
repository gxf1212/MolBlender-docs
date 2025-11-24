# Graph Representations

Graph-based representations are essential for leveraging the power of Graph Neural Networks (GNNs) in molecular machine learning. In MolBlender, molecules can be converted into graph structures where atoms are nodes and bonds are edges, enriched with chemical features.

This guide will walk you through the core concepts of graph construction and how to use built-in utilities to work with graph representations. Future content will include support for:

* Atom and bond featurization
* Converting RDKit molecules into graphs
* Integrating with libraries such as DeepChem's `GraphConvFeaturizer`
* Exporting to PyTorch Geometric or DGL formats

These representations are useful for property prediction, molecular classification, and generative modeling.

For detailed API documentation, see:{doc}`/api/representations/index`

```{toctree}
:maxdepth: 1
:hidden:

deepchem
```
