.. automodule:: molblender.representations.protein.sequence.plm
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: BaseFeaturizer

Protein Language Model (PLM) Featurizers
=========================================

This section details the featurizers available for generating representations from protein sequences, primarily focusing on Protein Language Models (PLMs).

Core PLM Functionality
-----------------------

.. py:class:: molblender.representations.protein.sequence.plm.PLMBaseFeaturizer
   :noindex:

   The abstract base class for all Protein Language Model featurizers within MolBlender.
   It handles common logic such as model loading, device placement, and the featurization workflow.
   Subclasses implement model-specific loading and embedding logic. For inherited methods like ``featurize``, please see :py:class:`~molblender.representations.utils.base.BaseFeaturizer`.


Available PLM Featurizers
-------------------------

The following specific PLM featurizers are available. They inherit from :py:class:`~molblender.representations.protein.sequence.plm.PLMBaseFeaturizer`.

.. py:class:: molblender.representations.protein.sequence.plm.ESM2Featurizer
   :noindex:

   Featurizer for ESM-2 models (e.g., ``esm2_t33_650M_UR50D``, ``esm2_t6_8M_UR50D``).
   Uses the `fair-esm` library via `torch.hub`.
   See {doc}`/usage/representations/protein/sequence/plm` for registered names.

.. py:class:: molblender.representations.protein.sequence.plm.CARPFeaturizer
   :noindex:

   Featurizer for CARP models (e.g., ``carp_640M``).
   Uses the `sequence-models` library.
   See {doc}`/usage/representations/protein/sequence/plm` for registered names.

.. py:class:: molblender.representations.protein.sequence.plm.ProtT5Featurizer
   :noindex:

   Featurizer for ProtT5 models (e.g., ``Rostlab/prot_t5_xl_half_uniref50``).
   Uses the Hugging Face `transformers` library.
   See {doc}`/usage/representations/protein/sequence/plm` for registered names.

.. py:class:: molblender.representations.protein.sequence.plm.AnkhFeaturizer
   :noindex:

   Featurizer for Ankh models (e.g., ``ElnaggarLab/ankh-large``).
   Uses the Hugging Face `transformers` library.
   See {doc}`/usage/representations/protein/sequence/plm` for registered names.

.. py:class:: molblender.representations.protein.sequence.plm.PepBERTFeaturizer
   :noindex:

   Featurizer for PepBERT models, typically loaded from a local snapshot.
   Uses Hugging Face `tokenizers` and a custom PyTorch model structure.
   See {doc}`/usage/representations/protein/sequence/plm` for registered names.


Registry Access Functions for Protein Featurizers
-------------------------------------------------

These functions are used to access and list protein-specific featurizers. They operate on a separate registry from the general/small-molecule featurizers.

.. py:function:: molblender.representations.get_protein_featurizer(name_or_cls, **user_kwargs)
   :noindex:

   Retrieves and instantiates a protein-specific featurizer by its registered name or class.
   (Full documentation inherited from the registry module).

.. py:function:: molblender.representations.list_available_protein_featurizers() -> List[str]
   :noindex:

   Returns a sorted list of registered protein featurizer configuration names.
   (Full documentation inherited from the registry module).

.. py:function:: molblender.representations.get_protein_featurizer_info(name: str) -> Optional[Dict[str, Any]]
   :noindex:

   Retrieves information (class, default arguments, metadata) about a registered protein featurizer.
   (Full documentation inherited from the registry module).


Related Base Classes
--------------------
.. py:class:: molblender.representations.utils.base.BaseFeaturizer
   :noindex:
   
   The main abstract base class for all featurizers in MolBlender, which `PLMBaseFeaturizer` inherits from.

.. py:class:: molblender.representations.protein.base.BaseProteinFeaturizer
   :noindex:

   (If defined) A potential future base class specifically for protein featurizers, possibly inheriting from `BaseFeaturizer`.

