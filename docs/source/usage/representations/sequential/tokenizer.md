
# Tokenizers

This guide describes how to use molecular tokenizer featurizers in **MolBlender**.

Tokenization converts SMILES strings or molecules into sequences of token IDs, enabling the application of natural language processing (NLP) techniques to chemical structures. MolBlender provides a unified interface for various tokenization approaches through DeepChem's tokenizers.

## Available Tokenizers

MolBlender currently supports the following tokenizer types:

| Featurizer Name                   | Description                                       | Return Type      |
| --------------------------------- | ------------------------------------------------- | ---------------- |
| `DeepChem-SmilesTokenizer`        | BERT-based SMILES tokenizer with vocabulary file  | Integer IDs      |
| `DeepChem-BasicSmilesTokenizer`   | Regex-based SMILES tokenizer                      | String tokens    |
| `DeepChem-HuggingFaceFeaturizer`  | Wrapper for any Hugging Face tokenizer            | Integer IDs      |
| `DeepChem-GroverAtomTokenizer`    | Atom-level vocabulary-based tokenization          | Integer IDs      |
| `DeepChem-GroverBondTokenizer`    | Bond-level vocabulary-based tokenization          | Integer IDs      |

## Quick Example

```python
# 1. Import MolBlender
import molblender as mbl

# 2. Set up a basic tokenizer
featurizer = mbl.get_featurizer('DeepChem-BasicSmilesTokenizer')

# 3. Tokenize a SMILES string
tokens = featurizer.featurize("CC(=O)C")
print(tokens)  # Array of token strings: ['C', 'C', '(', '=', 'O', ')', 'C']

# 4. Using a vocabulary-based tokenizer (with numerical IDs)
# First, you need a vocabulary file
vocab_file = "/path/to/vocab.txt"  
featurizer = mbl.get_featurizer('DeepChem-SmilesTokenizer', vocab_file=vocab_file)
token_ids = featurizer.featurize("CC(=O)C")
print(token_ids)  # Array of integer IDs: [12, 12, 17, 22, 19, 18, 12]
```

## Notes

* Most tokenizers accept **SMILES strings** as input, but also handle RDKit Mol objects.
* For batch input, pass a list of SMILES strings or molecules.
* If tokenization fails, it will be logged and return None.
* Some tokenizers (like `DeepChem-BasicSmilesTokenizer`) return string tokens, while others return integer IDs.
* Vocabulary-based tokenizers require a vocabulary file.

## When to Use Different Tokenizers

| Tokenizer Type             | Suitable Use Cases                                           |
| -------------------------- | ------------------------------------------------------------ |
| SmilesTokenizer            | Deep learning models requiring numerical token IDs           |
| BasicSmilesTokenizer       | Simple tokenization, or when HuggingFace is not installed    |
| HuggingFaceFeaturizer      | Using pre-trained tokenizers from Hugging Face               |
| GroverAtomTokenizer        | Atom-level tokenization for graph-based models               |
| GroverBondTokenizer        | Bond-level tokenization for graph-based models               |

## Advanced Usage

### Using Hugging Face Tokenizers

For more advanced tokenization options, you can use Hugging Face tokenizers:

```python
from transformers import AutoTokenizer
from molblender.representations import get_featurizer

# Load a pre-trained SMILES tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_4k")

# Create the wrapper featurizer
featurizer = get_featurizer('DeepChem-HuggingFaceFeaturizer', 
                            tokenizer=hf_tokenizer)

# Use it like any other featurizer
token_ids = featurizer.featurize("CC(=O)C")
```

### Customizing BasicSmilesTokenizer

You can customize the tokenization pattern:

```python
custom_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
featurizer = get_featurizer('DeepChem-BasicSmilesTokenizer', 
                            regex_pattern=custom_pattern)
```

### Training Your Own Vocabulary

For Grover tokenizers, you can train your own vocabulary using DeepChem's vocabulary builders:

```python
import deepchem as dc
from deepchem.feat.vocabulary_builders.grover_vocab import GroverAtomVocabularyBuilder

# Create and train2 a vocabulary builder
vocab_builder = GroverAtomVocabularyBuilder()
dataset = dc.data.NumpyDataset(X=[['CC(=O)C'], ['c1ccccc1']])
vocab_builder.build(dataset)

# Save the vocabulary for future use
vocab_file = "atom_vocab.json"
vocab_builder.save(vocab_file)

# Use with MolBlender
atom_tokenizer = get_featurizer('DeepChem-GroverAtomTokenizer', 
                                vocab_file=vocab_file)
```

## API Reference

For full class and method documentation, see the the API documentation page.

---

Happy tokenizing!

```{toctree}
:maxdepth: 1
:hidden:
```