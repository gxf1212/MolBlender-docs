# Configuration

```{toctree}
:maxdepth: 1
:hidden:
```

MolBlender automatically manages model caching for pre-trained models used by advanced featurizers (PLMs, UniMol, etc.). Models are downloaded once and reused from local cache.

## Cache Priority

MolBlender determines cache locations in this order:

1. **Environment variables** (highest priority)
2. **API settings** via {func}`~molblender.config.set_cache_dir`  
3. **Default paths** under `~/.cache/molblender/`

## Environment Variables

Set these before importing MolBlender:

```bash
# PyTorch Hub models (ESM, CARP)
export TORCH_HOME=/my/custom/torch/cache

# Hugging Face models (ProtT5, Ankh, PepBERT)  
export HF_HOME=/my/custom/hf/cache

# Optional: Use mirror for faster downloads
export HF_ENDPOINT=https://hf-mirror.com
```

## Default Paths

If environment variables aren't set, MolBlender uses:
- `~/.cache/molblender/torch_hub/` for PyTorch Hub
- `~/.cache/molblender/huggingface_hub/` for Hugging Face

## Programmatic Control

```python
from molblender.config import set_cache_dir, get_cache_dir

# Set custom cache directories
set_cache_dir("torch", "/data/models/torch")
set_cache_dir("hf", "/data/models/huggingface")

# Check current settings
print(f"PyTorch cache: {get_cache_dir('torch')}")
print(f"HuggingFace cache: {get_cache_dir('hf')}")
# Output: PyTorch cache: /data/models/torch
# Output: HuggingFace cache: /data/models/huggingface
```

## Verifying Settings

MolBlender logs effective paths on import:

```python
import molblender
# INFO: [MolBlender Settings] Effective TORCH_HOME: /home/user/.cache/molblender/torch_hub
# INFO: [MolBlender Settings] Effective HF_HOME: /home/user/.cache/molblender/huggingface_hub
# INFO: [MolBlender Settings] Using HF_ENDPOINT (mirror): https://hf-mirror.com (if set)
```

## Model Loading Example

```python
from molblender.representations.protein.sequence.plm import ProteinLanguageModelFeaturizer

# Models are automatically downloaded to configured cache
featurizer = ProteinLanguageModelFeaturizer(
    model_name="Rostlab/prot_t5_xl_half_uniref50",
    model_type="t5",
    batch_size=8
)

# First run downloads model (~892MB for ProtT5-XL)
# Subsequent runs load from cache instantly
embeddings = featurizer.featurize(["MKTAYIAKQRQISFVKSHFSRQ"])
print(f"Embedding shape: {embeddings.shape}")
# Output: Embedding shape: (1, 1024)
```

## Disk Space Requirements

Common model sizes:
- ESM-2 (650M params): ~2.5GB
- ProtT5-XL: ~900MB  
- Ankh Large: ~1.5GB
- UniMol: ~300MB

## Offline Usage

Once models are cached, MolBlender works offline:

```python
# After initial download, this works without internet
from molblender.representations.spatial.unimol import UniMolFeaturizer

unimol = UniMolFeaturizer()
# Loads from local cache at TORCH_HOME or HF_HOME
```

## API Reference

- {func}`~molblender.config.set_cache_dir` - Set cache directory
- {func}`~molblender.config.get_cache_dir` - Get current cache path
- {attr}`~molblender.config.EFFECTIVE_TORCH_HOME` - Active PyTorch cache
- {attr}`~molblender.config.EFFECTIVE_HF_HOME` - Active HuggingFace cache

## Related Links

- [PyTorch Hub Documentation](https://pytorch.org/docs/stable/hub.html)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/huggingface_hub/)