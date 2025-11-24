# docs/create_placeholders.py
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Type, Union, Dict, Tuple
# Define the base source directory
SOURCE_DIR = Path("source")

# Define the expected file structure (relative to SOURCE_DIR)
# Use tuples: (filename, type, title_if_rst, optional_automodule_target)
# type can be 'md' or 'rst'
EXPECTED_FILES = [
    ("index.rst", 'rst', None, None), # Should exist
    ("conf.py", 'py', None, None), # Should exist
    ("introduction.md", 'md', None, None),
    ("installation.md", 'md', None, None),
    ("quickstart.md", 'md', None, None),
    ("contributing.md", 'md', None, None), # Will be moved/created under development
    ("references.md", 'md', None, None), # Added based on warning

    # --- Usage Section ---
    ("usage/index.md", 'md', None, None),
    ("usage/basics.md", 'md', None, None), # Added based on previous structure
    ("usage/data.md", 'md', None, None),
    ("usage/models.md", 'md', None, None),
    ("usage/representations/index.md", 'md', None, None),
    ("usage/representations/fingerprints.md", 'md', None, None),
    ("usage/representations/fingerprints/rdkit.md", 'md', None, None), # Should exist
    ("usage/representations/fingerprints/cdk.md", 'md', None, None),
    ("usage/representations/fingerprints/deepchem.md", 'md', None, None),
    # ("usage/representations/fingerprints/datamol.md", 'md', None, None), # Example
    ("usage/representations/graph.md", 'md', None, None),
    ("usage/representations/image.md", 'md', None, None),
    ("usage/representations/protein/index.md", 'md', None, None),
    ("usage/representations/protein/sequence.md", 'md', None, None),
    ("usage/representations/sequential/index.md", 'md', None, None),
    ("usage/representations/sequential/language_model.md", 'md', None, None),
    ("usage/representations/sequential/tokenizer.md", 'md', None, None),
    ("usage/representations/spatial.md", 'md', None, None),
    ("usage/representations/temporal.md", 'md', None, None),
    ("usage/representations/topological.md", 'md', None, None),
    ("usage/representations/utils.md", 'md', None, None),
    ("usage/dataset_handling.md", 'md', None, None), # Added based on previous structure

    # --- API Section ---
    ("api/index.rst", 'rst', "API Reference", None),
    ("api/data.rst", 'rst', "Data Handling API (`molblender.data`)", "molblender.data"),
    ("api/models.rst", 'rst', "Models API (`molblender.models`)", "molblender.models"),
    ("api/representations/index.rst", 'rst', "Representations API (`molblender.representations`)", "molblender.representations"),
    ("api/representations/fingerprints/index.rst", 'rst', "Fingerprints API (`...fingerprints`)", "molblender.representations.fingerprints"),
    ("api/representations/fingerprints/rdkit.rst", 'rst', "RDKit Fingerprints API (`...rdkit`)", "molblender.representations.fingerprints.rdkit"),
    ("api/representations/fingerprints/cdk.rst", 'rst', "CDK Fingerprints API (`...cdk`)", "molblender.representations.fingerprints.cdk"),
    ("api/representations/fingerprints/deepchem.rst", 'rst', "DeepChem Fingerprints API (`...deepchem`)", "molblender.representations.fingerprints.deepchem"),
    # ("api/representations/fingerprints/datamol.rst", 'rst', "Datamol Fingerprints API (`...datamol`)", "molblender.representations.fingerprints.datamol"), # Example
    ("api/representations/graph.rst", 'rst', "Graph Representations API (`...graph`)", "molblender.representations.graph"),
    ("api/representations/image.rst", 'rst', "Image Representations API (`...image`)", "molblender.representations.image"),
    ("api/representations/protein/index.rst", 'rst', "Protein Representations API (`...protein`)", "molblender.representations.protein"),
    ("api/representations/protein/sequence.rst", 'rst', "Protein Sequence API (`...sequence`)", "molblender.representations.protein.sequence"),
    ("api/representations/sequential/index.rst", 'rst', "Sequential Representations API (`...sequential`)", "molblender.representations.sequential"),
    ("api/representations/sequential/language_model.rst", 'rst', "Language Model API (`...language_model`)", "molblender.representations.sequential.language_model"),
    ("api/representations/sequential/tokenizer.rst", 'rst', "Tokenizer API (`...tokenizer`)", "molblender.representations.sequential.tokenizer"),
    ("api/representations/spatial.rst", 'rst', "Spatial Representations API (`...spatial`)", "molblender.representations.spatial"),
    ("api/representations/temporal.rst", 'rst', "Temporal Representations API (`...temporal`)", "molblender.representations.temporal"),
    ("api/representations/topological.rst", 'rst', "Topological Representations API (`...topological`)", "molblender.representations.topological"),
    ("api/representations/utils.rst", 'rst', "Representation Utilities API (`...utils`)", "molblender.representations.utils"),

    # --- Development Section ---
    ("development/index.rst", 'rst', "Development Guide", None),
    ("development/contributing.md", 'md', None, None), # Should exist
    ("development/setup_dev_env.md", 'md', None, None), # Should exist
    ("development/testing.md", 'md', None, None), # Should exist
    ("development/add_featurizer.md", 'md', None, None), # Should exist
]

def create_placeholder(filepath: Path, filetype: str, title: Optional[str], automodule_target: Optional[str]):
    """Creates a placeholder file with basic content."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            if filetype == 'rst':
                if title:
                    title_underline = "#" * len(title)
                    f.write(f"{title_underline}\n")
                    f.write(f"{title}\n")
                    f.write(f"{title_underline}\n\n")
                else:
                    # Add a default title if none provided for RST to avoid warning
                    default_title = filepath.stem.replace('_', ' ').title()
                    title_underline = "#" * len(default_title)
                    f.write(f"{title_underline}\n")
                    f.write(f"{default_title}\n") # Use filename as title
                    f.write(f"{title_underline}\n\n")
                    print(f"  WARNING: Added default title to {filepath.name}")

                if automodule_target:
                    f.write(f".. automodule:: {automodule_target}\n")
                    f.write(f"   :members:\n")
                    f.write(f"   :undoc-members:\n")
                    f.write(f"   :show-inheritance:\n")
                else:
                     f.write(".. Add content or toctree here\n")

            elif filetype == 'md':
                if title:
                    f.write(f"# {title}\n\n")
                else:
                     default_title = filepath.stem.replace('_', ' ').title()
                     f.write(f"# {default_title}\n\n") # Use filename as title
                f.write("(Content coming soon...)\n")
            elif filetype == 'py':
                 f.write(f"# Placeholder for {filepath.name}\n")

        print(f"  Created: {filepath}")
    except Exception as e:
        print(f"  ERROR creating {filepath}: {e}")

# --- Main Script ---
if __name__ == "__main__":
    print(f"Checking documentation structure in: {SOURCE_DIR.resolve()}")
    created_count = 0
    skipped_count = 0

    # Ensure base source dir exists
    SOURCE_DIR.mkdir(exist_ok=True)

    for rel_path_str, filetype, title, automodule_target in EXPECTED_FILES:
        filepath = SOURCE_DIR / rel_path_str

        # Handle moving contributing.md first if it exists at top level
        old_contrib_path = SOURCE_DIR / "contributing.md"
        new_contrib_path = SOURCE_DIR / "development" / "contributing.md"
        if rel_path_str == "development/contributing.md" and old_contrib_path.exists() and not new_contrib_path.exists():
             print(f"Moving contributing.md to development/")
             new_contrib_path.parent.mkdir(parents=True, exist_ok=True)
             old_contrib_path.rename(new_contrib_path)
             skipped_count += 1 # Count as skipped for creation check below
             continue # Skip further processing for this entry now

        if filepath.exists():
            # print(f"  Exists: {filepath}")
            skipped_count += 1
        else:
             if filetype in ['rst', 'md', 'py']:
                 create_placeholder(filepath, filetype, title, automodule_target)
                 created_count += 1
             else:
                  print(f"  Skipping unknown file type '{filetype}' for: {filepath}")


    print(f"\nCheck complete. Created {created_count} placeholder files, {skipped_count} files already existed.")
    if created_count > 0:
         print("Please review the created placeholder files and add content.")
    print("Remember to update toctree directives in relevant index.rst files if structure changed.")


