# Signature

Signature-based enumeration of molecules from morgan fingerprints.

## Table of Contents

- [Signature](#signature)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [From conda package](#from-conda-package)
    - [From source code](#from-source-code)
  - [Usage](#usage)
    - [Build a signature from SMILES](#build-a-signature-from-smiles)
  - [Build an alphabet](#build-an-alphabet)
  - [Citation](#citation)
  - [License](#license)

## Installation

### From conda package

Installation using conda is the easiest way to get started. First, install Conda
and then install the package from the conda-forge channel.

1. **Install Conda:**
   Download the installer for your operating system from the [Conda Installation
   page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
   Follow the instructions on the page to install Conda. For example, on
   Windows, you would download the installer and run it. On macOS and Linux, you
   might use a command like:

    ```bash
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ```

    Follow the prompts on the installer to complete the installation.

2. **Install signature from conda-forge:**

    ```bash
    conda install -c conda-forge signature
    ```

### From source code

One can also install the tool from the source code. This method is useful for
development purposes.

1. **Install dependencies:**

    ```bash
    conda env create -f environment.yml
    ```

2. **Add the signature to conda:**

    ```bash
    conda activate sig
    pip install -e .  # From the root of the repository
    ```

3. **Add development dependencies:**

    ```bash
    pip install -e .[dev]
    ```

## Usage

### Build a signature from SMILES

Here a simple example showing how to build a signature from a SMILES string. For
more example, one can refer to the `notebooks/signature-basic` notebook.

```python
from rdkit import Chem
from signature.Signature import MoleculeSignature

mol = Chem.MolFromSmiles("CCO")
mol_sig = MoleculeSignature(mol)
mol_sig.to_list()
# [
#  '80-1410 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]',
#  '807-222 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]',
#  '1057-294 ## [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]'
# ]
```

## Build an alphabet

...

## Citation

...

## License

This project is licensed under the MIT License. See the LICENSE file for details.
