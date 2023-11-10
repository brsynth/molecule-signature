# signature

Signature based molecule enumeration from morgan fingerprints.

## Install

```sh
conda env create -f environment.yml
conda activate chemigen
pip install -e .
```

## Use

Buid signature from SMILES, includes 2048-width morgan fingerprints (radius 2) and atomic neighborhood:

```python
from signature.signature_alphabet import SignatureAlphabet, SignatureFromSmiles

input_smi = "CC=C(N=CC(N)O)C(C)=O"
Alphabet = SignatureAlphabet(neighbors=True, radius=2, nBits=2048)
sig, mol, smi = SignatureFromSmiles(smi, Alphabet, verbose=False)
```
