## Unreleased

### Feat

- **MoleculeSignature**: add MoleculeSignature equality comparison

### Fix

- **AtomSignature**: AtomSignature equality comparison was always false
- **Signature**: disable SMARTS canonization
- **Signature**: fix canon_smarts unexpected argument

### Refactor

- **Signature**: log status of SMARTS canonization at loading

## 0.4.0 (2024-06-19)

### Feat

- **Signature**: enable new signature computation from MoleculeSignature objects
- **Signature**: enable new signature generation from AtomSignature objects
- **Signature**: improve atom to smarts conversion
- **Signature**: additional options for AtomSignature
- **Signature**: sort atom signatures
- **Signature**: add few tests of combinations
- **Signature**: add few examples
- **Signature**: add as_deprecated_str methods
- **Signature**: new interface for interacting with signatures

### Fix

- **canon_smarts**: add file of primitives
- **canon_smarts**: work around for unknown primitives
- **Signature**: stop crashing for H and HH molecules
- **Signature**: stop complaining about molecule having a max radius of 1
- **Signature**: additional args could be passed to MolToSmiles
- **Signature**: fix exception on molecule made of a single atom
- **Signature**: fix deprecated fingerprint generation

### Refactor

- **Signature**: update examples
- **Signature**: sweep code
- update imports
- **signature**: rename deprecated file
- **signature**: use available rdkit objects for BondType
- **.gitignore**: ignore .coverage file

## 0.3.0 (2024-04-25)

### Feat

- **signature_alphabet**: enable fill from signature
- **signature_alphabet**: enable fill for existing alphabet

### Fix

- update import

### Refactor

- **tox.ini**: add ignore case to flake8
- delete jupyter checkpoints
- sort imports, add docstrings and solve_partition tests
- **src**: add code
- hello world
