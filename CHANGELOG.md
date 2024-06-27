## Unreleased

### Feat

- enumeration with smiles radius 2 and boundaries

### Fix

- **Signature**: remove rdcanon calls
- correction of the [nH] pb in the enum sig to mol
- suppress src. from local imports
- **Signature**: double check neighbor output is required

## 0.8.0 (2024-06-26)

### Feat

- **Signature**: revisit how to build SMARTS to be sure it will be canonic

### Fix

- **Signature**: put back boundary_bonds on / off option
- **.gitignore**: **/.ipynb_checkpoints/
- **signature_alphabet**: integration of the SMARTS and SMILES sig in the computation of the alphabet
- add of root=False when signature is used in the enumeration
- **enumerate_signature**: mol None bug in enumerate_molecule_from_signature
- **Signature**: fix map_root=False
- **Signature**: remove duplicated arg

## 0.7.0 (2024-06-21)

### Feat

- **MoleculeSignature**: use multi-key sorting for the output

### Fix

- **Signature**: homogenize MolToSmiles options
- **MoleculeSignature**: fix sorting
- **as_deprecated_string**: fix args order

## 0.6.0 (2024-06-21)

### Feat

- **as_deprecated_string**: add on/off switch for root-style as_deprecated_string output
- **MoleculeSignature**: add MoleculeSignature equality comparison
- integration of the new SMILES & SMARTS sig to the enum algo

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
