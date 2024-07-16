## Unreleased

### Fix

- **AtomSignature**: always use tuple for morgans
- **MoleculeSignature**: wrong attribute name

## 1.3.0 (2024-07-16)

### Feat

- **MoleculeSignature**: allow generation of molecule signature with multiple morgan bit per atom
- **AtomSignature**: enable multiple morgan bits

## 1.2.0 (2024-07-15)

### Feat

- **AtomSignature**: stop computing the with neighbors signature on __init__

### Fix

- **AtomSignature**: force no implicit Hs
- **AtomSignature**: correctly handle negative charges

## 1.1.0 (2024-07-15)

### Feat

- **Signature**: retrieve the signature with neighbors from the without one
- **AtomSignature**: export an AtomSignature as a RDKit Mol

### Fix

- **AtomSignature**: use hashtag syntax for hydrogens
- **enumerate_molecule_from_signature**: smiles cleaning before timeout condition
- **signature_alphabet**: suppress outdated arguments
- **enumerate_signature**: suppress outdated arguments
- **MoleculeSignature**: wrong attribute in test equality

### Refactor

- **AtomSignature**: atom_signature as a class function
- **Signature**: sweep deprecated functions
- **Signature**: update default value for atom_signature
- **reduced_fingerprint**: int type of morgan vectors
- **enumerate_signature_from_morgan**: simplification of some loops

## 1.0.0 (2024-07-11)

### Feat

- **MoleculeSignature**: enable MoleculeSignature init from list and from string
- **MoleculeSignature**: complete rewrite of as_str now named to_string
- **AtomSignature**: add from_string constructor
- **AtomSignature**: improve to_string export
- **utils**: add functions to sanitize and filter molecular structures
- **signature_alphabet**: add merge and compatibility functions
- **3.drug_application.ipynb**: draft of a notebook to enumerate drug molecules and create drug databases
- **enumerate_signature**: add new timeouts
- **2.reverse_engineer**: add outputs of timeout with enumeration

### Fix

- **AtomSignature**: fix attribute error
- **enumerate_signature_from_morgan**: bug when not enough fragments have been found
- **enumerate_signature**: second correction of the [nH] pb in the enum sig to mol
- **atomic_num_charge**: correction of formal charge extraction bug from smarts
- **signature_alphabet**: add missing parameters in print_out and load functions
- **solve_partitions**: bug when clean_local_solutions gives a zero solution
- **update_constraint_matrices**: bug correction when AS is zero length v2
- **update_constraint_matrices**: bug correction when AS is zero length
- **enumerate_molecule_from_signature**: add boundary_bonds argument
- **signature_alphabet**: add boundary_bounds and use_smarts in save method

### Refactor

- **MoleculeSignature**: constructor accept None mol
- **MoleculeSignature**: rewrite to_list to stick with AtomSignature behaviour
- **MoleculeSignature**: update repr
- **MoleculeSignature**: clean code
- **MoleculeSignature**: rename attributes
- **MoleculeSignature**: remove unused neighbor argument
- **AtomSignature**: change repr method to keep on track with attribute names
- **AtomSignature**: edit class attribute names
- **MoleculeSignature**: remove dispensable attributes
- **AtomSignature**: rename variables
- **AtomSignature**: remove dispensable attributes
- **AtomSignature**: update atom signature comparison methods
- **Signature**: remvove legacy implementation for computing signature
- **Signature**: use smarts syntex as default
- **enumerate_signature**: suppress useless arguments
- **signature_old**: move the last usefull functions from signature_old to other scripts
- **enumerate_utils**: suppress useless signature_neighbor
- **enumerate_utils**: move test functions from notebooks to scripts

## 0.9.0 (2024-06-27)

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
