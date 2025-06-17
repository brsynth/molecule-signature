## 2.1.1 (2025-06-17)

### Fix

- **solutions_of_P**: non constant timeout

## 2.1.0 (2025-02-06)

### Feat

- enable entrypoint for command line usage

### Fix

- hack rm file for windows
- parsing entry

### Refactor

- **cmd**: shorten arguments name
- **notebooks**: refactor imports

## 2.0.1 (2025-01-29)

### Refactor

- **utils.py**: trigger cicd

## 2.0.0 (2025-01-29)

### Feat

- **enumerate_utils**: new function to compare ecfp with ecfp of a smiles
- **enumerate_signature**: new function to handle automatically enum ecfp to mols
- **fill_from_signatures**: possibility to fill the alphabet from precomputed atomic or molecular signatures
- **generate_stereoisomers**: limit number of stereoisomers
- **enumerate_molecule_from_signature**: limit number of stereoisomers
- **enumerate_signature**: implicit Hs count in SMARTS
- **MoleculeSignature**: use stereo by default
- **Signature**: implicit Hs count in SMARTS
- **signature_alphabet**: add stereochemistry
- **enumerate_utils**: add utilities about stereochemistry and isotopic information
- **custom_sort_with_dependent**: new function to sorted lists
- **enumerate_molecule_from_signature**: chirality addition
- **Signature**: includes stereo in neighbors
- **utils::mol_filter**: additional filters based on dative bonds and radicals
- **utils::mol_from_smiles**: change option from 'keep_stereo' to 'clear_stereo'
- **Signature**: add 'use_stereo' argument to switch on stereo in Signatures
- **enumerate_molecule_from_signature**: possibility to save the molecule in SVG at each reconstruction step

### Fix

- **utils**: put back missing import
- **enumerate_molecule_from_signature**: default value of repeat
- **enumerate_signature_from_morgan**: single atom morgan
- **SignatureAlphabet**: try except on signature computation
- **test_sol_ECFP**: use_stereo and test mol is none
- **load_alphabet**: use_stereo argument
- **flat_molecule_copy**: pass though SMILES to update implicit Hs count
- **flat_molecule_copy**: use RemoveHs to get rid of explicit Hs
- **mol_from_smiles**: remove superfluous Hs
- **MoleculeSignature**: update morgan bits index when flattening molecules
- **AtomSignature**: morgans bit always considered as a tuple
- **generate_stereoisomers**: high number of possible stereoisomers
- **mol_from_smiles**: keep stereo by default
- **Signature**: stereo for FP but fragments remain flat
- **enumerate_signature_from_morgan**: sort fragments by morgan bits
- **MolecularGraph**: simplification of the smiles computation
- **MolecularGraph**: better atom initialization
- **Signature**: only cis/trans stereo when the environment is big enough
- **Signature**: fix old attribute name 'morgan'
- **enumerate_signature_from_morgan**: sorting of the morgan bits
- **enumerate_signature_from_morgan**: single atom atomic signature
- **enumerate_molecule_from_signature**: single atom atomic signature
- **enumerate_signature**: formal charge, explicit H, implicit H, degree, valency
- **atomic_num_charge**: new sig version

### Refactor

- change the location of test_signature_sorted_array
- **enumerate_utils**: move a function + docstrings
- signature to molsig and metanetx alphabet as bonus only
- molsig to signature and signature_alphabet to SignatureAlphabet
- **enumerate_utils**: signature to molsig and signature_alphabet to SignatureAlphabet
- **enumerate_signature**: from signature to from molsig
- **SignatureAlphabet**: from signature to from molsig
- **test_SignatureAlphabet**: change script name
- **SignatureAlphabet**: change script name
- **tests**: rename module name
- rename module as molsig
- **Signature**: change 'rooted_smiles' arts to 'rooted‘
- **Signature**: remove deprecated signature str
- **Signature**: remove use_smarts arg
- **notebooks**: sweep
- **Signature**: remove deprecated function
- **signature**: sweep old code
- **signature_alphabet**: suppress useless functions
- **signature_sorted_array**: suppress useless parameter
- **enumerate_signature**: rename functions
- **enumerate_utils**: rename functions
- **Signature**: sweep unused method
- **Signature**: clean unused code
- **datasets**: remove old data
- **enumerate_utils**: black the code and sort packages
- **enumerate_signature**: black the code and sort packages
- **solve_partitions**: change timeout vocabulary to threshold
- **enumerate_signature**: change timeout vocabulary to threshold
- **enumerate_signature_from_morgan**: total ct in function
- **enumerate_signature**: suppress useless max_nbr_solution argument
- **utils**: suppress useless vector-dict conversion functions
- **signature_alphabet**: suppress useless vector-dict conversion functions
- **signature_alphabet**: suppress map_root parameter
- **enumerate_signature_from_morgan**: alphabet dictionary structure changed to a set
- **signature_alphabet**: alphabet dictionary structure changed to a set
- **enumerate_signature_from_morgan**: change useless dictionary structures to lists
- **enumerate_utils**: suppress useless MIN set
- **enumerate_signature_from_morgan**: suppress useless MIN set
- **signature_alphabet**: remove unnecessary or outdated functions
- **SignatureAlphabet**: remove unuseful parameters
- **signature_alphabet**: remove unnecessary or outdated functions
- **enumerate_utils**: remove unnecessary or outdated functions
- **enumerate_utils**: unification of smiles notation
- **enumerate_utils**: remove unnecessary or outdated functions
- **Signature**: morgan_bit renamed to morgan bits and does not accept interger anymore
- **Signature**: remove unused import
- **Signature**: remove 'all_bits' option as we always use all bits

## 1.6.0 (2024-11-06)

### Feat

- **Signature**: append cis/trans bonds
- **Signature**: append chiral tags

### Fix

- **Signature**: debug messages
- **solve_partitions**: part_C zero length

## 1.5.0 (2024-10-01)

### Feat

- **Signature**: add morgans argument for to_string methods
- **solve_partitions**: clean solutions early by max values
- **1.create_alphabets**: update with new sig form
- **2.reverse_engineer**: update enum pipeline for sig without neigh and full ecfp
- **enumerate_signature**: signature without neighbors and change of separators
- **signature_alphabet**: change of separators
- **solution_of_one_group**: new method to find a candidate min
- **enumerate_signature**: alphabet without neighbors and change of separators
- **enumerate_utils**: change of separators

### Fix

- **solve_partitions**: part_C zero length
- **get_constraint_matrices**: using unique bond signatures is enough
- **enumerate_signature_from_morgan**: different eq diophantine solutions can give the same signature
- **sol_max**: smaller sol max when non constant partition
- **solve_partitions**: negative max_nbr_partition
- **partitions_on_non_constant**: bad local bound when small max_nbr_partition
- **solve_partitions**: stop when empty dictionary
- **enumerate_signature_from_morgan**: take account of the counted bits in fragments
- **restrict_sol_by_C**: verbose badly placed
- **solve_partitions**: bug on timeout flag

### Refactor

- **sol_max**: simplify computation of shape
- **solve_partitions**: satisfied constraint lines save as set instead of list
- **sanitize_molecule**: suppress unuseful comment
- **signature_sorted_array**: change sig in string to list
- **enumerate_signature**: change sig in string to list
- **solve_partitions**: add a function to refactor the restriction on C
- **solve_partitions**: restriction on C only when necessary
- **solution_of_one_group**: ensure merged_parts is sorted
- **partitions_on_non_constant**: shorter way to compute final partitions

## 1.4.0 (2024-08-02)

### Feat

- **Signature**: function to set canonic aam for fragment
- **Signature**: add function to extract features from a SMARTS
- **Signature**: add function to extract features from a SMARTS

### Fix

- **AtomSignature**: Ensure canonic SMARTS whatever the input
- **AtomSignature**: stop trying to fix MolFromSmarts
- **atom_to_smarts**: when possible direct extraction from SMARTS of query features
- **MoleculeSignature**: vector of morgan bits

### Refactor

- **Signature**: sweep unused code
- **atom_signature**: better handle radius cuts
- **atom_to_smarts**: SMARTS feature separator as a variable

## 1.3.2 (2024-07-29)

### Feat

- **signature_alphabet**: alphabet without neighbors
- **solve_partitions**: new solving method for diophantine systems coming from full ecfp
- **enumerate_signature**: reduced ecfp changed to full ecfp

### Fix

- **AtomSignature**: wrong radius for generation of neighbors
- **enumerate_signature**: import correction

## 1.3.1 (2024-07-16)

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
