# -*- coding: utf-8 -*-
"""
Tests for the 'Signature' class.

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
  - Philippe Meyer <philippe.meyer@inrae.fr>
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from signature.Signature import AtomSignature, MoleculeSignature


@pytest.fixture
def benzene():
    return Chem.MolFromSmiles('c1ccccc1')


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles('CCO')


@pytest.fixture
def caffeine():
    return Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')


@pytest.fixture
def adrenaline():
    return Chem.MolFromSmiles("CNC[C@H](O)c1ccc(O)c(O)c1")


@pytest.fixture
def pos_charged_molecule():
    return Chem.MolFromSmiles("C[N+](C)(C)CCO")


@pytest.fixture
def neg_charged_molecule():
    return Chem.MolFromSmiles("CC(=O)[O-]")


def test_atom_signature_basic(benzene):
    # Test basic atom signature
    atom = benzene.GetAtomWithIdx(0)  # Get a carbon atom

    atom_sig = AtomSignature(atom, radius=2)
    assert atom_sig.root is not None, "Root signature should not be None"
    assert isinstance(atom_sig.root, str), "Root signature should be a string"
    assert atom_sig.morgans is None, "Morgan bits should be None by default"


def test_atom_signature_neighbors(benzene):
    # Test atom signature with neighbors
    atom = benzene.GetAtomWithIdx(0)  # Get the carbon atom

    atom_sig = AtomSignature(atom, radius=2)
    atom_sig.post_compute_neighbors()
    neighbors = atom_sig.neighbors
    assert isinstance(neighbors, list), "Neighbors should be a list"
    assert len(neighbors) > 0, "Neighbors should not be empty"


def test_atom_signature_none_atom():
    # Test None atom
    atom_sig = AtomSignature(None, radius=2)
    assert atom_sig.root is None, "Root signature should be None"


def test_atom_signature_invalid_atom():
    # Test invalid atom
    with pytest.raises(AssertionError):
        AtomSignature("invalid", radius=2)


def test_molecule_signature_basic(benzene):
    # Test a simple molecule (benzene)
    sig = MoleculeSignature(benzene, radius=2)
    assert len(sig.atoms) == benzene.GetNumAtoms(), "Number of signatures should match number of atoms"


def test_molecule_signature_large_radius(ethanol):
    # Test molecule signature with a large radius
    sig = MoleculeSignature(ethanol, radius=10)
    assert len(sig.atoms) == ethanol.GetNumAtoms(), "Number of signatures should match number of atoms"


def test_molecule_signature_stereochemistry(adrenaline):
    # Test molecule with stereochemistry
    AllChem.AssignStereochemistry(adrenaline, cleanIt=True, force=True)

    sig = MoleculeSignature(adrenaline, radius=2, use_stereo=True)
    assert len(sig.atoms) == adrenaline.GetNumAtoms(), "Number of signatures should match number of atoms"


def test_molecule_signature_invalid_molecule():
    # Test invalid molecule
    with pytest.raises(AssertionError):
        MoleculeSignature("invalid", radius=2)


def test_molecule_signature_empty_molecule():
    # Test an empty molecule
    mol = Chem.MolFromSmiles("")

    with pytest.raises(AssertionError):
        MoleculeSignature(mol, radius=2)


def test_molecule_signature_repr(ethanol):
    # Test molecule signature representation
    sig = MoleculeSignature(ethanol, radius=2)
    sig_repr = repr(sig)

    assert isinstance(sig_repr, str), "Signature representation should be a string"
    assert len(sig_repr) > 0, "Signature representation should not be empty"


def test_molecule_signature_to_string(ethanol):
    # Test molecule signature string representation
    sig = MoleculeSignature(ethanol, radius=2)
    sig_str = sig.to_string()

    assert isinstance(sig_str, str), "Signature string representation should be a string"
    assert len(sig_str) > 0, "Signature string representation should not be empty"


def test_molecule_signature_to_list(ethanol):
    # Test molecule signature list representation
    sig = MoleculeSignature(ethanol, radius=2)
    sig_list = sig.to_list()

    assert isinstance(sig_list, list), "Signature list representation should be a list"
    assert all(isinstance(s, str) for s in sig_list), "All elements in the list should be strings"


def test_molecule_neighbors(ethanol):
    # Test molecule signature neighbors
    sig = MoleculeSignature(ethanol, radius=2)
    sig.post_compute_neighbors()
    neighbors = sig.neighbors

    assert isinstance(neighbors, list), "Neighbors should be a list"
    assert len(neighbors) == ethanol.GetNumAtoms(), "Number of neighbors should match number of atoms"


def test_molecule_signature_eq(ethanol, caffeine):
    # Test molecule signature equality
    sig1 = MoleculeSignature(ethanol, radius=2)
    sig2 = MoleculeSignature(ethanol, radius=2)
    sig3 = MoleculeSignature(caffeine, radius=3)

    assert sig1 == sig2, "Molecule signatures should be equal"
    assert sig1 != sig3, "Molecule signatures should not be equal"


def test_molecule_signature_from_string(caffeine):
    # Test molecule signature from string
    sig = MoleculeSignature(caffeine, radius=2)
    sig_str = sig.to_string()

    new_sig = MoleculeSignature.from_string(sig_str)
    assert sig == new_sig, "Molecule signatures should be equal"


def test_molecule_signature_from_string_with_neighbors(caffeine):
    # Test molecule signature from string with neighbors
    sig = MoleculeSignature(caffeine, radius=2)
    sig.post_compute_neighbors()
    sig_str = sig.to_string()

    new_sig = MoleculeSignature.from_string(sig_str)
    new_sig.post_compute_neighbors()
    assert sig == new_sig, "Molecule signatures should be equal"
    assert sig.neighbors == new_sig.neighbors, "Neighbors should be equal"


def test_molecule_signature_charged_molecules(pos_charged_molecule, neg_charged_molecule):
    # Test molecule signature with charged molecule
    pos_sig = MoleculeSignature(pos_charged_molecule, radius=2)
    assert len(pos_sig.atoms) == pos_charged_molecule.GetNumAtoms(), "Number of signatures should match number of atoms"
    assert "+" in pos_sig.to_string(), "Positive charge should be present in the signature string"

    neg_sig = MoleculeSignature(neg_charged_molecule, radius=2)
    assert len(neg_sig.atoms) == neg_charged_molecule.GetNumAtoms(), "Number of signatures should match number of atoms"
    assert "-" in neg_sig.to_string(), "Negative charge should be present in the signature string"


if __name__ == "__main__":
    pytest.main()
