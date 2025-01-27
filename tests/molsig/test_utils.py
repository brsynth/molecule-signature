import numpy as np
from rdkit import Chem
from rdkit.rdBase import BlockLogs

from molsig.utils import (
    mol_from_smiles,
    mol_filter,
    read_txt,
    write_txt,
    read_csv,
    write_csv,
    read_tsv,
    write_tsv,
)


# =================================================================================================
# Tests for `mol_from_smiles`
# =================================================================================================

def test_mol_from_smiles_valid():
    mol = mol_from_smiles("CCO")
    assert mol is not None
    assert mol.GetNumAtoms() == 3


def test_mol_from_smiles_invalid():
    log_lock = BlockLogs()
    mol = mol_from_smiles("INVALID")
    del log_lock
    assert mol is None


def test_mol_from_smiles_clear_aam():
    mol = mol_from_smiles("[CH3:1][CH2:2][OH:3]", clear_aam=False)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetAtomMapNum() != 0  # Mapping

    mol = mol_from_smiles("[CH3:1][CH2:2][OH:3]", clear_aam=True)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetAtomMapNum() == 0  # No mapping


def test_mol_from_smiles_clear_isotope():
    mol = mol_from_smiles("[13CH3][12CH2][16OH]", clear_isotope=False)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetIsotope() != 0  # Isotope

    mol = mol_from_smiles("[13CH3][12CH2][16OH]", clear_isotope=True)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetIsotope() == 0  # No isotope


def test_mol_from_smiles_reject_conditions():
    assert mol_from_smiles("C.C") is None  # Fragmented
    assert mol_from_smiles("*CCO") is None  # Generic atom


# =================================================================================================
# Tests for `mol_filter`
# =================================================================================================

def test_mol_filter_valid():
    mol = mol_from_smiles("CCO")
    filtered = mol_filter(mol, max_mw=500)
    assert filtered is not None


def test_mol_filter_large_molecule():
    large_molecule = Chem.AddHs(Chem.MolFromSmiles("C" * 100))
    filtered = mol_filter(large_molecule, max_mw=500)
    assert filtered is None


def test_mol_filter_with_radicals():
    mol = Chem.MolFromSmiles("[CH3]")  # Méthyle radicalaire
    filtered = mol_filter(mol, exclude_radical=True)
    assert filtered is None


# =================================================================================================
# Tests for `read_txt` and `write_txt`
# =================================================================================================

def test_read_write_txt(tmp_path):
    test_file = str(tmp_path / "test.txt")
    data = ["line1", "line2", "line3"]
    write_txt(test_file, data)
    read_data = read_txt(test_file)
    assert read_data == data


# =================================================================================================
# Tests for `read_csv` and `write_csv`
# =================================================================================================

def test_read_write_csv(tmp_path):
    test_file = str(tmp_path / "test")
    header = ["col1", "col2", "col3"]
    data = np.array([[1, 2, 3], [4, 5, 6]])
    write_csv(test_file, header, data)
    read_header, read_data = read_csv(test_file)
    assert header == read_header
    assert np.array_equal(data, read_data)


# =================================================================================================
# Tests for `read_tsv` and `write_tsv`
# =================================================================================================

def test_read_write_tsv(tmp_path):
    test_file = str(tmp_path / "test")
    header = ["col1", "col2", "col3"]
    data = np.array([[1, 2, 3], [4, 5, 6]])
    write_tsv(test_file, header, data)
    read_header, read_data = read_tsv(test_file)
    assert header == read_header
    assert np.array_equal(data, read_data)
