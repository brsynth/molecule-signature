import pytest
from rdkit import Chem
from signature.Signature import AtomSignature, MoleculeSignature


# Fixtures for creating molecules
@pytest.fixture
def benzene():
    return Chem.MolFromSmiles('c1ccccc1')


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles('CCO')


@pytest.fixture
def caffeine():
    return Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')


# Test initialization and outputs
def test_atom_signature(benzene):
    atom = benzene.GetAtomWithIdx(0)  # First carbon in benzene
    atom_sig = AtomSignature(atom, radius=1, use_smarts=False)
    assert atom_sig.sig is not None
    assert atom_sig.sig.startswith("C:[CH:1]:C")


def test_molecule_signature(ethanol):
    mol_sig = MoleculeSignature(ethanol, radius=1, use_smarts=True, nbits=2048)
    assert len(mol_sig) > 0
    assert all(isinstance(sig, AtomSignature) for sig in mol_sig.atom_signatures)


# Performance testing
def test_large_molecule():
    # Assume this SMILES represents a large molecule
    large_mol_smiles = 'C' * 50  # Simplified example
    large_mol = Chem.MolFromSmiles(large_mol_smiles)
    MoleculeSignature(large_mol, radius=2, use_smarts=True, nbits=2048)


# Error handling
def test_invalid_smiles():

    with pytest.raises(ValueError):
        invalid_mol = Chem.MolFromSmiles('CCCCC(P')
        MoleculeSignature(invalid_mol, radius=2)

    with pytest.raises(ValueError):
        invalid_mol = None
        MoleculeSignature(invalid_mol, radius=2)


# Test the deprecated string output
def test_deprecated_string():
    mol = Chem.MolFromSmiles("CCO")
    mol_sig = MoleculeSignature(mol, radius=1, use_smarts=False, nbits=0)
    deprecated_string = mol_sig.as_deprecated_string(morgan=True, neighbors=False)

    assert isinstance(deprecated_string, str)
    assert "C" in deprecated_string, "The string should contain atom representations"


# Running the tests
if __name__ == "__main__":
    pytest.main(["-v"])
