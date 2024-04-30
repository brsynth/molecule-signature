import pytest
from rdkit import Chem
from signature.Signature import AtomSignature, MoleculeSignature


# Test the AtomSignature initialization and basic attributes
def test_atom_signature_initialization():
    mol = Chem.MolFromSmiles("CCO")  # Ethanol as a simple test case
    atom = mol.GetAtomWithIdx(0)  # Get the first carbon atom

    atom_sig = AtomSignature(atom, radius=2, use_smarts=True)

    assert atom_sig.radius == 2
    assert atom_sig.use_smarts is True
    assert atom_sig.sig != "", "The signature should not be empty"


# Test the molecule signature with various radii
@pytest.mark.parametrize(
    "radius,expected_length",
    [(0, 3), (1, 3), (2, 3)],  # Testing with radius 0  # Testing with radius 1  # Testing with full molecule radius
)
def test_molecule_signature(radius, expected_length):
    mol = Chem.MolFromSmiles("CCO")
    mol_sig = MoleculeSignature(mol, radius=radius, use_smarts=False, nbits=2048)

    assert (
        len(mol_sig.atom_signatures) == expected_length
    ), "The number of atom signatures should match the expected length"


# Test handling of invalid inputs
def test_invalid_inputs():
    with pytest.raises(ValueError):
        # This should raise an error as mol is None
        mol = None
        MoleculeSignature(mol, radius=2)


# Test the deprecated string output
def test_deprecated_string():
    mol = Chem.MolFromSmiles("CCO")
    mol_sig = MoleculeSignature(mol, radius=1, use_smarts=False, nbits=0)
    deprecated_string = mol_sig.as_deprecated_string(morgan=True, neighbors=False)

    assert isinstance(deprecated_string, str)
    assert "C" in deprecated_string, "The string should contain atom representations"


# Running the tests
if __name__ == "__main__":
    pytest.main()
