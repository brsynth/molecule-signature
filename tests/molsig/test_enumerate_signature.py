# -*- coding: utf-8 -*-
"""
Tests for functions in the `signature.enumerate_signature` module.

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
  - Philippe Meyer <philippe.meyer@inrae.fr>
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from molsig.enumerate_signature import (atom_sig_to_root,
                                           atomic_sig_to_smiles,
                                           custom_sort_with_dependent,
                                           enumerate_molecule_from_morgan,
                                           enumerate_molecule_from_signature,
                                           enumerate_signature_from_morgan,
                                           extract_atomic_num,
                                           extract_formal_charge,
                                           get_H_h_x_d_value_regex,
                                           is_counted_subset)
from molsig.SignatureAlphabet import SignatureAlphabet


def test_atom_sig_to_root():
    sa = "[C;H0;h0;D3;X3]-[C;H0;h0;D4;X4:1](-[C;H3;h3;D1;X4])(-[C;H3;h3;D1;X4])-[O;H0;h0;D2;X2] && SINGLE <> [C;H0;h0;D3;X3]-[O;H0;h0;D2;X2:1]-[C;H0;h0;D4;X4] && SINGLE <> [C;H0;h0;D4;X4]-[C;H0;h0;D3;X3:1](-[N;H0;h0;D3;X3])=[O;H0;h0;D1;X1] && SINGLE <> [C;H0;h0;D4;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H0;h0;D4;X4]-[C;H3;h3;D1;X4:1]"
    result_root = atom_sig_to_root(sa)
    expected_root = "[C;H0;h0;D4;X4:1]"
    assert result_root == expected_root


def test_extract_formal_charge():
    sa_root = "[N;H1;h1;D2;X3:1]"
    result_charge = extract_formal_charge(sa_root)
    assert result_charge == 0

    sa_root = "[O;H0;h0;D1;X1;-:1]"
    result_charge = extract_formal_charge(sa_root)
    assert result_charge == -1


def test_extract_atomic_num():
    sa_root = "[C;H3;h3;D1;X4:1]"
    result_atomic_num = extract_atomic_num(sa_root)
    assert result_atomic_num == 6

    sa_root = "[N;H1;h1;D2;X3:1]"
    result_atomic_num = extract_atomic_num(sa_root)
    assert result_atomic_num == 7

    sa_root = "[O;H0;h0;D1;X1;-:1]"
    result_atomic_num = extract_atomic_num(sa_root)
    assert result_atomic_num == 8


def test_get_H_h_x_d_value_regex():
    sa_root = "[C;H3;h3;D1;X4:1]"
    result_H_value, result_h_value, result_d_value, result_x_value = get_H_h_x_d_value_regex(sa_root)
    assert result_H_value == 3
    assert result_h_value == 3
    assert result_d_value == 1
    assert result_x_value == 4

    sa_root = "[n;H1;h0;D2;X3:1]"
    result_H_value, result_h_value, result_d_value, result_x_value = get_H_h_x_d_value_regex(sa_root)
    assert result_H_value == 1
    assert result_h_value == 0
    assert result_d_value == 2
    assert result_x_value == 3


def test_atomic_sig_to_smiles():
    sa = "[O;H2;h2;D0;X2:1] && "
    result_smiles = atomic_sig_to_smiles(sa)
    assert result_smiles == "O"


def test_is_counted_subset():
    # Test case 1: Sublist is a counted subset of mainlist
    assert is_counted_subset([1, 2], [1, 2, 3]) is True

    # Test case 2: Sublist is not a counted subset (insufficient count of an element)
    assert is_counted_subset([1, 1, 2], [1, 2, 3]) is False

    # Test case 3: Sublist is empty
    assert is_counted_subset([], [1, 2, 3]) is True

    # Test case 4: Mainlist is empty, sublist is not
    assert is_counted_subset([1], []) is False


def test_custom_sort_with_dependent():
    # Test case 1: Basic sorting
    primary = [[3, 1], [2, 4]]
    dependents = [[10, 20], [30, 40]]
    sorted_primary, sorted_dependents = custom_sort_with_dependent(primary, dependents)
    assert sorted_primary == [[1, 3], [2, 4]]
    assert sorted_dependents == [[10, 20], [30, 40]]

    # Test case 2: Empty primary and dependent lists
    primary = []
    dependents = []
    sorted_primary, sorted_dependents = custom_sort_with_dependent(primary, dependents)
    assert sorted_primary == []
    assert sorted_dependents == []

    # Test case 3: Single element in primary and dependent lists
    primary = [[5]]
    dependents = [[99]]
    sorted_primary, sorted_dependents = custom_sort_with_dependent(primary, dependents)
    assert sorted_primary == [[5]]
    assert sorted_dependents == [[99]]

    # Test case 4: Handling zeros in the primary list
    primary = [[0, 2], [1, 0]]
    dependents = [[5, 6], [7, 8]]
    sorted_primary, sorted_dependents = custom_sort_with_dependent(primary, dependents)
    assert sorted_primary == [[0, 1], [0, 2]]
    assert sorted_dependents == [[6, 5], [8, 7]]


def test_enumerate_signature_from_morgan():
    smi = "CCO"
    mol = Chem.MolFromSmiles(smi)
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    morgan = fpgen.GetCountFingerprint(mol).ToList()
    Alphabet = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    signatures = set(
        [
            "1057-294 ## [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]",
            "80-1410 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "807-222 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
    )
    Alphabet.fill_from_signatures(signatures, atomic=True, verbose=True)
    result_Ssig, result_bool_partition_threshold_reached, _, _ = enumerate_signature_from_morgan(morgan, Alphabet)
    expected_Ssig = [
        [
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
    ]
    assert result_Ssig == expected_Ssig
    assert result_bool_partition_threshold_reached == False


def test_enumerate_molecule_from_signature():
    sig = [
        "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
    ]
    Alphabet = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    result_S, result_recursion_threshold_reached, _ = enumerate_molecule_from_signature(sig, Alphabet)
    assert result_S
    assert result_recursion_threshold_reached == False


def test_enumerate_molecule_from_morgan():
    smi = "CCO"
    mol = Chem.MolFromSmiles(smi)
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    morgan = fpgen.GetCountFingerprint(mol).ToList()
    Alphabet = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    signatures = set(
        [
            "1057-294 ## [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]",
            "80-1410 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "807-222 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
    )
    Alphabet.fill_from_signatures(signatures, atomic=True, verbose=True)
    result_Ssig, result_Smol, result_Nsig, _, _ = enumerate_molecule_from_morgan(morgan, Alphabet)
    expected_Ssig = [
        [
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
    ]
    expected_Smol = set(["CCO"])
    assert result_Ssig == expected_Ssig
    assert result_Smol == expected_Smol
    assert result_Nsig == 1
