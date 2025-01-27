# -*- coding: utf-8 -*-
"""
Tests for functions in the `signature.enumerate_utils` module.

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
  - Philippe Meyer <philippe.meyer@inrae.fr>
"""

import numpy as np
from rdkit import Chem

from signature.enumerate_utils import (bond_matrices, bond_signature_occurence,
                                       constraint_matrix,
                                       generate_stereoisomers,
                                       get_constraint_matrices,
                                       get_first_stereoisomer, remove_isotopes,
                                       signature_bond_type,
                                       smiles_ecfp_same_ecfp_or_not,
                                       smiles_same_ecfp_or_not,
                                       update_constraint_matrices)


def test_bond_signature_occurence():
    bsig = "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]|SINGLE|[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]"
    asig = "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]"
    result_asig0, result_as1, result_as2, result_occ1, result_occ2 = bond_signature_occurence(bsig, asig)
    expected_asig0 = "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]"
    expected_as1 = "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]"
    expected_as2 = "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]"
    expected_occ1 = 1
    expected_occ2 = 0
    assert result_asig0 == expected_asig0
    assert result_as1 == expected_as1
    assert result_as2 == expected_as2
    assert result_occ1 == expected_occ1
    assert result_occ2 == expected_occ2


def test_constraint_matrix():
    AS = np.array(
        [
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        ]
    )
    BS = np.array(
        [
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]|SINGLE|[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]|SINGLE|[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        ]
    )
    deg = np.array([2, 1, 1])
    expected_C = np.array([[-1.0, 0.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0, 0.0], [0.0, -1.0, -1.0, 2.0, -2.0]])
    result_C = constraint_matrix(AS, BS, deg)
    assert np.array_equal(result_C, expected_C)


def test_bond_matrices():
    AS = np.array(
        [
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        ]
    )
    NAS = np.array([1, 1, 1])
    deg = np.array([2, 1, 1])
    result_B, result_BS = bond_matrices(AS, NAS, deg)
    expected_B = np.array(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0],
        ]
    )
    expected_BS = np.array(
        [
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]|SINGLE|[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]|SINGLE|[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        ]
    )
    assert np.array_equal(result_B, expected_B)
    assert np.array_equal(result_BS, expected_BS)


if 1 == 1:

    def test_get_constraint_matrices():
        sig = [
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
        result_AS, result_NAS, result_deg, result_A, result_B, result_C = get_constraint_matrices(sig, unique=False)
        expected_AS = np.array(
            [
                "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
                "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
                "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
            ]
        )
        expected_NAS = np.array([1, 1, 1])
        expected_deg = np.array([1, 1, 2])
        expected_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        expected_B = np.array(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 1],
            ]
        )
        expected_C = np.array([[1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 2.0, -2.0]])
        assert np.array_equal(result_AS, expected_AS)
        assert np.array_equal(result_NAS, expected_NAS)
        assert np.array_equal(result_deg, expected_deg)
        assert np.array_equal(result_A, expected_A)
        assert np.array_equal(result_B, expected_B)
        assert np.array_equal(result_C, expected_C)


def test_update_constraint_matrices():
    AS = np.array(
        [
            "[I;H0;h0;D3;X3]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H0;h0;D2;X2]-[I;H0;h0;D3;X3:1](-[O;H1;h1;D1;X2])-[c;H0;h0;D3;X3]",
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H0;h0;D3;X3]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[Si;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [N;H2;h2;D1;X3]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H1;h1;D3;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[n;H0;h0;D3;X3]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[P;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H1;h1;D2;X3]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H0;h0;D2;X2]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [N;H0;h0;D3;X3]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [N;H0;h0;D2;X2]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H0;h0;D4;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H0;h0;D2;X2]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [N;H1;h1;D2;X3]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H1;h0;D2;X3;-]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[c;H0;h0;D3;X3]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H2;h2;D2;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H0;h0;D3;X3;-]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [Cl;H0;h0;D1;X1]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[S;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4:1]-[S;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H1;h1;D2;X3]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Pb;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H1;h0;D3;X4;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[S;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[n;H0;h0;D3;X3]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H0;h0;D4;X4]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H0;h0;D4;X4;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Hg;H0;h0;D1;X1;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H1;h1;D2;X3]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H2;h0;D2;X4;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Cl;H0;h0;D1;X1]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [As;H0;h0;D4;X4;+]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H0;h0;D3;X3;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H3;h0;D1;X4;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[S;H0;h0;D3;X3;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[c;H0;h0;D3;X3]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[S;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H0;h0;D2;X2]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[P;H0;h0;D3;X3]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Si;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H0;h0;D3;X3]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Zn;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H0;h0;D3;X3]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H1;h1;D3;X4]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Li;H0;h0;D1;X1]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[P;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[P;H0;h0;D4;X4;-]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [Br;H0;h0;D1;X1]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Hg;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[Mg;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [As;H1;h0;D2;X3]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[P;H2;h2;D1;X3]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H1;h0;D2;X3;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H0;h0;D2;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[I;H0;h0;D1;X1]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[n;H0;h0;D3;X3;+]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[C;H3;h3;D1;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[S;H0;h0;D4;X4]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H0;h0;D1;X1;-]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[S;H0;h0;D3;X3]",
            "[P;H0;h0;D1;X1]#[P;H0;h0;D1;X1:1] && TRIPLE <> [P;H0;h0;D1;X1]#[P;H0;h0;D1;X1:1]",
            "[O;H1;h1;D1;X2]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[O;H1;h1;D1;X2:1]",
            "[I;H0;h0;D1;X1]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[I;H0;h0;D1;X1:1]",
            "[B;H2;h2;D1;X3]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[B;H2;h2;D1;X3:1]",
            "[F;H0;h0;D1;X1]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[F;H0;h0;D1;X1:1]",
            "[S;H1;h1;D1;X2]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[S;H1;h1;D1;X2:1]",
            "[As;H2;h0;D1;X3]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[As;H2;h0;D1;X3:1]",
            "[Cl;H0;h0;D1;X1]-[O;H1;h1;D1;X2:1] && SINGLE <> [O;H1;h1;D1;X2]-[Cl;H0;h0;D1;X1:1]",
            "[F;H0;h0;D1;X1]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[F;H0;h0;D1;X1:1]",
            "[C;H3;h3;D1;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H3;h3;D1;X4:1]",
            "[As;H2;h0;D1;X3]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[As;H2;h0;D1;X3:1]",
            "[P;H2;h2;D1;X3]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[P;H2;h2;D1;X3:1]",
            "[Cl;H0;h0;D1;X1]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[Cl;H0;h0;D1;X1:1]",
            "[O;H0;h0;D1;X1;-]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[O;H0;h0;D1;X1;-:1]",
            "[O;H1;h1;D1;X2]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[O;H1;h1;D1;X2:1]",
            "[B;H2;h2;D1;X3]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[B;H2;h2;D1;X3:1]",
            "[S;H1;h1;D1;X2]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[S;H1;h1;D1;X2:1]",
            "[Hg;H0;h0;D1;X1]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[Hg;H0;h0;D1;X1:1]",
            "[N;H3;h0;D1;X4;+]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[N;H3;h0;D1;X4;+:1]",
            "[Hg;H0;h0;D1;X1;+]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[Hg;H0;h0;D1;X1;+:1]",
            "[Br;H0;h0;D1;X1]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[Br;H0;h0;D1;X1:1]",
        ]
    )
    IDX = [
        [80, 807],
        [80, 1410],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [222, 807],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [294, 1057],
        [807],
        [807],
        [807],
        [807],
        [807],
        [807],
        [807],
        [807],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
        [1057],
    ]
    MAX = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    deg = np.array(
        [
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    result_AS, result_IDX, result_MAX, result_deg, result_C = update_constraint_matrices(AS, IDX, MAX, deg)
    expected_AS = np.array(
        [
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        ]
    )
    expected_IDX = [[80, 1410], [222, 807], [294, 1057]]
    expected_MAX = np.array([1, 1, 1])
    expected_deg = np.array([2, 1, 1])
    expected_C = np.array([[-1.0, 0.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0, 0.0], [0.0, -1.0, -1.0, 2.0, -2.0]])
    assert np.array_equal(result_AS, expected_AS)
    assert np.array_equal(result_IDX, expected_IDX)
    assert np.array_equal(result_MAX, expected_MAX)
    assert np.array_equal(result_deg, expected_deg)
    assert np.array_equal(result_C, expected_C)


def test_smiles_same_ecfp_or_not():
    class Alphabet:
        radius = 2
        nBits = 2048
        use_stereo = True

    smis_1 = ["CCO", "N", "C1=CC=CC=C1"]
    smis_2 = [
        "CN1CCN2CCN(C)CCN(CC1)C1=C(Cl)C(=O)C2=C(Cl)C1=O",
        "CN1CCN(C)CCN2CCN(CC1)C1=C(Cl)C(=O)C2=C(Cl)C1=O",
        "CN1CCN(C2=C(Cl)C(=O)C(N3CCN(C)CC3)=C(Cl)C2=O)CC1",
    ]
    assert not smiles_same_ecfp_or_not(smis_1, Alphabet)
    assert smiles_same_ecfp_or_not(smis_2, Alphabet)


def test_smiles_ecfp_same_ecfp_or_not():
    class Alphabet:
        radius = 2
        nBits = 256
        use_stereo = True

    morgan = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert smiles_ecfp_same_ecfp_or_not(morgan, "CCO", Alphabet)
    assert not smiles_ecfp_same_ecfp_or_not([0] * 256, "CCO", Alphabet)


def test_signature_bond_type():
    # Test cases for different bond types
    test_cases = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
        "UNSPECIFIED": Chem.BondType.UNSPECIFIED,
    }

    # Test each bond type
    for bt_str, expected_bt in test_cases.items():
        result = signature_bond_type(bt_str)
        assert result == expected_bt, f"Expected {expected_bt} for bond type {bt_str}, but got {result}"

    # Test invalid bond type
    invalid_bt = "INVALID"
    try:
        signature_bond_type(invalid_bt)
        assert False, "Expected an exception for invalid bond type"
    except KeyError:
        pass  # Expected behavior


def test_generate_stereoisomers():
    # Test case 1: A molecule with stereocenters
    smi_stereo = "NC(O)C(=O)O"
    expected_stereo = ["N[C@H](O)C(=O)O", "N[C@@H](O)C(=O)O"]
    result_stereo = generate_stereoisomers(smi_stereo)
    assert result_stereo == expected_stereo

    # Test case 2: A molecule with no stereocenters
    smi_no_stereo = "CCO"  # Ethanol (no stereocenters)
    expected_no_stereo = ["CCO"]
    result_no_stereo = generate_stereoisomers(smi_no_stereo)
    assert result_no_stereo == expected_no_stereo

    # Test case 3: A molecule with a very high max_nb_stereoisomers limit
    smi_high_limit = "C(C(Cl)(Br)F)C(C(Cl)(Br)F)C(Cl)(Br)F"
    max_nb_stereoisomers_high = 1000
    result_high_limit = generate_stereoisomers(smi_high_limit, max_nb_stereoisomers_high)
    assert len(result_high_limit) <= max_nb_stereoisomers_high


def test_get_first_stereoisomer():
    # Test case 1: A molecule with stereocenters
    smi_stereo = "NC(O)C(=O)O"
    expected_stereo = "N[C@H](O)C(=O)O"
    result_stereo = get_first_stereoisomer(smi_stereo)
    assert result_stereo == expected_stereo

    # Test case 2: A molecule with no stereocenters
    smi_no_stereo = "CCO"  # Ethanol (no stereocenters)
    expected_no_stereo = "CCO"
    result_no_stereo = get_first_stereoisomer(smi_no_stereo)
    assert result_no_stereo == expected_no_stereo


def test_remove_isotopes():
    # Test case 1: SMILES with isotopic information
    smi_with_isotopes = "[13C]CO"
    expected_no_isotopes = "[C]CO"  # Isotopic labels removed
    result_no_isotopes = remove_isotopes(smi_with_isotopes)
    assert result_no_isotopes == expected_no_isotopes

    # Test case 2: SMILES without isotopic information
    smi_no_isotopes = "CCO"  # Ethanol
    expected_no_change = "CCO"
    result_no_change = remove_isotopes(smi_no_isotopes)
    assert result_no_change == expected_no_change
