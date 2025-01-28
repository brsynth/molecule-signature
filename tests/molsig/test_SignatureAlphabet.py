# -*- coding: utf-8 -*-
"""
Tests for functions in the `signature.signature_alphabet` module.

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
  - Philippe Meyer <philippe.meyer@inrae.fr>
"""

import numpy as np

from molsig.SignatureAlphabet import (SignatureAlphabet,
                                      compatible_alphabets,
                                      merge_alphabets,
                                      signature_sorted_array)


def test_compatible_alphabets():
    Alphabet_1 = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    Alphabet_2 = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    results_compatibility = compatible_alphabets(Alphabet_1, Alphabet_2)
    assert results_compatibility == True

    Alphabet_1 = SignatureAlphabet(radius=3, nBits=1024, use_stereo=False)
    Alphabet_2 = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    results_compatibility = compatible_alphabets(Alphabet_1, Alphabet_2)
    assert results_compatibility == False


def test_merge_alphabets():
    Alphabet_1 = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    signatures_1 = set(
        [
            "1057-294 ## [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]",
            "80-1410 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "807-222 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
    )
    Alphabet_1.fill_from_signatures(signatures_1, atomic=True, verbose=True)
    Alphabet_2 = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    signatures_2 = set(
        [
            "1057-294 ## [N;H2;h2;D1;X3]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]",
            "1171-981 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[N;H2;h2;D1;X3:1]",
            "80-789 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H2;h2;D1;X3]",
        ]
    )
    Alphabet_2.fill_from_signatures(signatures_2, atomic=True, verbose=True)
    result_Alphabet_3 = merge_alphabets(Alphabet_1, Alphabet_2)
    expected_Alphabet_3 = SignatureAlphabet(radius=2, nBits=2048, use_stereo=True)
    expected_dict = set(
        [
            "1057-294 ## [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]",
            "80-1410 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "807-222 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
            "1057-294 ## [N;H2;h2;D1;X3]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]",
            "1171-981 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[N;H2;h2;D1;X3:1]",
            "80-789 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[N;H2;h2;D1;X3]",
        ]
    )
    assert compatible_alphabets(result_Alphabet_3, expected_Alphabet_3)
    assert result_Alphabet_3.Dict == expected_dict


def test_signature_sorted_array():
    LAS = [
        "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
        "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
    ]
    result_AS, result_NAS, result_deg = signature_sorted_array(LAS, unique=False)
    expected_AS = np.array(
        [
            "[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1] && SINGLE <> [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]",
            "[C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2] && SINGLE <> [C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1] && SINGLE <> [C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]",
        ]
    )
    expected_NAS = np.array([1, 1, 1])
    expected_deg = np.array([1, 1, 2])
    assert np.array_equal(result_AS, expected_AS)
    assert np.array_equal(result_NAS, expected_NAS)
    assert np.array_equal(result_deg, expected_deg)
