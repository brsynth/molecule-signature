# =================================================================================================
# This library compute save and load Alphabets of atomic signatures associated with Morgan bits.
# The Alphabet is a central step to enumerate molecules from ECFP vectors.
#
# Authors:
#  - Jean-loup Faulon <jfaulon@gmail.com>
#  - Thomas Duigou <thomas.duigou@inrae.fr>
#  - Philippe Meyer <philippe.meyer@inrae.fr>
# =================================================================================================

import copy
import time
from itertools import chain

import numpy as np
from rdkit import Chem

from molsig.Signature import MoleculeSignature

# =================================================================================================
# Alphabet class
# =================================================================================================


class SignatureAlphabet:
    """
    This class represents an Alphabet of atomic signatures associated with Morgan bits.

    Attributes
    ----------
    filename : str
        The name of the file used for saving or loading the alphabet (default is an empty string).
    radius : int
        The radius used for computing molecular signatures.
    nBits : int
        The number of bits in the Morgan fingerprint vector (default is 0, indicating no vector is used).
    use_stereo : bool
        Whether stereochemistry information is included in the molecular signatures.
    Dict : set
        A set containing the atomic or molecular signatures.

    Methods
    -------
    get_attributes()
        Retrieve the attributes of the object as a dictionary.
    fill(Smiles, verbose=False)
        Populate the signature dictionary from a list of SMILES strings.
    fill_from_signatures(signatures, atomic=False, verbose=False)
        Populate the alphabet from a set of atomic or molecular signature strings.
    save(filename)
        Save the SignatureAlphabet object to a compressed numpy file (.npz).
    print_out()
        Print the attributes of the SignatureAlphabet object.
    """

    def __init__(
        self,
        radius=2,
        nBits=0,
        use_stereo=False,
        Dict=set(),
    ):
        self.filename = ""
        self.radius = radius  # radius signatures are computed
        self.nBits = nBits  # the number of bits in Morgan vector (defaut 0 = no vector)
        self.use_stereo = use_stereo  # include information about stereochemistry
        self.Dict = Dict  # the set of atomic signatures

    def get_attributes(self):
        """
        Retrieve the attributes of the object as a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary containing the following key-value pairs:
            - 'radius': The radius attribute of the object.
            - 'nBits': The nBits attribute of the object.
            - 'use_stereo': The use_stereo attribute of the object.
        """

        return {
            "radius": self.radius,
            "nBits": self.nBits,
            "use_stereo": self.use_stereo,
        }

    def fill(self, Smiles, verbose=False):
        """
        Fill the signature dictionary from of list of smiles.

        Parameters
        ----------
        Smiles : list of str
            An array of SMILES strings representing molecules.
        verbose : bool, optional
            If True, print verbose output (default is False).
        """

        if verbose and len(self.Dict) > 0:
            print(f"WARNING alphabet non empty, alphabet length: {len(self.Dict)}")
        start_time = time.time()
        for i in range(len(Smiles)):
            smi = Smiles[i]
            if i % 1000 == 0:
                print(
                    f"... processing alphabet iteration: {i} size: {len(self.Dict)} time: {(time.time() - start_time):2f}"
                )
                start_time = time.time()
            if "*" in smi:  # no wild card allowed
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                if verbose:
                    print(f"WARNING no signature for molecule {i} {smi}")
                continue
            try:
                ms = MoleculeSignature(
                    mol,
                    radius=self.radius,
                    nbits=self.nBits,
                    map_root=True,
                    use_stereo=self.use_stereo,
                )
            except:
                if verbose:
                    print(f"PB signature {smi}")
                continue
            if len(ms.to_list()) == 0:
                if verbose:
                    print(f"WARNING no signature for molecule {i} {smi}")
                continue
            self.Dict = self.Dict | set(ms.to_list())

    def fill_from_signatures(self, signatures, atomic=False, verbose=False):
        """
        Fill the alphabet from a set of atomic signatures.

        Parameters
        ----------
        signatures : set of str
            A set of atomic or molecular signatures strings.
        atomic : bool, optional
            If False, the set signatures is composed of molecular signatures.
            If True, the set signatures is composed of atomic signatures (default is False).
        verbose : bool, optional
            If True, print verbose output (default is False).
        """

        if verbose and len(self.Dict) > 0:
            print(f"WARNING alphabet non empty, alphabet length: {len(self.Dict)}")
        if atomic:
            self.Dict = self.Dict | signatures
        else:
            atomic_signatures = [sig.split(" .. ") for sig in signatures]
            atomic_signatures_flattened = set(chain.from_iterable(atomic_signatures))
            self.Dict = self.Dict | atomic_signatures_flattened

    def save(self, filename):
        """
        Save the SignatureAlphabet object to a compressed numpy file (.npz).

        Parameters
        ----------
        filename : str
            The name of the file to save.
        """

        filename = filename + ".npz" if filename.find(".npz") == -1 else filename
        np.savez_compressed(
            filename,
            filename=filename,
            radius=self.radius,
            nBits=self.nBits,
            use_stereo=self.use_stereo,
            Dict=list(self.Dict),
        )

    def print_out(self):
        """
        Print out the attributes of the SignatureAlphabet object.
        """

        print(f"filename: {self.filename}")
        print(f"radius: {self.radius}")
        print(f"nBits: {self.nBits}")
        print(f"use_stereo: {self.use_stereo}")
        print(f"alphabet length: {len(self.Dict)}")


def compatible_alphabets(Alphabet_1, Alphabet_2):
    """
    Determine if two alphabets are compatible by comparing their attributes.

    Parameters
    ----------
    Alphabet_1 : object
        The first alphabet object. Must have a `get_attributes` method.
    Alphabet_2 : object
        The second alphabet object. Must have a `get_attributes` method.

    Returns
    -------
    bool
        True if the attributes of both alphabets are identical, False otherwise.
    """

    attributes_1 = Alphabet_1.get_attributes()
    attributes_2 = Alphabet_2.get_attributes()
    if len(attributes_1) != len(attributes_2):
        return False
    for x in attributes_1.keys():
        if x not in attributes_2.keys():
            return False
        if attributes_1[x] != attributes_2[x]:
            return False
    return True


def merge_alphabets(Alphabet_1, Alphabet_2):
    """
    Merge two alphabet objects into a new alphabet object.

    This function combines the dictionaries of two alphabet objects, ensuring that the resulting
    dictionary contains unique keys from both inputs. The values in the new dictionary are
    assigned as sequential integers.

    Parameters
    ----------
    Alphabet_1 : object
        The first alphabet object. Must have a `Dict` attribute.
    Alphabet_2 : object
        The second alphabet object. Must have a `Dict` attribute.

    Returns
    -------
    Alphabet_3 : object
        A new alphabet object with a combined dictionary from Alphabet_1 and Alphabet_2.
    """

    Alphabet_3 = copy.deepcopy(Alphabet_1)
    Alphabet_3.Dict = Alphabet_1.Dict | Alphabet_2.Dict
    return Alphabet_3


def load_alphabet(filename, verbose=False):
    """
    Load a signature alphabet from a NumPy compressed file (.npz).

    Parameters
    ----------
    filename : str
        The filename of the NumPy compressed file containing the alphabet.
    verbose : bool, optional
        If True, print information about the loaded alphabet (default is False).

    Returns
    -------
    Alphabet : SignatureAlphabet
        The loaded signature alphabet.
    """

    filename = filename + ".npz" if filename.find(".npz") == -1 else filename
    load = np.load(filename, allow_pickle=True)
    Alphabet = SignatureAlphabet()
    Alphabet.filename = filename
    Alphabet.Dict = set(load["Dict"])
    # Flags to compute signatures
    Alphabet.radius = int(load["radius"])
    Alphabet.nBits = int(load["nBits"])
    Alphabet.use_stereo = bool(load["use_stereo"])
    if verbose:
        Alphabet.print_out()
    return Alphabet
