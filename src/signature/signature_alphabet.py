########################################################################################################################
# This library compute save and load Alphabet of atom signatures
# The library also provides functions to compute a molecule signature as a
# vector of occurence numbers over an Alphabet, and a Morgan Fingerprint
# from a molecule signature string.
# Note that when molecules have several connected components
# the individual molecular signatures string are separated by ' . '
# Authors: Jean-loup Faulon jfaulon@gmail.com
# March 2023, Updated Jan 2024
########################################################################################################################

import copy
import sys
import time

import numpy as np
from rdkit import Chem

from signature.Signature import MoleculeSignature
from signature.utils import dic_to_vector, vector_to_dic


########################################################################################################################
# Alphabet callable class
########################################################################################################################


class SignatureAlphabet:
    def __init__(
        self,
        radius=2,
        nBits=0,
        map_root=True,
        use_stereo=False,
        Dict={},
    ):
        self.filename = ""
        self.radius = radius  # radius signatures are computed
        self.nBits = nBits  # the number of bits in Morgan vector (defaut 0 = no vector)
        self.map_root = map_root # put :1 to identify the root
        self.use_stereo = use_stereo # include information about stereochemistry
        self.Dict = Dict  # the alphabet dictionary keys = atom signature, values = index

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
            - 'use_smarts': The use_smarts attribute of the object.
            - 'boundary_bonds': The boundary_bonds attribute of the object.
            - 'map_root': The map_root attribute of the object.
        """

        return {
            "radius": self.radius,
            "nBits": self.nBits,
            "map_root": self.map_root,
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

        if self.Dict != {}:  # there's already signatures in the alphabet
            Dict = dic_to_vector(self.Dict)  # return a set
        else:
            Dict = set()
        start_time = time.time()
        for i in range(len(Smiles)):
            smi = Smiles[i]
            if i % 1000 == 0:
                print(
                    f"... processing alphabet iteration: {i} size: {len(list(Dict))} time: {(time.time() - start_time):2f}"
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
                    map_root=self.map_root,
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
            for _as in ms.to_list():
                Dict.add(_as)
        self.Dict = vector_to_dic(list(Dict))

    def fill_from_signatures(self, signatures, verbose=False):
        """
        Fill the signature dictionary from an array of signatures.

        Parameters
        ----------
        signatures : list of str
            An array of signature strings.
        verbose : bool, optional
            If True, print verbose output (default is False).
        """

        if self.Dict != {}:
            Dict = dic_to_vector(self.Dict)
        else:
            Dict = set()
        start_time = time.time()
        for i, signature in enumerate(signatures):
            if i % 10000 == 0:
                print(
                    f"... processing alphabet iteration: {i:,} size: {len(Dict):,} time: {(time.time()-start_time):.2f}"
                )
                start_time = time.time()
            for sig in signature.split(" . "):  # separate molecules
                for s in sig.split(" "):  # separate atom signatures
                    Dict.add(s)
        self.Dict = vector_to_dic(list(Dict))

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
            map_root=self.map_root,
            use_stereo=self.use_stereo,
            Dict=list(self.Dict.keys()),
        )

    def print_out(self):
        """
        Print out the attributes of the SignatureAlphabet object.
        """

        print(f"filename: {self.filename}")
        print(f"radius: {self.radius}")
        print(f"nBits: {self.nBits}")
        print(f"map_root: {self.map_root}")
        print(f"use_stereo: {self.use_stereo}")
        print(f"alphabet length: {len(self.Dict.keys())}")


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

    d_1 = Alphabet_1.Dict
    d_2 = Alphabet_2.Dict
    d_3_keys = list(d_1.keys()) + list(d_2.keys())
    d_3_keys = list(set(d_3_keys))
    d_3_values = list(range(len(d_3_keys)))
    d_3 = dict(zip(d_3_keys, d_3_values))
    Alphabet_3 = copy.deepcopy(Alphabet_1)
    Alphabet_3.Dict = d_3
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
    Alphabet.Dict = vector_to_dic(load["Dict"])
    # Flags to compute signatures
    Alphabet.radius = int(load["radius"])
    Alphabet.nBits = int(load["nBits"])
    Alphabet.map_root = int(load["map_root"])
    Alphabet.use_stereo = bool(load["use_stereo"])
    if verbose:
        Alphabet.print_out()
    return Alphabet


########################################################################################################################
# Signature utilities
########################################################################################################################


def signature_string_to_vector(signature, Dict, verbose=False):
    """
    Convert a string signature into a vector representation using a given dictionary.

    Parameters
    ----------
    signature : str
        The string signature to convert.
    Dict : dict
        A dictionary of unique atom signatures.
    verbose : bool, optional
        If True, print error messages for atom signatures not found in the dictionary
        (default is False).

    Returns
    -------
    signatureV : numpy.ndarray
        An array of Dict size where SigV[i] is the occurence number of the atom signatures in
        the sig signature.
    """

    signatureV = np.zeros(len(Dict.keys()))
    for sig in signature.split(" . "):  # separate molecules
        for s in sig.split(" "):  # separate atom signatures
            try:
                index = Dict[s]
                signatureV[index] += 1
            except Exception:
                print(f"Error atom signature not found in Alphabet {s}")
                continue  # !!!
                sys.exit("Error")

    return signatureV


def signature_sorted_string(sig, verbose=False):
    """
    Sort a given signature string.

    Parameters
    ----------
    sig : str
        The input signature string.
    verbose : bool, optional
        If True, print additional information (default is False).

    Returns
    -------
    sigsorted : str
        The sorted signature string.
    """

    AS, NAS, Deg = signature_sorted_array(sig, Alphabet=None, unique=False, verbose=verbose)
    sigsorted = AS[0]
    for i in range(1, AS.shape[0]):
        sigsorted = sigsorted + " " + AS[i]

    return sigsorted


def signature_vector_to_string(sigV, Dict, verbose=False):
    """
    Convert a vector signature into a string representation using a given dictionary.

    Parameters
    ----------
    sigV : numpy.ndarray
        A vector signature where sigV[i] is the occurrence number of the atom signatures.
    Dict : dict, list, or array
        A dictionary, list, or array containing atom signatures.
    verbose : bool, optional
        If True, enable verbose output (default is False).

    Returns
    -------
    signature_sorted_string(sig) : str
        A string representation of the signature.
    """

    I, sig = np.transpose(np.argwhere(sigV != 0))[0], ""
    if isinstance(Dict, (dict)):
        A = list(Dict.keys())
    else:
        A = list(Dict)
    for i in I:
        for k in range(int(sigV[i])):
            sig = A[int(i)] if sig == "" else sig + " " + A[i]

    return signature_sorted_string(sig, verbose=verbose)


def signature_sorted_array(LAS, Alphabet=None, unique=False, verbose=False):
    """
    Convert a signature into a sorted array of atom signatures along with occurrence numbers and degrees.

    Parameters
    ----------
    LAS : str
        A signature string.
    Alphabet : SignatureAlphabet, optional
        A SignatureAlphabet object. If provided, the signature is converted to a string using the dictionary
        in the alphabet.
    unique : bool, optional
        A flag indicating if the atom signature list must contain only unique atom signatures (default is False).
    verbose : bool, optional
        If True, enable verbose output (default is False).

    Returns
    -------
    AS : numpy.ndarray
        An array of atom signatures.
    NAS : numpy.ndarray
        An array of occurrence numbers (degree) of each atom signature.
    deg : numpy.ndarray
        An array of degrees of each atom signature.
    """

    if Alphabet is not None:
        LAS = signature_vector_to_string(sigV, Alphabet.Dict, verbose=verbose)
    # LAS.sort()
    # AS = list(set(LAS)) if unique else LAS
    # AS.sort()
    AS = np.asarray(LAS)
    N = AS.shape[0]  # nbr of atoms
    NAS, deg, M = {}, {}, 0
    for i in range(N):
        NAS[i] = LAS.count(AS[i]) if unique else 1
        deg[i] = len(AS[i].split(" && ")) - 1
        M = M + deg[i]
    Ncycle = int(M / 2 - N + 1)
    NAS = np.asarray(list(NAS.values()))
    deg = np.asarray(list(deg.values()))

    if verbose:
        print(f"Nbr atoms, bonds, Cycle, {N}, {int(M/2)}, {Ncycle}")
        print(f"LAS, {len(AS)}")
        for i in range(len(LAS)):
            print(f"- {i}: {LAS[i]}")
        print(f"Deg {deg}, {len(deg)}")
        print(f"NAS, {NAS}, {len(NAS)}")

    return AS, NAS, deg
