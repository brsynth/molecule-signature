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


import sys
import time

import numpy as np
from rdkit import Chem

from src.signature.utils import dic_to_vector, vector_to_dic
from signature.signature_old import get_molecule_signature, sanitize_molecule


########################################################################################################################
# Alphabet callable class
########################################################################################################################


class SignatureAlphabet:
    def __init__(
        self,
        radius=2,
        nBits=0,
        splitcomponent=False,
        isomericSmiles=False,
        formalCharge=True,
        atomMapping=False,
        kekuleSmiles=False,
        allHsExplicit=False,
        maxvalence=4,
        Dict={},
    ):
        self.filename = ""
        self.radius = radius  # radius signatures are computed
        # the number of bits in Morgan vector (defaut 0 = no vector)
        self.nBits = nBits
        # when True the signature is computed for each molecule
        self.splitcomponent = splitcomponent
        # include information about stereochemistry
        self.isomericSmiles = isomericSmiles
        # Remove charges on atom when False. Defaults to False
        self.formalCharge = formalCharge
        # Remove atom mapping when False
        self.atomMapping = atomMapping
        self.kekuleSmiles = kekuleSmiles
        # if true, all H  will be explicitly in signatures
        self.allHsExplicit = allHsExplicit
        self.maxvalence = maxvalence  # for all atoms
        # the alphabet dictionary keys = atom signature, values = index
        self.Dict = Dict

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
            if i % 1000 == 0:
                print(
                    f"... processing alphabet iteration: {i} size: {len(list(Dict))} time: {(time.time()-start_time)}"
                )
                start_time = time.time()
            if "*" in Smiles[i]:  # no wild card allowed
                continue
            signature, _, _ = signature_from_smiles(
                Smiles[i], self, neighbor=True, verbose=verbose
            )
            if len(signature) == 0:
                if verbose:
                    print(f"WARNING no signature for molecule {i} {Smiles[i]}")
                continue
            for sig in signature.split(" . "):  # separate molecules
                for s in sig.split(" "):  # separate atom signatures
                    Dict.add(s)
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
                )  # noqa: E501
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
            splitcomponent=self.splitcomponent,
            isomericSmiles=self.isomericSmiles,
            formalCharge=self.formalCharge,
            atomMapping=self.atomMapping,
            kekuleSmiles=self.kekuleSmiles,
            allHsExplicit=self.allHsExplicit,
            maxvalence=self.maxvalence,
            Dict=list(self.Dict.keys()),
        )

    def print_out(self):
        """
        Print out the attributes of the SignatureAlphabet object.
        """

        print(f"filename: {self.filename}")
        print(f"radius: {self.radius}")
        print(f"nBits: {self.nBits}")
        print(f"splitcomponent: {self.splitcomponent}")
        print(f"isomericSmiles: {self.isomericSmiles}")
        print(f"formalCharge: {self.formalCharge}")
        print(f"atomMapping: {self.atomMapping}")
        print(f"kekuleSmiles: {self.kekuleSmiles}")
        print(f"allHsExplicit: {self.allHsExplicit}")
        print(f"maxvalence: {self.maxvalence}")
        print(f"alphabet length: {len(self.Dict.keys())}")


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
    Alphabet.maxvalence = int(load["maxvalence"])
    Alphabet.splitcomponent = bool(load["splitcomponent"])
    Alphabet.isomericSmiles = bool(load["isomericSmiles"])
    Alphabet.formalCharge = bool(load["formalCharge"])
    Alphabet.atomMapping = bool(load["atomMapping"])
    Alphabet.kekuleSmiles = bool(load["kekuleSmiles"])
    Alphabet.allHsExplicit = bool(load["allHsExplicit"])
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

    AS, NAS, Deg = signature_sorted_array(
        sig, Alphabet=None, unique=False, verbose=verbose
    )
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


def signature_sorted_array(sig, Alphabet=None, unique=False, verbose=False):
    """
    Convert a signature into a sorted array of atom signatures along with occurrence numbers and degrees.

    Parameters
    ----------
    sig : str
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
        sig = signature_vector_to_string(sigV, Alphabet.Dict, verbose=verbose)
    LAS = sig.split(" ")
    LAS.sort()
    AS = list(set(LAS)) if unique else LAS
    AS.sort()
    AS = np.asarray(AS)
    N = AS.shape[0]  # nbr of atoms
    NAS, deg, M = {}, {}, 0
    for i in range(N):
        NAS[i] = LAS.count(AS[i]) if unique else 1
        deg[i] = len(AS[i].split(".")) - 1
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


########################################################################################################################
# Signature string or vector computed from smiles
########################################################################################################################


def signature_from_smiles(smiles, Alphabet, neighbor=False, string=True, verbose=False):
    """
    Get a sanitized signature vector for the provided SMILES. A local routine to make sure all signatures are standard.

    Parameters
    ----------
    smiles : str
        A SMILES string (can contain several molecules separated by '.').
    Alphabet : SignatureAlphabet
        A SignatureAlphabet object.
    neighbor : bool, optional
        Return a signature neighbor when True (default is False).
    string : bool, optional
        Return a string when True, else return a vector (default is True).
    verbose : bool, optional
        If True, enable verbose output (default is False).

    Returns
    -------
    signature : str or numpy.ndarray
        The signature string or vector.
    molecule : list
        An array of RDKit molecule objects.
    smiles : str
        The SMILES representation of the molecules.
    """

    S = smiles.split(".") if Alphabet.splitcomponent else [smiles]
    signature, temp, molecule, smiles = "", [], [], ""
    for i in range(len(S)):
        mol = Chem.MolFromSmiles(S[i])
        mol, smi = sanitize_molecule(
            mol,
            kekuleSmiles=Alphabet.kekuleSmiles,
            allHsExplicit=Alphabet.allHsExplicit,
            isomericSmiles=Alphabet.isomericSmiles,
            formalCharge=Alphabet.formalCharge,
            atomMapping=Alphabet.atomMapping,
            verbose=verbose,
        )
        if mol is None:
            continue
        sig = get_molecule_signature(
            mol,
            radius=Alphabet.radius,
            neighbor=neighbor,
            nBits=Alphabet.nBits,
            isomericSmiles=Alphabet.isomericSmiles,
            allHsExplicit=Alphabet.allHsExplicit,
            verbose=verbose,
        )
        if sig != "":
            temp.append(sig)
            molecule.append(mol)
            smiles = f"{smiles}.{smi}" if len(smiles) else smi

    if len(temp) < 1:
        if string is False and Alphabet.Dict != {}:
            return [], molecule, ""
        else:
            return "", molecule, ""

    temp = sorted(temp)
    signature = " . ".join(sig for sig in temp)

    if string is False and Alphabet.Dict != {}:
        signature = signature_string_to_vector(
            signature, Alphabet.Dict, verbose=verbose
        )

    return signature, molecule, smiles


########################################################################################################################
# Morgan Vector of a signature
########################################################################################################################


def morgan_vector_string(morgan):
    """
    Convert a Morgan vector to a string representation.

    Parameters
    ----------
    morgan : numpy.ndarray
        A Morgan vector.

    Returns
    -------
    s : str
        A string representation of the Morgan vector.
    """

    s = ""
    for i in range(len(morgan)):
        if morgan[i]:
            s = f"{i}:{morgan[i]}" if s == "" else s + f",{i}:{morgan[i]}"
    return s


def morgan_bit_from_signature(sa, verbose=False):
    """
    Extract the Morgan bit from a signature.

    Parameters
    ----------
    sa : str
        The signature containing Morgan bit information.
    verbose : bool, optional
        If True, print verbose output. Default is False.

    Returns
    -------
    int
        The Morgan bit extracted from the signature.
    str
        The signature with Morgan bit removed.
    """

    if len(sa.split(",")) == 0:
        return -1, sa
    else:
        return int(sa.split(",")[0]), sa.split(",")[1]


def morgan_vector_from_signature(signature, Alphabet, verbose=False):
    """
    Get the Morgan vector from a signature.

    Parameters
    ----------
    signature : str
        The signature string of a molecule.
    Alphabet : SignatureAlphabet
        The alphabet of atom signatures.
    verbose : bool, optional
        If True, print verbose output. Default is False.

    Returns
    -------
    MorganVector : numpy.ndarray
        A Morgan vector of size nBits.
    signature : str
        The signature stripped of Morgan bits.
    """

    MorganVector = np.zeros(Alphabet.nBits)
    lsas = []
    for sa in signature.split(" "):  # separate atom signatures
        mbit, sas = morgan_bit_from_signature(sa, verbose=verbose)
        if mbit < 0:
            if verbose:
                print("Error signature does not include Morgan bits")
            return MorganVector, signature
        lsas.append(sas)
        MorganVector[mbit] += 1

    # sort signature stripped of morgan bits
    lsas = sorted(lsas)
    signature = " ".join(sig for sig in lsas)

    return MorganVector, signature


def signature_alphabet_from_morgan_bit(MorganBit, Alphabet, verbose=False):
    """
    Get all signatures in the alphabet having the provided Morgan bit.

    Parameters
    ----------
    MorganBit : int
        An integer in the range [0, nBits].
    Alphabet : SignatureAlphabet
        The alphabet of atom signatures.
    verbose : bool, optional
        If True, print verbose output. Default is False.

    Returns
    -------
    list(Signatures) : list
        A list of signatures having the provided Morgan bit.
    """

    Signatures = []
    if Alphabet.Dict == {}:
        if verbose:
            print("WARNING Empty Alphabet")
        return Signatures
    if MorganBit > Alphabet.nBits:
        if verbose:
            print(f"WARNING MorganBit {MorganBit} exceeds nBits {Alphabet.nBits}")
        return Signatures
    for sig in Alphabet.Dict.keys():
        mbit, sa = int(sig.split(",")[0]), sig.split(",")[1]
        if mbit == MorganBit:
            Signatures.append(sa)

    return list(Signatures)
