########################################################################################################################
# This library compute the matrices necessary the enumeration function
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format
# Authors: Jean-loup Faulon jfaulon@gmail.com
# May 2023
########################################################################################################################


import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                               StereoEnumerationOptions)

from signature.signature_alphabet import signature_sorted_array

########################################################################################################################
# Local functions
########################################################################################################################


def bond_signature_occurence(bsig, asig):
    """
    Get the left and right atom signatures of the provided bond signature, along with their occurrence numbers in the
    neighbor atom signature.

    Parameters
    ----------
    bsig : str
        The bond signature (format: 'as1|bondtype|as2').
    asig : str
        The atom signature.

    Returns
    -------
    asig0 : str
        The signature of the root atom.
    as1 : str
        The signature of the left atom (as1).
    as2 : str
        The signature of the right atom (as2).
    occ1 : int
        The occurrence number of as1 in asign.
    occ2 : int
        The occurrence number of as2 in asign.
    """

    as1, as2 = bsig.split("|")[0], bsig.split("|")[2]
    btype = bsig.split("|")[1]
    asig0 = asig.split(" && ")[0]
    asign = asig.split(" && ")[1:]
    asig1, asig2 = btype + " <> " + as1, btype + " <> " + as2
    occ1, occ2 = asign.count(asig1), asign.count(asig2)
    return asig0, as1, as2, occ1, occ2


def constraint_matrix(AS, BS, deg, verbose=False):
    """
    Compute the constraints between bond and atom signatures. cf. C.J. Churchwell et al. Journal of Molecular Graphics
    and Modelling 22 (2004) 263–273.

    Parameters
    ----------
    AS : numpy.ndarray
        An array containing atom signatures.
    BS : numpy.ndarray
        An array containing bond signatures.
    deg : numpy.ndarray
        An array containing the degrees of atoms.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    C : numpy.ndarray
        Constraints between bond and atom signatures.
    """

    C = np.zeros((BS.shape[0], AS.shape[0]))
    # Constraints between bond and atom signatures
    for i in range(BS.shape[0]):
        for j in range(AS.shape[0]):
            asj, bs1, bs2, occ1, occ2 = bond_signature_occurence(BS[i], AS[j])
            if bs1 == asj:
                C[i, j] = occ2
            elif bs2 == asj and bs2 != bs1:
                C[i, j] = -occ1
        if bs1 == bs2:
            # adding even-valued column variable
            C = np.concatenate((C, np.zeros((C.shape[0], 1))), axis=1)
            C[i, -1] = -2
    if verbose == 2:
        print(f"Bond constraint: {C.shape},\n{C}")
    # The graphicality equation Sum_deg (deg-2)n_deg = 2z  - 2
    # Two cases:
    # Sum_deg (deg-2)n_deg < 0 <=> Sum_deg (deg-2)n_deg + 2z = 0
    #   here max (Z) must be 1, otherwise molecule cannot be connected
    # Sum_deg (deg-2)n_deg > 0 <=> Sum_deg (deg-2)n_deg - 2z = 0
    #   here max (Z) is bounded by Natom
    C = np.concatenate((C, np.zeros((1, C.shape[1]))), axis=0)
    for i in range(AS.shape[0]):
        C[-1, i] = deg[i] - 2
    C = np.concatenate((C, np.zeros((C.shape[0], 1))), axis=1)
    C[-1, -1] = 2
    C = np.concatenate((C, np.zeros((C.shape[0], 1))), axis=1)
    C[-1, -1] = -2
    if verbose == 2:
        print(f"Graphicality constraint: {C.shape},\n{C}")

    return C


########################################################################################################################
# Get the matrices necessary for the diophantine solver and the enumeration
########################################################################################################################


def bond_matrices(AS, NAS, deg, unique=True, verbose=False):
    """
    Generate bond matrices based on the provided atom signatures and the degree of each atom signature.

    Parameters
    ----------
    AS : numpy.ndarray
        An array of atom signatures.
    NAS : numpy.ndarray
        An array containing the occurrence number (degree) of each atom signature.
    deg : numpy.ndarray
        An array containing the degrees of atoms.
    unique : bool, optional
        If True, ensure unique bond signatures. Defaults to True.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    B : numpy.ndarray
        An array of bond candidate matrix.
    BS : numpy.ndarray
        An array of bond signatures.
    """

    N, K = AS.shape[0], np.max(deg)
    # Fill ABS, BBS (temp arrays used to find compatible bonds)
    ABS, BBS = [], []
    for i in range(N):
        asig0 = AS[i].split(" && ")[0]
        asign = AS[i].split(" && ")[1:]
        for k in range(K):
            if k < len(asign):
                btype = asign[k].split(" <> ")[0]  # bond type
                asigk = asign[k].split(" <> ")[1]  # neighbor signature
                ABS.append(f"{btype}|{asig0}")  # type + root signature
                BBS.append(f"{btype}|{asigk}")  # type + neighbor signature
            else:
                ABS.append("")
                BBS.append("")
    ABS, BBS = np.asarray(ABS), np.asarray(BBS)
    # Fill B (bond candidate matrix) and BS (bond signature)
    B, BS = np.zeros((N * K + 1, N * K), dtype="uint8"), []
    B[N * K] = np.zeros(N * K)

    for n in range(N):
        for k in range(K):
            i = n * K + k
            bsi = BBS[i]
            if bsi == "":
                break
            J = np.transpose(np.argwhere(ABS == bsi))[0]
            for j in J:
                if BBS[j] == ABS[i]:
                    ai, aj = int(i / K), int(j / K)
                    if ai == aj and NAS[ai] < 2:
                        continue  # cannot bind an atom to itself
                    B[i, j], B[N * K, i], B[N * K, j] = 1, 1, 1
                    bt = ABS[i].split("|")[0]
                    si = ABS[i].split("|")[1]
                    sj = ABS[j].split("|")[1]
                    bs = f"{si}|{bt}|{sj}" if si < sj else f"{sj}|{bt}|{si}"
                    BS.append(bs)
    BS = list(set(BS)) if unique else BS
    BS.sort()
    BS = np.asarray(BS)
    return B, BS


########################################################################################################################
# Callable functions
########################################################################################################################


def get_constraint_matrices(sig, unique=True, verbose=False):
    """
    Generate constraint matrices based on the provided molecule signature.

    Parameters
    ----------
    sig : str
        A molecule signature.
    unique : bool, optional
        If True, ensure unique atom signatures. Defaults to True.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    AS : numpy.ndarray
        An array of atom signatures.
    NAS : numpy.ndarray
        An array containing the occurrence number (degree) of each atom signature.
    deg : numpy.ndarray
        An array containing the degrees of atoms.
    A : numpy.ndarray
        An empty adjacency matrix between the atoms of the molecule with diagonal = atom degree.
    B : numpy.ndarray
        An adjacency matrix between the bond candidates of the molecule. The last row indicate is used during
        enumeration and filled with 0 at initialization.
    C : numpy.ndarray
        A constraint matrix between bond signature (row) and atom signature (columns).
    """

    AS, NAS, deg = signature_sorted_array(sig, unique=unique, verbose=verbose)
    N, K = AS.shape[0], np.max(deg)
    # Fill A (diag = degree, 0 elsewhere)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = deg[i]
    # Get B (bond candidate matrix) and BS (bond signature)
    B, BS = bond_matrices(AS, NAS, deg, unique=True, verbose=verbose)
    # Get constraint matrices
    C = constraint_matrix(AS, BS, deg, verbose=verbose)
    if verbose:
        print(f"A {A.shape}, B {B.shape} BS {BS.shape}, C {C.shape}")
    if verbose == 2:
        print(f"A\n {A} \nB\n {B} \nBS\n {BS} \nC\n {C}")

    return AS, NAS, deg, A, B, C


def update_constraint_matrices(AS, IDX, MAX, deg, verbose=False):
    """
    Update the constraint matrices based on the provided atom signatures and their properties.

    Parameters
    ----------
    AS : numpy.ndarray
        An array of atom signatures.
    IDX : numpy.ndarray
        An array containing the atom index.
    MAX : numpy.ndarray
        An array containing the maximum atom occurrence.
    deg : numpy.ndarray
        An array containing the degrees of atoms.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    AS : numpy.ndarray
        Updated array of atom signatures.
    IDX : numpy.ndarray
        Updated array containing the atom index.
    MAX : numpy.ndarray
        Updated array containing the maximum atom occurrence.
    deg : numpy.ndarray
        Updated array containing the degrees of atoms.
    C: numpy.ndarray
        Updated constraint matrix.
    """

    N = float("inf")
    while AS.shape[0] > 0 and AS.shape[0] < N:
        N, K, I = AS.shape[0], np.max(deg), []
        B, BS = bond_matrices(AS, MAX, deg, unique=True, verbose=verbose)
        for i in range(N):
            keep = True
            for k in range(deg[i]):
                if np.sum(B[i * K + k]) == 0:
                    keep = False
                    break
            if keep:
                I.append(i)
        AS = AS[I]
        IDX = [IDX[j] for j in I]
        MAX, deg = MAX[I], deg[I]

    if AS.shape[0] == 0:
        return AS, IDX, MAX, deg, np.array([])

    # Get constraint matrices
    C = constraint_matrix(AS, BS, deg, verbose=verbose)

    if verbose:
        print(f"UpdateConstraintMatrices AS {AS.shape} C {C.shape} BS {BS.shape}")
    if verbose == 2:
        print("UpdateConstraintMatrices AS")
        for i in range(AS.shape[0]):
            print(f"{i} {AS[i]}")
        print("UpdateConstraintMatrices BS")
        for i in range(BS.shape[0]):
            print(f"{i} {BS[i]}")

    return AS, IDX, MAX, deg, C


########################################################################################################################
# Function to test if a set of smiles has the same ECFP representation.
########################################################################################################################


def smiles_same_ecfp_or_not(smis, Alphabet):
    """
    Generate Extended-Connectivity Fingerprints (ECFP) for a list of SMILES strings and check if all generated fingerprints are identical.

    Parameters
    ----------
    smis : list of str
        A list of SMILES (Simplified Molecular Input Line Entry System) strings representing the molecules.
    Alphabet : object
        An object with attributes 'radius' and 'nBits' which specify the parameters for the Morgan fingerprint generator.
        'radius' determines the radius of the atom environments considered, and 'nBits' specifies the size of the fingerprint bit vector.

    Returns
    -------
    bool
        True if all generated ECFPs are identical, otherwise False.
    """

    fpgen = AllChem.GetMorganGenerator(
        radius=Alphabet.radius, fpSize=Alphabet.nBits, includeChirality=Alphabet.use_stereo
    )
    ecfp_list = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        morgan = fpgen.GetCountFingerprint(mol).ToList()
        ecfp_list.append(morgan)
    return len(set(tuple(i) for i in ecfp_list)) == 1


########################################################################################################################
# Function to test if a smiles has the same ECFP representation that a given ECFP.
########################################################################################################################


def smiles_ecfp_same_ecfp_or_not(morgan, smi, Alphabet):
    """
    Generate the Extended-Connectivity Fingerprints (ECFP) for the input SMILES string smi and check if it is equal to the input morgan ECFP vector.

    Parameters
    ----------
    morgan : list of int
        A counted ECFP having radius, fpSize and includeChirality parameters defined by the parameters of the Alphabet.
    smi : str
        A SMILES (Simplified Molecular Input Line Entry System) string representing a molecule.
    Alphabet : object
        An object with attributes 'radius', 'nBits' and 'use_stereo' which specify the parameters for the Morgan fingerprint generator.

    Returns
    -------
    bool
        True if the ECFP of the SMILES smi is equal to the input morgan ECFP, otherwise False.
    """

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    fpgen = AllChem.GetMorganGenerator(
        radius=Alphabet.radius, fpSize=Alphabet.nBits, includeChirality=Alphabet.use_stereo
    )
    morgan_2 = fpgen.GetCountFingerprint(mol).ToList()
    return morgan == morgan_2


########################################################################################################################
# Signature Callable functions.
########################################################################################################################


def signature_bond_type(bt="UNSPECIFIED"):
    """
    Convert a bond type string to its corresponding RDKit BondType object (Must be updated with new RDKit release).

    Parameters
    ----------
    bt : str, optional
        The bond type string. Defaults to "UNSPECIFIED".

    Returns
    -------
    RDKit.Chem.BondType
        The corresponding RDKit BondType object.
    """
    return Chem.BondType.names[bt]


########################################################################################################################
# Stereochemistry related functions.
########################################################################################################################


def generate_stereoisomers(smi, max_nb_stereoisomers=2048):
    """
    Generate all stereoisomers of a molecule.

    Parameters
    ----------
    smi : str
        The SMILES string of the molecule.

    Returns
    -------
    list
        A list of the SMILES of the stereoisomers of the input molecule.
    """

    mol = Chem.MolFromSmiles(smi)
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=max_nb_stereoisomers)
    stereoisomers = list(EnumerateStereoisomers(mol, options=options))
    return [Chem.MolToSmiles(isomer, isomericSmiles=True) for isomer in stereoisomers]


def get_first_stereoisomer(smi):
    """
    Generate the first stereoisomer of a molecule.

    Parameters
    ----------
    smi : str
        The SMILES string of the molecule.

    Returns
    -------
    str
        The SMILES string of the first stereoisomer of the input molecule.
    """

    # Create a molecule and set options
    mol = Chem.MolFromSmiles(smi)
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True)
    # Generate stereoisomers (as a generator)
    stereoisomers = EnumerateStereoisomers(mol, options=options)
    # Get only the first stereoisomer
    first_isomer = next(stereoisomers, None)  # None is a default if the generator is empty
    return Chem.MolToSmiles(first_isomer)


########################################################################################################################
# Isotopic related functions.
########################################################################################################################


def remove_isotopes(smi):
    """
    Remove isotopic information from a SMILES string.

    Parameters
    ----------
    smi : str
        Input SMILES string, potentially with isotopic information.

    Returns
    -------
    str
        SMILES string without isotopic information.
    """

    # Convert SMILES to an RDKit molecule
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    # Set the isotope of each atom to 0
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    # Convert the molecule back to a SMILES string
    return Chem.MolToSmiles(mol)
