########################################################################################################################
# This library compute signature on atoms and molecules using RDKit
#
# Molecule signature: the signature of a molecule is composed of the signature
# of its atoms. The string is separated by ' ' between atom signature
#
# Atom signature are represented by a rooted SMILES string
# (the root is the atom laleled 1)
#
# Below are format examples for the oxygen atom in phenol with radius = 2
#  - Default (nBbits=0)
#    C:C(:C)[OH:1]
#    here the root is the oxygen atom labeled 1: [OH:1]
#  - nBits=2048
#    91,C:C(:C)[OH:1]
#    91 is the Morgan bit of oxygen computed at radius 2
#
# Atom signature can also be computed using neighborhood.
# A signature neighbor (string) is the signature of the
# atom at radius followed but its signature at raduis-1
# and the atom signatutre of its neighbor computed at radius-1
# Example:
# signature = C:C(:C)[OH:1]
# signature-neighbor = C:C(:C)[OH:1]&C[OH:1].SINGLE|C:[C:1](:C)O
#    after token &,  the signature is computed for the root (Oxygen)
#    and its neighbor for radius-1, root and neighbor are separated by '.'
#    The oxygen atom is linked by a SINGLE bond to
#    a carbon of signature C:[C:1](:C)O

# Authors: Jean-loup Faulon jfaulon@gmail.com
# Jan 2023 modified July 2023, Jan. 2024
########################################################################################################################


import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


########################################################################################################################
# Signature Callable functions.
########################################################################################################################


def atom_signature(atm, radius=2, isomericSmiles=False, allHsExplicit=False, verbose=False):
    """
    Compute the signature (SMILES string) of an atom, where the root has label 1.

    Parameters
    ----------
    atm : RDKit.Atom
        The atom for which the signature is computed.
    radius : int, optional
        The radius up to which neighbor atoms and bonds are considered. Defaults to 2.
    isomericSmiles : bool, optional
        If True, generate isomeric SMILES. Defaults to False.
    allHsExplicit : bool, optional
        If True, include all hydrogen atoms explicitly. Defaults to False.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    signature : str
        The computed signature (SMILES string) of the atom.
    """

    signature = ""
    if atm is None:
        return signature
    if allHsExplicit is False:  # one keep charged hydrogen
        if atm.GetAtomicNum() == 1 and atm.GetFormalCharge() == 0:
            return signature
    mol = atm.GetOwningMol()
    if atm is None:
        return signature
    if radius < 0:
        radius = mol.GetNumAtoms()
    if radius > mol.GetNumAtoms():
        radius = mol.GetNumAtoms()

    # We get in atomToUse and env all neighbor atoms and bonds up to given radius
    atmidx = atm.GetIdx()
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atmidx, useHs=True)
    while len(env) == 0 and radius > 0:
        radius = radius - 1
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atmidx, useHs=True)
    if radius > 0:
        atoms = set()
        for bidx in env:
            atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        atomsToUse = list(atoms)
    else:
        atomsToUse = [atmidx]
        env = None

    # Now we get to the business of computing the atom signature
    atm.SetAtomMapNum(1)
    try:
        signature = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse,
            bondsToUse=env,
            rootedAtAtom=atmidx,
            isomericSmiles=isomericSmiles,
            kekuleSmiles=True,
            canonical=True,
            allBondsExplicit=True,
            allHsExplicit=allHsExplicit,
        )
        # Chem.MolFragmentToSmiles canonicalizes the rooted fragment
        # but does not do the job properly.
        # To overcome the issue the atom is mapped to 1, and the smiles
        # is canonicalized via Chem.MolToSmiles
        signature = Chem.MolFromSmiles(signature)
        if allHsExplicit:
            signature = Chem.rdmolops.AddHs(signature)
        signature = Chem.MolToSmiles(signature)
        if verbose == 2:
            print(f"signature for {atm.GetIdx()}: {signature}")

    except Exception:
        if verbose:
            print(f"WARNING cannot compute atom signature for: atom num: {atmidx} {atm.GetSymbol()} radius: {radius}")
        signature = ""
    atm.SetAtomMapNum(0)

    return signature


def get_atom_signature(
    atm,
    radius=2,
    neighbor=False,
    isomericSmiles=False,
    allHsExplicit=False,
    verbose=False,
):
    """
    Compute the signature of an atom and its neighbors in a molecule.

    Parameters
    ----------
    atm : RDKit.Atom
        The RDKit atom object for which the signature is computed.
    radius : int, optional
        The radius for considering neighboring atoms. Defaults to 2.
    neighbor : bool, optional
        If True, include neighbor atom signatures. Defaults to False.
    isomericSmiles : bool, optional
        If True, include information about stereochemistry in the signature. Defaults to False.
    allHsExplicit : bool, optional
        If True, include all hydrogen atoms explicitly in the signature. Defaults to False.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    signature : str
        A signature (string) where atom signatures are sorted in lexicographic order and separated by ' '.
    """

    signature = ""
    if neighbor and radius < 1:
        return ""

    # We compute atom signature for atm
    signature = atom_signature(
        atm,
        radius=radius,
        isomericSmiles=isomericSmiles,
        allHsExplicit=allHsExplicit,
        verbose=verbose,
    )
    if neighbor is False:
        return signature

    # We compute atm signature at radius-1
    radius = radius - 1
    s = atom_signature(
        atm,
        radius=radius,
        isomericSmiles=isomericSmiles,
        allHsExplicit=allHsExplicit,
        verbose=verbose,
    )
    if s == "":
        return ""
    signature = signature + "&" + s

    # We compute atom signatures for all neighbor at radius-1
    mol = atm.GetOwningMol()
    atmset = atm.GetNeighbors()
    sig_neighbor, temp_sig = "", []
    for a in atmset:
        s = atom_signature(
            a,
            radius=radius,
            isomericSmiles=isomericSmiles,
            allHsExplicit=allHsExplicit,
            verbose=verbose,
        )
        if s != "":
            bond = mol.GetBondBetweenAtoms(atm.GetIdx(), a.GetIdx())
            s = str(bond.GetBondType()) + "|" + s
            temp_sig.append(s)

    if len(temp_sig) < 1:
        return ""  # no signature because no neighbor
    temp_sig = sorted(temp_sig)
    sig_neighbor = ".".join(s for s in temp_sig)
    signature = signature + "." + sig_neighbor

    return signature


def get_molecule_signature(
    mol,
    radius=2,
    neighbor=False,
    nBits=0,
    isomericSmiles=False,
    allHsExplicit=False,
    verbose=False,
):
    """
    Compute the signature of a molecule.

    Parameters
    ----------
    mol : RDKit.Mol
        The molecule in RDKit format for which the signature is computed.
    radius : int, optional
        The radius of the signature. When `radius < 0`, the radius is set to the size of the molecule.
        Defaults to 2.
    neighbor : bool, optional
        If True, include neighbor atom signatures in the molecule signature. Defaults to False.
    nBits : int, optional
        Number of bits for the Morgan bit vector. When `nBits = 0`, the Morgan bit vector is not computed.
        Defaults to 0.
    isomericSmiles : bool, optional
        If True, include information about stereochemistry in the SMILES representation.
        Defaults to False.
    allHsExplicit : bool, optional
        If True, include all hydrogen counts explicitly in the output SMILES.
        Defaults to False.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    signature : str
        The molecule signature where atom signatures are sorted in lexicographic order and separated by ' '.
        See GetAtomSignature for atom signatures format.
    """

    signature, temp_signature, morgan = "", [], []

    # First get radius and Morgan bits for all atoms
    if nBits:
        bitInfo = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=nBits,
            bitInfo=bitInfo,
            useChirality=isomericSmiles,
            useFeatures=False,
        )
        Radius = -np.ones(mol.GetNumAtoms())
        morgan = np.zeros(mol.GetNumAtoms())
        for bit, info in bitInfo.items():
            for atmidx, rad in info:
                if rad > Radius[atmidx]:
                    Radius[atmidx] = rad
                    morgan[atmidx] = bit

    # We compute atom signatures for all atoms
    for atm in mol.GetAtoms():
        if atm.GetAtomicNum() == 1 and atm.GetFormalCharge() == 0:
            continue

        # We compute atom signature for atm
        sig = get_atom_signature(
            atm,
            radius=radius,
            neighbor=neighbor,
            isomericSmiles=isomericSmiles,
            allHsExplicit=allHsExplicit,
            verbose=verbose,
        )
        if sig != "":
            if nBits:  # Add morgan bit if any
                sig = str(int(morgan[atm.GetIdx()])) + "," + sig
            temp_signature.append(sig)

    # collect the signature for all atoms
    if len(temp_signature) < 1:
        return signature
    temp_signature = sorted(temp_signature)
    signature = " ".join(sig for sig in temp_signature)

    return signature


def signature_neighbor(sig):
    """
    Extract the neighbor atom signature from a given signature.

    Parameters
    ----------
    sig : str
        The input signature.

    Returns
    -------
    signature : str
        The neighbor atom signature.
    """

    L = sig.split(" ")
    for i in range(len(L)):
        s = L[i]
        if "," in s:
            s = s.split(",")[1]
        if "&" in s:
            s = s.split("&")[1]
        L[i] = s
    L = sorted(L)
    signature = " ".join(s for s in L)

    return signature


def atom_signature_mod(sa):
    """
    Modify the atom signature by replacing '.' with '_' and '|' with '§'.

    Parameters
    ----------
    sa : str
        Atom signature to be modified.

    Returns
    -------
    rsa : str
        Modified atom signature.
    """

    rsa = copy.copy(sa)
    rsa = rsa.replace(".", "_")
    rsa = rsa.replace("|", "§")

    return rsa
