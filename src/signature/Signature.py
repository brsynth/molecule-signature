"""This library compute signature on atoms and molecules using RDKit.

Molecule signature: the signature of a molecule is composed of the signature of its atoms. Molecule signatures are
implemented using MoleculeSignature objects.

Atom signature are represented by a rooted SMILES string (the root is the atom laleled 1). Atom signatures are
implemented as AtomSignature objects.

Below are format examples for the oxygen atom in phenol with radius=2
  - Default (nbits=0)
    C:C(:C)[OH:1]
    here the root is the oxygen atom labeled 1: [OH:1]
  - nBits=2048
    91,C:C(:C)[OH:1]
    91 is the Morgan bit of oxygen computed at radius 2

Atom signature can also be computed using neighborhood. A signature neighbor (string) is the signature of the
atom at radius followed but its signature at raduis-1 and the atom signatutre of its neighbor computed at radius-1

Example:
signature = C:C(:C)[OH:1]
signature-neighbor = C:C(:C)[OH:1]&C[OH:1].SINGLE|C:[C:1](:C)O
    after token &,  the signature is computed for the root (Oxygen)
    and its neighbor for radius-1, root and neighbor are separated by '.'
    The oxygen atom is linked by a SINGLE bond to a carbon of signature C:[C:1](:C)O

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
"""
import numpy as np
import logging
import re

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdqueries

# from rdkit.Chem import AllChem  # used for deprecated GetMorganFingerprintAsBitVect

# Logging settings
logger = logging.getLogger(__name__)

# Try importing rdcanon for canonicalization
try:
    from rdcanon import canon_smarts

except ImportError:
    logger.warning("rdcanon not found. Using default canonicalization function.")

    def canon_smarts(smarts, mapping=False):
        return smarts


# =====================================================================================================================
# Atom Signature
# =====================================================================================================================


class AtomSignature:
    def __init__(
        self, atom: Chem.Atom, radius: int = 2, use_smarts: bool = True, morgan_bit: int = None, **kwargs: dict
    ) -> None:
        # Parameters reminder
        self.radius = radius
        self.use_smarts = use_smarts
        self.kwargs = clean_kwargs(kwargs)

        # Meaningful information
        self._morgan = morgan_bit
        self._sig = None
        self._sig_minus = None
        self._neighbors = []

        # Compute signature of the atom itself
        self._sig = atom_signature(atom, self.radius, self.use_smarts, **self.kwargs)

        # Compute signature with neighbors
        if self.radius > 0:
            # Get the signatures of the neighbors at radius - 1
            self._sig_minus = atom_signature(atom, self.radius - 1, self.use_smarts, **self.kwargs)

            for neighbor_atom in atom.GetNeighbors():
                neighbor_sig = atom_signature(neighbor_atom, radius - 1, use_smarts, **self.kwargs)

                assert neighbor_sig != "", "Empty signature for neighbor"

                bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
                self._neighbors.append(
                    (
                        str(bond.GetBondType()),
                        neighbor_sig,
                    )
                )

            self._neighbors.sort()

        elif self.radius == 0:
            # There's nothing to do if radius is 0
            pass

        else:
            raise NotImplementedError("Radius must be a positive integer or zero.")

    def __repr__(self) -> str:
        _ = "AtomSignature("
        _ += f"morgan={self._morgan}, "
        _ += f"signature='{self._sig}', "
        _ += f"signature_minus='{self._sig_minus}', "
        _ += f"neighbor_signatures={self._neighbors})"
        _ += ")"
        return _

    def __lt__(self, other) -> bool:
        if self.sig == other.sig:
            if self.neighbors == other.neighbors:
                if self.morgan == other.morgan:
                    return False
                return self.morgan < other.morgan
            return self.neighbors < other.neighbors
        return self.sig < other.sig

    def __eq__(self, other) -> bool:
        # check if the signature are the same type
        if not isinstance(other, AtomSignature):
            return False
        return (
            self is other
            and self.sig == other.sig
            and self.neighbors == other.neighbors
            and self.morgan == other.morgan
        )

    @property
    def morgan(self) -> int:
        return self._morgan

    @property
    def sig(self) -> str:
        return self._sig

    @property
    def sig_minus(self) -> str:
        return self._sig_minus

    @property
    def neighbors(self) -> tuple:
        return " && ".join(f"{bond} <> {sig}" for bond, sig in self._neighbors)

    def as_deprecated_string(self, morgan=True, neighbors=False) -> str:
        s = ""
        if morgan:
            s += f"{self.morgan},"
        s += self.sig
        if neighbors:
            s += f"&{self.sig_minus}"
            for bond, sig in self._neighbors:
                s += f".{bond}|{sig}"
        return s


# =====================================================================================================================
# Atom Signature Helper Functions
# =====================================================================================================================


def atom_signature(atom: Chem.Atom, radius: int = 2, smarts: bool = True, **kwargs: dict):
    """Generate a signature for an atom

    This function generates a signature for an atom based on its environment up to a given radius. The signature is
    either represented as a SMARTS string (smarts=True) or a SMILES string (smarts=False). The atom is labeled as 1.

    Parameters
    ----------
    atom : Chem.Atom
        The atom to generate the signature for.
    radius : int
        The radius of the environment to consider. If negative, the whole molecule is considered.
    smarts : bool
        Whether to use SMARTS syntax for the signature.
    **kwargs
        Additional arguments to pass to Chem.MolFragmentToSmiles calls.

    Returns
    -------
    str
        The atom signature
    """

    # Initialize the signature
    signature = ""

    # Check if the atom is None
    if atom is None:
        return ""

    # Check if we're working on a hydrogen
    if atom.GetAtomicNum() == 1 and atom.GetFormalCharge() == 0 and not kwargs.get("allHsExplicit", False):
        return ""

    # Get the parent molecule
    mol = atom.GetOwningMol()

    # Refine the radius
    if radius < 0:
        # If radius is negative, consider the whole molecule
        radius = mol.GetNumAtoms()
    elif radius > mol.GetNumAtoms():
        # Radius cannot be larger than the number of atoms in the molecule
        radius = mol.GetNumAtoms()
    for radius in range(radius, 0, -1):
        # Check if the atom has an environment at the given radius
        # If the radius falls outside of the molecule (i.e. it does not reach any atom), the list of bonds is empty
        # In this case, we reduce the radius until we find a non-empty environment, or we reach radius 1.
        if len(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)) > 0:
            break
        assert radius > 1, "Atom environment not found"

    # Get the atoms and bonds to use
    if radius == 0:
        bonds = []
        atoms = [atom.GetIdx()]
    else:
        bonds = sorted(list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)))
        atoms = list()
        for bidx in bonds:
            atoms.append(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.append(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        atoms = sorted(list(set(atoms)))

    # Now we get to the business of computing the atom signature
    if smarts:
        signature = frag_to_smarts(mol, atoms, bonds, root_atom=atom.GetIdx(), **kwargs)
    else:
        signature = frag_to_smiles(mol, atoms, bonds, root_atom=atom.GetIdx(), **kwargs)

    # Debugging
    logger.debug(f"Atom signature: {signature}")

    return signature


def atom_to_smarts(atom: Chem.Atom, atom_map: int = 0) -> str:
    """Generate a SMARTS string for an atom

    Parameters
    ----------
    atom : Chem.Atom
        The atom to generate the SMARTS for
    atom_map : int
        The atom map number to use in the SMARTS string. If 0 (default), the atom map number is not used.

    Returns
    -------
    str
        The SMARTS string
    """

    # Initialize from the atom number
    qatom = Chem.MolFromSmarts(f"[#{atom.GetAtomicNum()}]").GetAtomWithIdx(0)

    # valence (most of the time, valence would remain the same for a given atom)
    # qatom.ExpandQuery(rdqueries.TotalValenceEqualsQueryAtom(atom.GetTotalValence()))
    # degree (number of explicit connections)
    # qatom.ExpandQuery(rdqueries.ExplicitDegreeEqualsQueryAtom(atom.GetDegree()))
    # total-H-count
    qatom.ExpandQuery(rdqueries.HCountEqualsQueryAtom(atom.GetTotalNumHs()))
    # connectivity (explicit + implict connections)
    qatom.ExpandQuery(rdqueries.TotalDegreeEqualsQueryAtom(atom.GetTotalDegree()))
    # ring membership
    qatom.ExpandQuery(rdqueries.IsInRingQueryAtom(negate=not atom.IsInRing()))
    # charges
    qatom.ExpandQuery(rdqueries.FormalChargeEqualsQueryAtom(atom.GetFormalCharge()))

    # Atom map
    if atom_map != 0:
        qatom.SetAtomMapNum(atom_map)

    # Transition to SMARTS for last minute refinements
    smarts = qatom.GetSmarts()

    # Aromatic vs aliphatic
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic():
        symbol = symbol.lower()
    smarts = smarts.replace(f"#{atom.GetAtomicNum()}", symbol)

    return smarts


def frag_to_smarts(mol: Chem.Mol, atoms: list, bonds: list, root_atom: int = -1, **kwargs) -> str:
    """Generate a SMARTS string for a subpart of molecule

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to generate the SMARTS for
    atoms : list
        The list of atom indices to include in the fragment
    bonds : list
        The list of bond indices to include in the fragment
    root_atom : int
        The atom index to use as the root atom. If -1 (default), no root atom is used, which means the fragment is not
        rooted.
    **kwargs
        Additional arguments to pass to Chem.MolFragmentToSmiles


    Returns
    -------
    str
        The SMARTS string
    """
    # Save atom map numbers if any
    mol_aams = {}
    for atom in mol.GetAtoms():
        mol_aams[atom.GetIdx()] = atom.GetAtomMapNum()

    # Store indices as atom map numbers
    # Note: we append +1 otherwise the atom map number 0 is not considered
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    # Gen the SMILES string to be used as scaffold
    smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atoms,
        bondsToUse=bonds,
        allBondsExplicit=True,
        allHsExplicit=kwargs.get("allHsExplicit", True),
        isomericSmiles=kwargs.get("isomericSmiles", True),
        canonical=True,
        kekuleSmiles=False,
        rootedAtAtom=root_atom,
    )

    # For debugging purposes
    logger.debug(f"Fragment SMILES: {smiles}")

    # Substitute with SMARTS
    smarts = smiles
    pattern = re.compile(r"(\[[^:\]]+:(\d+)\])")
    for atom_str, atom_map in re.findall(pattern, smiles):
        # Debugging
        logger.debug(f"    ├── Atom: {atom_str} (map: {atom_map})")

        # Get the atom index
        atom_idx = int(atom_map) - 1

        # Replace the atom string with the SMARTS
        smarts = smarts.replace(
            atom_str, atom_to_smarts(mol.GetAtomWithIdx(atom_idx), atom_map=1 if atom_idx == root_atom else 0)
        )

    # Restore atom map numbers
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(mol_aams[atom.GetIdx()])

    # Debugging
    logger.debug(f"Fragment SMARTS: {smarts}")

    # Canonicalize the SMARTS
    smarts = canon_smarts(smarts, mapping=True)

    return smarts


def frag_to_smiles(mol: Chem.Mol, atoms: list, bonds: list, root_atom: int = -1, **kwargs) -> str:
    """Generate a SMILES string for a subpart of molecule

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to generate the SMILES for
    atoms : list
        The list of atom indices to include in the fragment
    bonds : list
        The list of bond indices to include in the fragment
    root_atom : int
        The atom index to use as the root atom. If -1 (default), no root atom is used, which means the fragment is not
        rooted.
    **kwargs
        Additional arguments to pass to Chem.MolFragmentToSmiles calls

    Returns
    -------
    str
        The SMILES string
    """
    # Save atom map numbers if any
    mol_aams = {}
    for atom in mol.GetAtoms():
        mol_aams[atom.GetIdx()] = atom.GetAtomMapNum()

    # Label the root atom so we can retrieve it later within the SMILES, and remove any other atom map number
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(1 if atom.GetIdx() == root_atom else 0)

    smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atoms,
        bondsToUse=bonds,
        allBondsExplicit=True,
        allHsExplicit=kwargs.get("allHsExplicit", False),
        isomericSmiles=kwargs.get("isomericSmiles", False),
        kekuleSmiles=True,
        canonical=True,
        rootedAtAtom=root_atom,
    )
    # Chem.MolFragmentToSmiles canonicalizes the rooted fragment
    # but does not do the job properly.
    # To overcome the issue the atom is mapped to 1, and the smiles
    # is canonicalized via Chem.MolToSmiles
    _mol = Chem.MolFromSmiles(smiles)
    if kwargs.get("allHsExplicit", False):
        _mol = Chem.rdmolops.AddHs(_mol)
    smiles = Chem.MolToSmiles(_mol)

    # Restore atom map numbers
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(mol_aams[atom.GetIdx()])

    return smiles


# =====================================================================================================================
# Molecule Signature
# =====================================================================================================================


class MoleculeSignature:
    def __init__(
        self,
        mol: Chem.Mol,
        radius: int = 2,
        neighbor: bool = False,
        use_smarts: bool = True,
        nbits: int = 0,
        **kwargs: dict,
    ):
        # Deprecation warnings
        if "nBits" in kwargs:
            logger.warning("nBits is deprecated, use nbits instead.")
            nbits = kwargs.pop("nBits")

        # Parameters reminder
        self.radius = radius
        self.neighbor = neighbor
        self.use_smarts = use_smarts
        self.nbits = nbits
        self.kwargs = clean_kwargs(kwargs)

        # Meaningful information
        self.atom_signatures = []

        # Get Morgan bits
        if nbits > 0:
            # Prepare recipient to collect bits information
            bits_info = rdFingerprintGenerator.AdditionalOutput()
            bits_info.AllocateBitInfoMap()

            # Compute Morgan bits
            rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=nbits,
            ).GetFingerprint(mol, additionalOutput=bits_info)
            radius_vect = -np.ones(mol.GetNumAtoms(), dtype=int)
            morgan_vect = np.zeros(mol.GetNumAtoms(), dtype=int)
            for bit, info in bits_info.GetBitInfoMap().items():
                for atmidx, rad in info:
                    if rad > radius_vect[atmidx]:
                        radius_vect[atmidx] = rad
                        morgan_vect[atmidx] = bit

        # # Get Morgan bits (old way, using deprecated GetMorganFingerprintAsBitVect call)
        # if nbits > 0:
        #     bits_info = {}
        #     AllChem.GetMorganFingerprintAsBitVect(
        #         mol,
        #         radius=radius,
        #         nBits=nbits,
        #         bitInfo=bits_info,
        #         useChirality=self.kwargs.get("isomericSmiles", True),
        #         useFeatures=False,
        #     )
        #     radius_vect = -np.ones(mol.GetNumAtoms(), dtype=int)
        #     morgan_vect = np.zeros(mol.GetNumAtoms(), dtype=int)
        #     for bit, info in bits_info.items():
        #         for atmidx, rad in info:
        #             if rad > radius_vect[atmidx]:
        #                 radius_vect[atmidx] = rad
        #                 morgan_vect[atmidx] = bit

        # Compute the signatures of all atoms
        for atom in mol.GetAtoms():
            # Skip hydrogens
            if atom.GetAtomicNum() == 1 and atom.GetFormalCharge() == 0:
                continue

            # Collect non-empty atom signatures
            _sig = AtomSignature(
                atom=atom,
                radius=self.radius,
                use_smarts=self.use_smarts,
                morgan_bit=int(morgan_vect[atom.GetIdx()]) if nbits > 0 else None,  # int to avoid numpy.int64 type
                **self.kwargs,
            )
            if _sig != "":
                self.atom_signatures.append(_sig)

        assert len(self.atom_signatures) > 0, "No atom signature found"

        # Sort the atom signatures
        self.atom_signatures.sort()

    def __repr__(self) -> str:
        _ = "MoleculeSignature("
        _ += f"radius={self.radius}, "
        _ += f"neighbor={self.neighbor}, "
        _ += f"use_smarts={self.use_smarts}, "
        _ += f"nbits={self.nbits}, "
        _ += f"kwargs={self.kwargs}, "
        _ += f"atom_signatures={self.atom_signatures}"
        _ += ")"
        return _

    def __len__(self) -> int:
        return len(self.atom_signatures)

    def __str__(self) -> str:
        return self.as_str()

    def as_deprecated_string(self, morgan=True, neighbors=False) -> str:
        return " ".join(atom.as_deprecated_string(morgan, neighbors) for atom in self.atom_signatures)

    @property
    def atoms(self) -> list:
        return [atom.sig for atom in self.atom_signatures]

    @property
    def atoms_minus(self) -> list:
        return [atom.sig_minus for atom in self.atom_signatures]

    @property
    def neighbors(self) -> list:
        return [atom.neighbors for atom in self.atom_signatures]

    @property
    def morgans(self) -> list:
        return [atom.morgan for atom in self.atom_signatures]

    def as_list(self, morgan=True, neighbors=False) -> list:
        out = []
        for _morgan, _atom, _atom_minus, _neighbors in zip(
            self.morgans,
            self.atoms,
            self.atoms_minus,
            self.neighbors,
        ):
            if morgan:
                s = f"{str(_morgan)}, "
            else:
                s = ""
            if neighbors:
                s += f"{_atom_minus} || {_neighbors}"
            else:
                s += _atom
            out.append(s)
        return out

    def as_str(self, morgan=True, neighbors=False) -> str:
        return " .. ".join(self.as_list(morgan=morgan, neighbors=neighbors))


# =====================================================================================================================
# Overall helper functions
# =====================================================================================================================


def clean_kwargs(kwargs: dict) -> dict:
    """Check the kwargs dictionary for valid arguments

    This function checks the kwargs dictionary for valid arguments and returns a cleaned version of the dictionary.

    Parameters
    ----------
    kwargs : dict
        The dictionary of keyword arguments

    Returns
    -------
    dict
        The cleaned dictionary of keyword arguments
    """
    # Initialize the cleaned dictionary
    cleaned_kwargs = {}

    # Check for valid arguments
    for key, value in kwargs.items():
        if key in ["allHsExplicit", "isomericSmiles"]:
            cleaned_kwargs[key] = value
        else:
            logger.warning(f"Invalid argument: {key} (value: {value}), skipping argument.")

    return cleaned_kwargs


if __name__ == "__main__":
    # Example usage
    smiles = "c1ccccc1O"
    mol = Chem.MolFromSmiles(smiles)

    # Compute the molecule signature
    ms = MoleculeSignature(mol, radius=2, neighbor=True, use_smarts=False, nbits=2048)

    # Print the molecule signature
    print("Molecule signature ======================")
    print()

    print("Atoms:")
    print(ms.atoms)
    print()

    print("Atoms minus:")
    print(ms.atoms_minus)
    print()

    print("Neighbors:")
    print(ms.neighbors)
    print()

    print("Morgans:")
    print(ms.morgans)
    print()

    print("As list (morgan=True, neighbors=False):")
    for item in ms.as_list(morgan=True, neighbors=False):
        print(f"- {item}")
    print()

    print("As list (morgan=False, neighbors=True):")
    for item in ms.as_list(morgan=False, neighbors=True):
        print(f"- {item}")
    print()

    print("As string (morgan=True, neighbors=False):")
    print(ms.as_str(morgan=True, neighbors=False))
    print()

    print("As string (morgan=True, neighbors=True):")
    print(ms.as_str(morgan=True, neighbors=True))
    print()

    print("As deprecated string (morgan=True, neighbors=False):")
    print(ms.as_deprecated_string(morgan=True, neighbors=False))
    print()

    print("As deprecated string (morgan=True, neighbors=True):")
    print(ms.as_deprecated_string(morgan=True, neighbors=True))
    print()

    print("Testing combinations of parameters ======================")
    print()
    import itertools

    # Array of parameters
    arr_radius = [0, 1, 2]
    arr_neighbor = [False, True]
    arr_smarts = [False, True]
    arr_nbits = [0, 2048]
    for use_smarts, neighbor, radius, nbit in itertools.product(arr_smarts, arr_neighbor, arr_radius, arr_nbits):
        ms = MoleculeSignature(mol, radius=radius, neighbor=neighbor, use_smarts=use_smarts, nbits=nbit)
        # Pretty printings
        print(f"Molecule signature (radius={radius}, neighbor={neighbor}, use_smarts={use_smarts}, nbits={nbit}):")
        for atom_sig in ms.atom_signatures:
            print(f"├── {atom_sig}")
        print()
