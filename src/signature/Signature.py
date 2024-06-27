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
import os
import re

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Logging settings
logger = logging.getLogger(__name__)

# =====================================================================================================================
# Atom Signature
# =====================================================================================================================


class AtomSignature:
    """Class representing the signature of an atom."""

    def __init__(
        self,
        atom: Chem.Atom,
        radius: int = 2,
        use_smarts: bool = False,
        boundary_bonds: bool = True,
        map_root: bool = True,
        rooted_smiles: bool = False,
        morgan_bit: int = None,
        legacy: bool = False,
        **kwargs: dict,
    ) -> None:
        """Initialize the AtomSignature object

        Parameters
        ----------
        atom : Chem.Atom
            The atom to generate the signature for.
        radius : int
            The radius of the environment to consider.
        boundary_bonds : bool
            Whether to add bonds at the boundary of the radius.
        use_smarts : bool
            Whether to use SMARTS syntax for the signature (otherwise, use SMILES syntax).
        map_root : bool
            Whether to map the root atom in the signature. If yes, the root atom is labeled as 1.
        rooted_smiles : bool
            Whether to use rooted SMILES syntax for the signature. If yes, the SMILES is rooted at the root atom.
        morgan_bit : int
            The Morgan bit to be associated to the atom (if any).
        legacy : bool
            Whether to use the legacy version of the atom signature.
        **kwargs
            Additional arguments to pass to Chem.MolFragmentToSmiles calls.
        """
        # Parameters
        self.radius = radius
        self.use_smarts = use_smarts
        self.boundary_bonds = boundary_bonds
        self.map_root = map_root
        self.rooted_smiles = rooted_smiles
        self.kwargs = clean_kwargs(kwargs)

        # Meaningful information
        self._morgan = morgan_bit
        self._sig = None
        self._sig_minus = None
        self._neighbors = []

        # Wildly switch between new and legacy version for atom signature
        #
        # This is a temporary solution to allow the use of the new atom signature
        # while keeping the old one for backward compatibility.
        if legacy:
            atom_signature = globals()["atom_signature_legacy"]
        else:
            atom_signature = globals()["atom_signature"]
            self.kwargs["boundary_bonds"] = self.boundary_bonds
            self.kwargs["map_root"] = self.map_root
            self.kwargs["rooted_smiles"] = self.rooted_smiles

        # Compute signature of the atom itself
        self._sig = atom_signature(
            atom,
            self.radius,
            self.use_smarts,
            **self.kwargs,
        )

        # Compute signature with neighbors
        if self.radius > 0:
            # Get the signatures of the neighbors at radius - 1
            self._sig_minus = atom_signature(
                atom,
                self.radius - 1,
                self.use_smarts,
                **self.kwargs,
            )

            for neighbor_atom in atom.GetNeighbors():
                neighbor_sig = atom_signature(
                    neighbor_atom,
                    radius - 1,
                    self.use_smarts,
                    **self.kwargs,
                )

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
            raise ValueError("Radius must be a positive integer or zero.")

    def __repr__(self) -> str:
        _ = "AtomSignature("
        _ += f"morgan={self._morgan}, "
        _ += f"signature='{self._sig}', "
        _ += f"signature_minus='{self._sig_minus}', "
        _ += f"neighbor_signatures={self._neighbors}, "
        _ += f"boundary_bonds={self.boundary_bonds}, "
        _ += f"use_smarts={self.use_smarts}, "
        _ += f"map_root={self.map_root}, "
        _ += f"rooted_smiles={self.rooted_smiles}"
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
            self.sig == other.sig
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

    def as_deprecated_string(self, morgan=True, root=True, neighbors=True) -> str:
        """Return the signature in the deprecated string format

        Parameters
        ----------
        morgan : bool
            Whether to include the Morgan bit in the string.
        neighbors : bool
            Whether to include the neighbors in the string.

        Returns
        -------
        str
            The signature in the deprecated string format.
        """
        s = ""
        if morgan:
            s += f"{self.morgan},"
        if root:
            s += f"{self.sig}&"
        if neighbors:
            s += self.sig_minus
            for bond, sig in self._neighbors:
                s += f".{bond}|{sig}"
        return s


# =====================================================================================================================
# Atom Signature Helper Functions
# =====================================================================================================================


def atom_signature(
    atom: Chem.Atom,
    radius: int = 2,
    use_smarts: bool = False,
    boundary_bonds: bool = True,
    map_root: bool = True,
    rooted_smiles: bool = False,
    **kwargs: dict,
) -> str:
    """Generate a signature for an atom (development version)

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
    boundary_bonds : bool
        Whether to use boundary bonds at the border of the radius. This option is only available for SMILES syntax.
    map_root : bool
        Whether to map the root atom in the signature. If yes, the root atom is labeled as 1.
    rooted_smiles : bool
        Whether to use rooted SMILES syntax for the signature. If yes, the SMILES is rooted at the root atom.
    **kwargs
        Additional arguments to pass to Chem.MolFragmentToSmiles calls.

    Returns
    -------
    str
        The atom signature
    """
    # Get the parent molecule
    mol = atom.GetOwningMol()

    # If radius is negative, consider the whole molecule
    if radius < 0:
        radius = mol.GetNumAtoms()

    # Get the bonds at the border of the radius
    bonds_radius = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
    bonds_radius_plus = Chem.FindAtomEnvironmentOfRadiusN(mol, radius + 1, atom.GetIdx())
    bonds = [b for b in bonds_radius_plus if b not in bonds_radius]

    # Fragment the molecule
    if len(bonds) > 0:
        # Fragment the molecule
        fragmented_mol = Chem.FragmentOnBonds(
            mol,
            bonds,
            addDummies=True,
            dummyLabels=[(0, 0) for _ in bonds],  # Do not label the dummies
        )
    else:  # No bonds to cut
        fragmented_mol = mol

    # Retrieve the rooted fragment from amongst all the fragments
    frag_to_mol_atom_mapping = []  # Mapping of atom indexes between original and fragments
    for _frag_idx, _fragment in enumerate(
        Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=frag_to_mol_atom_mapping)
    ):
        if atom.GetIdx() in frag_to_mol_atom_mapping[_frag_idx]:
            fragment = _fragment
            frag_to_mol_atom_mapping = frag_to_mol_atom_mapping[_frag_idx]  # Dirty..
            atom_in_frag_index = frag_to_mol_atom_mapping.index(atom.GetIdx())  # Atom index in the fragment
            break

    # Get the SMARTS / SMILES
    if use_smarts:  # Get the SMARTS

        # Set a canonical atom mapping
        if fragment.NeedsUpdatePropertyCache():
            fragment.UpdatePropertyCache(strict=False)
        canonical_map(fragment)

        # Build with / without the dummies
        if boundary_bonds:
            _atoms_to_use = list(range(fragment.GetNumAtoms()))
        else:
            _atoms_to_use = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() != 0]
        _atom_symbols = [
            atom_to_smarts(
                atom=_atom,
                atom_map=1 if _atom.GetIdx() == atom_in_frag_index and map_root else 0,
            )
            for _atom in fragment.GetAtoms()
        ]
        smarts = Chem.MolFragmentToSmiles(
            fragment,
            atomsToUse=_atoms_to_use,
            atomSymbols=_atom_symbols,
            isomericSmiles=kwargs.get("isomericSmiles", True),
            allBondsExplicit=kwargs.get("allBondsExplicit", True),
            allHsExplicit=kwargs.get("allHsExplicit", False),
            kekuleSmiles=kwargs.get("kekuleSmiles", False),
            canonical=True,
            rootedAtAtom=atom_in_frag_index if rooted_smiles else -1,
        )

        # Return the SMARTS
        return smarts

    else:  # Get the SMILES

        if map_root:  # Map the root atom
            fragment.GetAtomWithIdx(atom_in_frag_index).SetAtomMapNum(1)

        # Build with / without the dummies
        if boundary_bonds:
            _atoms_to_use = list(range(fragment.GetNumAtoms()))
        else:
            _atoms_to_use = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() != 0]

        return Chem.MolFragmentToSmiles(
            fragment,
            atomsToUse=_atoms_to_use,
            isomericSmiles=kwargs.get("isomericSmiles", True),
            allBondsExplicit=kwargs.get("allBondsExplicit", True),
            allHsExplicit=kwargs.get("allHsExplicit", False),
            kekuleSmiles=kwargs.get("kekuleSmiles", False),
            canonical=True,
            rootedAtAtom=atom_in_frag_index if rooted_smiles else -1,
        )

        # return Chem.MolToSmiles(
        #     fragment,
        #     isomericSmiles=kwargs.get("isomericSmiles", True),
        #     allBondsExplicit=kwargs.get("allBondsExplicit", True),
        #     allHsExplicit=kwargs.get("allHsExplicit", False),
        #     kekuleSmiles=kwargs.get("kekuleSmiles", False),
        #     canonical=True,
        #     rootedAtAtom=atom_in_frag_index if rooted_smiles else -1,
        # )


def atom_signature_legacy(
    atom: Chem.Atom,
    radius: int = 2,
    use_smarts: bool = False,
    **kwargs: dict,
) -> str:
    """Generate a signature for an atom

    This function generates a signature for an atom based on its environment up to a given radius. The signature is
    either represented as a SMARTS string (smarts=True) or a SMILES string (smarts=False). The atom is labeled as 1.

    Parameters
    ----------
    atom : Chem.Atom
        The atom to generate the signature for.
    radius : int
        The radius of the environment to consider. If negative, the whole molecule is considered.
    use_smarts : bool
        Whether to use SMARTS syntax for the signature.
    **kwargs
        Additional arguments to pass to Chem.MolFragmentToSmiles calls.

    Returns
    -------
    str
        The atom signature
    """
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
    for radius in range(radius, -1, -1):
        # Check if the atom has an environment at the given radius
        # If the radius falls outside of the molecule (i.e. it does not reach any atom) then
        # the list of bonds will be empty. In such a case, we reduce the radius until we find
        # a non-empty environment, or we reach radius 0 (which means the radius itself).
        if len(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)) > 0:
            break

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
    if use_smarts:
        signature = frag_to_smarts(mol, atoms, bonds, root_atom=atom.GetIdx(), **kwargs)
    else:
        signature = frag_to_smiles(mol, atoms, bonds, root_atom=atom.GetIdx(), **kwargs)

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
    # Special case for dummies
    if atom.GetAtomicNum() == 0:
        return "*"

    _symbol = atom.GetSymbol()
    _total_h_count = atom.GetTotalNumHs()  # Total number of Hs, including implicit Hs
    _connectivity = atom.GetTotalDegree()  # Missleading naming, but it's the total number of connections, including H
    _degree = atom.GetDegree()  # Number of explicit connections, hence excluding H if Hs are not explicit
    # _non_h_degree = _connectivity - _total_h_count
    _formal_charge = atom.GetFormalCharge()

    # Refine the symbol
    if atom.GetIsAromatic():
        _symbol = _symbol.lower()

    # Assemble the SMARTS
    smarts = f"[{_symbol}"
    if _total_h_count == 1:
        smarts += "H"
    else:
        smarts += f"H{_total_h_count}"
    if _degree == 1:
        smarts += ";D"
    else:
        smarts += f";D{_degree}"
    if _connectivity == 1:
        smarts += ";X"
    else:
        smarts += f";X{_connectivity}"
    if _formal_charge > 0:
        if _formal_charge == 1:
            smarts += ";+"
        else:
            smarts += f";+{_formal_charge}"
    elif _formal_charge < 0:
        if _formal_charge == -1:
            smarts += ";-"
        else:
            smarts += f";-{_formal_charge}"
    if atom_map != 0:
        smarts += f":{atom_map}"
    smarts += "]"

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
        isomericSmiles=kwargs.get("isomericSmiles", True),
        allBondsExplicit=kwargs.get("allBondsExplicit", True),
        allHsExplicit=kwargs.get("allHsExplicit", False),
        kekuleSmiles=kwargs.get("kekuleSmiles", False),
        canonical=True,
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
        isomericSmiles=kwargs.get("isomericSmiles", True),
        allBondsExplicit=kwargs.get("allBondsExplicit", True),
        allHsExplicit=kwargs.get("allHsExplicit", False),
        kekuleSmiles=kwargs.get("kekuleSmiles", False),
        canonical=True,
        rootedAtAtom=root_atom,
    )

    # Chem.MolFragmentToSmiles canonicalizes the rooted fragment but does not do the job properly. To overcome
    # the issue the atom is mapped to 1, and the smiles is canonicalized via Chem.MolToSmiles().
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
    """Class representing the signature of a molecule.

    The signature of a molecule is composed of the signature of its atoms.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        radius: int = 2,
        neighbor: bool = False,
        use_smarts: bool = False,
        boundary_bonds: bool = True,
        map_root: bool = True,
        rooted_smiles: bool = False,
        nbits: int = 0,
        **kwargs: dict,
    ) -> None:
        """Initialize the MoleculeSignature object

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to generate the signature for.
        radius : int
            The radius of the environment to consider.
        neighbor : bool
            Whether to include the neighbors in the signature.
        use_smarts : bool
            Whether to use SMARTS syntax for the signature (otherwise, use SMILES syntax).
        boundary_bonds : bool
            Whether to add bonds at the boundary of the radius.
        map_root : bool
            Whether to map root atoms in atom signatures. If yes, root atoms are labeled as 1.
        rooted_smiles : bool
            Whether to use rooted SMILES syntax within atom signatures. If yes, SMILES are rooted at the root atoms.
        nbits : int
            The number of bits to use for the Morgan fingerprint. If 0, no Morgan fingerprint is computed.
        **kwargs
            Additional arguments to pass to Chem.MolFragmentToSmiles calls.
        """
        # Deprecation warnings
        if "nBits" in kwargs:
            logger.warning("nBits is deprecated, use nbits instead.")
            nbits = kwargs.pop("nBits")

        # Check arguments
        if mol is None:
            raise ValueError("Molecule is None")

        # Parameters reminder
        self.radius = radius
        self.neighbor = neighbor
        self.use_smarts = use_smarts
        self.boundary_bonds = boundary_bonds
        self.map_root = map_root
        self.rooted_smiles = rooted_smiles
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

        # Compute the signatures of all atoms
        for atom in mol.GetAtoms():
            # # Skip hydrogens
            # if atom.GetAtomicNum() == 1 and atom.GetFormalCharge() == 0:
            #     continue

            # Collect non-empty atom signatures
            _sig = AtomSignature(
                atom=atom,
                radius=self.radius,
                use_smarts=self.use_smarts,
                boundary_bonds=self.boundary_bonds,
                map_root=self.map_root,
                rooted_smiles=self.rooted_smiles,
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
        _ += f"atom_signatures={self.atom_signatures}, "
        _ += f"radius={self.radius}, "
        _ += f"neighbor={self.neighbor}, "
        _ += f"use_smarts={self.use_smarts}, "
        _ += f"boundary_bonds={self.boundary_bonds}, "
        _ += f"map_root={self.map_root}, "
        _ += f"rooted_smiles={self.rooted_smiles}, "
        _ += f"nbits={self.nbits}, "
        _ += f"kwargs={self.kwargs}, "
        _ += ")"
        return _

    def __len__(self) -> int:
        return len(self.atom_signatures)

    def __str__(self) -> str:
        return self.as_str()

    def __eq__(self, other) -> bool:
        if not isinstance(other, MoleculeSignature):
            return False
        return (
            self.atoms == other.atoms
            and self.atoms_minus == other.atoms_minus
            and self.neighbors == other.neighbors
            and self.morgans == other.morgans
        )

    def as_deprecated_string(self, morgan=True, root=True, neighbors=True) -> str:
        """Return the signature in the deprecated string format.

        Parameters
        ----------
        morgan : bool
            Whether to include the Morgan bits in the string.
        neighbors : bool
            Whether to include the neighbors in the string.

        Returns
        -------
        str
            The signature in the deprecated string format.
        """
        return " ".join(sorted(atom.as_deprecated_string(morgan, root, neighbors) for atom in self.atom_signatures))

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
        """Return the signature as a list of features.

        If neighbors is False, the atom signature is used. If neighbors is True, the atom signature is used at
        a radius - 1, followed by the atom signature of the neighbor at radius - 1.

        Parameters
        ----------
        morgan : bool
            Whether to include the Morgan bits in the list.
        neighbors : bool
            Whether to include the neighbors in the list.

        Returns
        -------
        list
            The list of features
        """
        out = []
        for _morgan, _atom, _atom_minus, _neighbors in sorted(
            zip(
                self.morgans,
                self.atoms,
                self.atoms_minus,
                self.neighbors,
            ),
            key=lambda x: (x[0], x[1], x[2], x[3]),
        ):
            if morgan:
                s = f"{str(_morgan)}, "
            else:
                s = ""
            if neighbors and len(_neighbors):
                s += f"{_atom_minus} || {_neighbors}"
            else:
                s += _atom
            out.append(s)
        return out

    def as_str(self, morgan=True, neighbors=False) -> str:
        """Return the signature as a string.

        The signature is returned as a string, with each atom signature separated by a double dot surrounded
        by spaces (` .. `).

        If neighbors is False, the atom signature is used. If neighbors is True, the atom signature is used at
        a radius - 1, followed by the atom signature of the neighbor at radius - 1.

        Parameters
        ----------
        morgan : bool
            Whether to include the Morgan bits in the string.
        neighbors : bool
            Whether to include the neighbors in the string.

        Returns
        -------
        str
            The signature as a string
        """
        return " .. ".join(self.as_list(morgan=morgan, neighbors=neighbors))


# =====================================================================================================================
# Overall helper functions
# =====================================================================================================================


def canonical_map(mol: Chem.Mol) -> None:
    """Canonize the atom map numbers of a molecule

    This function canonizes the atom map numbers of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to canonicalize the atom map numbers for.

    Returns
    -------
    None
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeAtomMaps=False))
    for j, i in enumerate(ranks):
        mol.GetAtomWithIdx(j).SetIntProp('molAtomMapNumber', i+1)


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
        if key in [
            "isomericSmiles",
            "allBondsExplicit",
            "allHsExplicit",
            "kekuleSmiles",
            "legacy",
        ]:
            cleaned_kwargs[key] = value
        else:
            logger.warning(f"Invalid argument: {key} (value: {value}), skipping argument.")

    return cleaned_kwargs


if __name__ == "__main__":
    # Example usage
    smiles = "c1ccccc1O"
    mol = Chem.MolFromSmiles(smiles)

    # Compute the molecule signature
    ms = MoleculeSignature(mol, radius=2, neighbor=True, use_smarts=False, boundary_bonds=True, nbits=2048)

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
    arr_radius = [0, 2, 4]
    arr_neighbor = [False, True]
    arr_smarts = [False, True]
    arr_nbits = [0, 2048]
    arr_boundary_bonds = [False, True]
    for use_smarts, neighbor, radius, nbit, boundary_bonds in itertools.product(
        arr_smarts,
        arr_neighbor,
        arr_radius,
        arr_nbits,
        arr_boundary_bonds
    ):
        if boundary_bonds and use_smarts:  # Skip unsupported combinations
            continue
        ms = MoleculeSignature(
            mol, radius=radius, neighbor=neighbor, use_smarts=use_smarts, nbits=nbit, boundary_bonds=boundary_bonds
        )
        # Pretty printings
        print(
            f"Molecule signature (radius={radius}, neighbor={neighbor}, use_smarts={use_smarts}, nbits={nbit}), boundary_bonds={boundary_bonds}:"
        )
        for atom_sig in ms.atom_signatures:
            print(f"├── {atom_sig}")
        print()
