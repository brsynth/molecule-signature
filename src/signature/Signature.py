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

# Logging settings
logger = logging.getLogger(__name__)

# =====================================================================================================================
# Atom Signature
# =====================================================================================================================


class AtomSignature:
    """Class representing the signature of an atom."""

    # Separators defined at the class level
    _MORGAN_SEP = " ## "
    _NEIG_SEP = " && "
    _BOND_SEP = " <> "

    def __init__(
        self,
        atom: Chem.Atom = None,
        radius: int = 2,
        use_smarts: bool = True,
        boundary_bonds: bool = False,
        map_root: bool = True,
        rooted_smiles: bool = False,
        morgan_bit: int = None,
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
        **kwargs
            Additional arguments to pass to Chem.MolFragmentToSmiles calls.
        """
        # Parameters reminder
        self.kwargs = clean_kwargs(kwargs)

        # Meaningful information
        self._morgan = morgan_bit
        self._root = None
        self._root_minus = None
        self._neighbors = []

        # Early return if the atom is None
        if atom is None:
            return
        else:
            assert isinstance(atom, Chem.Atom), "atom must be a RDKit atom object"

        # Compute signature of the atom itself
        self._root = atom_signature(
            atom,
            radius,
            use_smarts,
            boundary_bonds,
            map_root,
            rooted_smiles,
            **self.kwargs,
        )

        # Compute signature with neighbors
        if radius > 0:
            # Get the signatures of the neighbors at radius - 1
            self._root_minus = atom_signature(
                atom,
                radius - 1,
                use_smarts,
                boundary_bonds,
                map_root,
                rooted_smiles,
                **self.kwargs,
            )

            for neighbor_atom in atom.GetNeighbors():
                neighbor_sig = atom_signature(
                    neighbor_atom,
                    radius - 1,
                    use_smarts,
                    boundary_bonds,
                    map_root,
                    rooted_smiles,
                    **self.kwargs,
                )

                assert neighbor_sig != "", "Empty signature for neighbor"

                bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
                self._neighbors.append(
                    (str(bond.GetBondType()), neighbor_sig)
                )

            self._neighbors.sort()

        elif radius == 0:  # There's nothing to do if radius is 0
            pass

        else:
            raise ValueError("Radius must be a positive integer or zero.")

    def __repr__(self) -> str:
        _ = "AtomSignature("
        _ += f"morgan={self._morgan}, "
        _ += f"root='{self._root}', "
        _ += f"root_minus='{self._root_minus}', "
        _ += f"neighbors={self._neighbors}"
        _ += ")"
        return _

    def __lt__(self, other) -> bool:
        if self.morgan == other.morgan:
            if self.root == other.sig:
                if self.neighbors == other.neighbors:
                    return False
                return self.neighbors < other.neighbors
            return self.root < other.sig
        return self.morgan < other.morgan

    def __eq__(self, other) -> bool:
        # check if the signature are the same type
        if not isinstance(other, AtomSignature):
            return False
        return (
            self.morgan == other.morgan
            and self.root == other.root
            and self.neighbors == other.neighbors
        )

    @property
    def morgan(self) -> int:
        return self._morgan

    @property
    def root(self) -> str:
        return self._root

    @property
    def root_minus(self) -> str:
        return self._root_minus

    @property
    def neighbors(self) -> tuple:
        return self._neighbors

    def to_string(self, neighbors=False) -> str:
        """Return the signature as a string

        Returns
        -------
        str
            The signature as a string
        """
        if self.morgan is not None:
            _ = f"{self._morgan}{AtomSignature._MORGAN_SEP}"
        else:
            _ = ""
        if neighbors:
            _ += f"{self._root_minus}{AtomSignature._NEIG_SEP}"
            _ += AtomSignature._NEIG_SEP.join([f"{bond}{AtomSignature._BOND_SEP}{sig}" for bond, sig in self.neighbors])
        else:
            _ += self._root
        return _

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
            s += f"{self.root}&"
        if neighbors:
            s += self.root_minus
            for bond, sig in self._neighbors:
                s += f".{bond}|{sig}"
        return s

    @classmethod
    def from_string(cls, signature: str) -> None:
        """Initialize the AtomSignature object from a string

        Parameters
        ----------
        signature : str
            The signature as a string
        """
        # Parse the string
        parts = signature.split(cls._MORGAN_SEP)
        if len(parts) == 2:
            morgan_bit, remaining = parts[0], parts[1]
            morgan_bit = int(morgan_bit)
        else:
            morgan_bit, remaining = None, parts[0]

        if cls._NEIG_SEP in remaining:
            root = None
            root_minus, neighbors_str = remaining.split(cls._NEIG_SEP, 1)
            neighbors = [
                (bond_sig.split(cls._BOND_SEP)[0], bond_sig.split(cls._BOND_SEP)[1])
                for bond_sig in neighbors_str.split(cls._NEIG_SEP)
            ]
        else:
            root_minus = None
            root, neighbors = remaining, []

        # Build the AtomSignature instance
        instance = cls()
        instance._morgan = morgan_bit
        instance._root = root
        instance._root_minus = root_minus
        instance._neighbors = neighbors

        return instance

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

    _ATOM_SEP = " .. "  # Separator between atom signatures

    def __init__(
        self,
        mol: Chem.Mol = None,
        radius: int = 2,
        use_smarts: bool = True,
        boundary_bonds: bool = False,
        map_root: bool = True,
        rooted_smiles: bool = False,
        nbits: int = 2048,
        **kwargs: dict,
    ) -> None:
        """Initialize the MoleculeSignature object

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to generate the signature for.
        radius : int
            The radius of the environment to consider.
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
            return
        else:
            assert isinstance(mol, Chem.Mol), "mol must be a RDKit molecule object"

        # Parameters reminder
        self.kwargs = clean_kwargs(kwargs)
        self._atoms = []

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
            _sig = AtomSignature(
                atom,
                radius,
                use_smarts,
                boundary_bonds,
                map_root,
                rooted_smiles,
                morgan_bit=int(morgan_vect[atom.GetIdx()]) if nbits > 0 else None,  # int to avoid numpy.int64 type
                **self.kwargs,
            )
            if _sig != "":  # only collect non-empty signatures
                self._atoms.append(_sig)

        assert len(self._atoms) > 0, "No atom signature found"

        # Sort the atom signatures
        self._atoms.sort()

    def __repr__(self) -> str:
        _ = "MoleculeSignature("
        _ += f"atoms={self.atoms}"
        _ += ")"
        return _

    def __len__(self) -> int:
        return len(self._atoms)

    def __str__(self) -> str:
        return self.to_string()

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
        return " ".join(sorted(atom.as_deprecated_string(morgan, root, neighbors) for atom in self._atoms))

    @property
    def atoms(self) -> list:
        return [atom for atom in self._atoms]

    @property
    def roots(self) -> list:
        return [atom.root for atom in self._atoms]

    @property
    def root_minus(self) -> list:
        return [atom.root_minus for atom in self._atoms]

    @property
    def neighbors(self) -> list:
        return [atom.neighbors for atom in self._atoms]

    @property
    def morgans(self) -> list:
        return [atom.morgan for atom in self._atoms]

    def to_list(self, neighbors=False) -> list:
        """Return the signature as a list of features.

        If neighbors is False, the signature of the root atum at full radius is used. If neighbors is True,
        the signature of the root atom at radius - 1 is used, followed by the atom signature of the neighbors
        at radius - 1.

        Parameters
        ----------
        neighbors : bool
            Whether to include the neighbors in the list.

        Returns
        -------
        list
            The signature as a list
        """
        return [atom.to_string(neighbors=neighbors) for atom in self._atoms]

    def to_string(self, neighbors=False) -> str:
        """Return the signature as a string.

        If neighbors is False, the signature of the root atum at full radius is used. If neighbors is True,
        the signature of the root atom at radius - 1 is used, followed by the atom signature of the neighbors
        at radius - 1.

        Parameters
        ----------
        neighbors : bool
            Whether to include the neighbors in the string.

        Returns
        -------
        str
            The signature as a string
        """
        return self._ATOM_SEP.join(self.to_list(neighbors=neighbors))

    @classmethod
    def from_list(cls, signatures: list) -> None:
        """Initialize the MoleculeSignature object from a list

        Parameters
        ----------
        signatures : list
            The list of signatures as strings
        """
        # Parse the list
        atoms = [AtomSignature.from_string(sig) for sig in signatures]

        # Build the MoleculeSignature instance
        instance = cls()
        instance._atoms = atoms

        return instance

    @classmethod
    def from_string(cls, signature: str) -> None:
        """Initialize the MoleculeSignature object from a string

        Parameters
        ----------
        signature : str
            The signature as a string
        """
        signatures = signature.split(cls._ATOM_SEP)
        return cls.from_list(signatures)


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
    for item in ms.to_list(morgan=True, neighbors=False):
        print(f"- {item}")
    print()

    print("As list (morgan=False, neighbors=True):")
    for item in ms.to_list(morgan=False, neighbors=True):
        print(f"- {item}")
    print()

    print("As string (morgan=True, neighbors=False):")
    print(ms.to_string(morgan=True, neighbors=False))
    print()

    print("As string (morgan=True, neighbors=True):")
    print(ms.to_string(morgan=True, neighbors=True))
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
            f"Molecule signature (radius={radius}, neighbor={neighbor}, use_smarts={use_smarts}, "
            f"nbits={nbit}), boundary_bonds={boundary_bonds}:"
        )
        for atom_sig in ms._atoms:
            print(f"├── {atom_sig}")
        print()
