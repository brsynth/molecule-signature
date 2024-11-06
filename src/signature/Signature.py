"""This library compute signature on atoms and molecules using RDKit.

Molecule signature: the signature of a molecule is composed of the signature of
its atoms. Molecule signatures are implemented using MoleculeSignature objects.

Atom signature are represented by a rooted SMILES string (the root is the atom
laleled 1). Atom signatures are implemented as AtomSignature objects.

Below are format examples for the oxygen atom in phenol with radius=2
  - Default (nbits=0)
    C:C(:C)[OH:1]
    here the root is the oxygen atom labeled 1: [OH:1]
  - nBits=2048
    91,C:C(:C)[OH:1]
    91 is the Morgan bit of oxygen computed at radius 2

Atom signature can also be computed using neighborhood. A signature neighbor
(string) is the signature of the atom at radius followed but its signature at
raduis-1 and the atom signatutre of its neighbor computed at radius-1

Example:
signature = C:C(:C)[OH:1]
signature-neighbor = C:C(:C)[OH:1]&C[OH:1].SINGLE|C:[C:1](:C)O
    after token &,  the signature is computed for the root (Oxygen)
    and its neighbor for radius-1, root and neighbor are separated by '.'
    The oxygen atom is linked by a SINGLE bond to a carbon of signature C:[C:1](:C)O

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
  - Philippe Meyer <philippe.meyer@inrae.fr>
"""
import numpy as np
import logging
import re
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Logging settings
logger = logging.getLogger(__name__)

# =================================================================================================
# Atom Signature
# =================================================================================================


class AtomSignature:
    """Class representing the signature of an atom."""

    # Separators defined at the class level
    _BIT_SEP = "-"
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
        morgan_bit: None | int | list[int] = None,
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
        morgan_bit : None | int | list[int]
            The Morgan bit(s) of the atom.
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

        # Refine the Morgan bit
        if isinstance(self._morgan, list):
            self._morgan = tuple(self._morgan)

        # Compute signature of the atom itself
        self._root = self.atom_signature(
            atom,
            radius,
            use_smarts,
            boundary_bonds,
            map_root,
            rooted_smiles,
            **self.kwargs,
        )

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
            if self.root == other.root:
                if self.neighbors == other.neighbors:
                    return False
                return self.neighbors < other.neighbors
            return self.root < other.root
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
    def morgan(self) -> None | int | list[int]:
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

    def to_string(self, neighbors=False, morgans=True) -> str:
        """Return the signature as a string

        Returns
        -------
        str
            The signature as a string
        """
        if self.morgan is None or not morgans:
            _ = ""
        elif isinstance(self._morgan, int):
            _ = f"{self._morgan}{self._MORGAN_SEP}"
        elif isinstance(self._morgan, (list, tuple)):
            _ = f"{self._BIT_SEP.join([str(bit) for bit in self._morgan])}{self._MORGAN_SEP}"
        else:
            raise NotImplementedError("Morgan bit must be an integer or a list of integers")
        if neighbors:
            _ += f"{self._root_minus}{self._NEIG_SEP}"
            _ += self._NEIG_SEP.join([f"{bond}{self._BOND_SEP}{sig}" for bond, sig in self.neighbors])  # noqa
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
            morgan, remaining = parts[0], parts[1]
            if cls._BIT_SEP in morgan:
                morgan = tuple(int(bit) for bit in morgan.split(cls._BIT_SEP))
            else:
                morgan = int(morgan)
        else:
            morgan, remaining = None, parts[0]

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
        instance._morgan = morgan
        instance._root = root
        instance._root_minus = root_minus
        instance._neighbors = neighbors

        return instance

    def to_mol(self) -> Chem.Mol:
        """Return the atom signature as a molecule

        Returns
        -------
        Chem.Mol
            The atom signature as a molecule
        """
        smarts = self.root.replace(";", "")  # RDkit does not like semicolons in SMARTS

        mol = Chem.MolFromSmarts(smarts)

        # Update properties
        if mol.NeedsUpdatePropertyCache():
            mol.UpdatePropertyCache()

        return mol

    @classmethod
    def atom_signature(
        cls,
        atom: Chem.Atom,
        radius: int = 2,
        use_smarts: bool = True,
        boundary_bonds: bool = False,
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
        use_smarts : bool
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

        # Generate atom symbols
        if use_smarts:
            for _atom in mol.GetAtoms():
                _atom_symbol = atom_to_smarts(
                    _atom,
                    atom_map=1 if _atom.GetIdx() == atom.GetIdx() and map_root else 0,
                )
                _atom.SetProp("atom_symbol", _atom_symbol)
        else:
            raise NotImplementedError("SMILES syntax not implemented yet.")

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
                addDummies=True if boundary_bonds else False,
                dummyLabels=[(0, 0) for _ in bonds],  # Do not label the dummies
            )
        else:  # No bonds to cut
            fragmented_mol = mol

        # Retrieve the rooted fragment from amongst all the fragments
        frag_to_mol_atom_mapping = []  # Mapping of atom indexes between original and fragments
        for _frag_idx, _fragment in enumerate(
            Chem.GetMolFrags(
                fragmented_mol,
                asMols=True,
                sanitizeFrags=False,
                fragsMolAtomMapping=frag_to_mol_atom_mapping,
            )
        ):
            if atom.GetIdx() in frag_to_mol_atom_mapping[_frag_idx]:
                fragment = _fragment
                frag_to_mol_atom_mapping = frag_to_mol_atom_mapping[_frag_idx]  # Dirty..
                atom_in_frag_index = frag_to_mol_atom_mapping.index(atom.GetIdx())  # Atom index in the fragment
                break

        if use_smarts:  # Get the SMARTS

            # Set a canonical atom mapping
            if fragment.NeedsUpdatePropertyCache():
                fragment.UpdatePropertyCache(strict=False)

            # Build the SMARTS
            _atoms_to_use = list(range(fragment.GetNumAtoms()))
            _atoms_symbols = [atom.GetProp("atom_symbol") for atom in fragment.GetAtoms()]

            # Set a canonical atom mapping
            if fragment.NeedsUpdatePropertyCache():
                fragment.UpdatePropertyCache(strict=False)
            canonical_map_fragment(fragment, _atoms_to_use, _atoms_symbols)

            # Rebuild the fragment using the computed atom symbols
            _fragment = Chem.RWMol(fragment)
            for _atom_idx in range(_fragment.GetNumAtoms()):
                _atom = _fragment.GetAtomWithIdx(_atom_idx)
                _atom_symbol = _atom.GetProp("atom_symbol")
                _fragment.ReplaceAtom(
                    _atom_idx,
                    Chem.AtomFromSmarts(_atom_symbol),
                    updateLabel=False,
                    preserveProps=False,
                )
                _fragment.GetAtomWithIdx(_atom_idx).SetProp("atom_symbol", _atom_symbol)  # Restore the atom symbol
            fragment = _fragment.GetMol()

            if fragment.NeedsUpdatePropertyCache():
                fragment.UpdatePropertyCache(strict=False)

            # DEBUG
            for idx in range(fragment.GetNumAtoms()):
                _atom = fragment.GetAtomWithIdx(idx)
                logging.debug(
                    f"idx: {_atom.GetIdx():2} "
                    f"symbol: {_atom.GetSymbol():2} "
                    f"map: {_atom.GetAtomMapNum():2} "
                    f"degree: {_atom.GetDegree():1} "
                    f"connec: {_atom.GetTotalDegree():1} "
                    f"arom: {_atom.GetIsAromatic():1} "
                    f"smarts: {_atom.GetSmarts():20} "
                    f"stored smarts: {_atom.GetProp('atom_symbol'):20}"
                )

            smarts = Chem.MolFragmentToSmiles(
                fragment,
                atomsToUse=_atoms_to_use,
                atomSymbols=_atoms_symbols,
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

    @classmethod
    def atom_signature_neighbors(
        cls,
        atom: Chem.Atom,
        radius: int = 1,
        use_smarts: bool = True,
        boundary_bonds: bool = False,
        map_root: bool = True,
        rooted_smiles: bool = False,
        **kwargs: dict,
    ) -> List[Tuple[str, str]]:
        """Compute the « with neighbors » signature flavor fo an atom

        Parameters
        ----------
        atom : Chem.Atom
            The root atom to consider.
        radius : int
            The radius of the environment to consider.

        Returns
        -------
        None
        """
        neighbors = []
        for neighbor_atom in atom.GetNeighbors():
            neighbor_sig = cls.atom_signature(
                neighbor_atom,
                radius,
                use_smarts,
                boundary_bonds,
                map_root,
                rooted_smiles,
                **kwargs,
            )

            assert neighbor_sig != "", "Empty signature for neighbor"

            bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
            neighbors.append(
                (str(bond.GetBondType()), neighbor_sig)
            )
        neighbors.sort()

        return neighbors

    def post_compute_neighbors(self, radius=2):
        """Compute the neighbors signature of the atom from the signature of the root atom

        Parameters
        ----------
        radius : int
            The radius of be used (usually radius - 1 compared to the root signature)

        Returns
        -------
        None
        """
        # Get the corresponding molecule
        mol = self.to_mol()

        # Get the root atom
        for _atom in mol.GetAtoms():
            if _atom.GetAtomMapNum() == 1:
                root_atom = _atom
                break

        # Compute the root signature at radius
        self._root_minus = self.atom_signature(
            root_atom,
            radius - 1,
        )

        # Compute the neighbors signatures at radius - 1
        self._neighbors = self.atom_signature_neighbors(
            root_atom,
            radius - 1,
        )

# =================================================================================================
# Atom Signature Helper Functions
# =================================================================================================


def get_smarts_features(qatom: Chem.Atom, wish_list=None) -> dict:
    """Get the features of a SMARTS query atom

    Parameters
    ----------
    qatom : Chem.Atom
        The SMARTS query atom
    wish_list : list
        The list of features to extract. If None, all features are extracted. The list of features is:
        - # : Atomic number
        - A : Aliphatic
        - a : Aromatic
        - H : Number of hydrogens
        - D : Degree
        - X : Connectivity
        - +-: Charge

    Returns
    -------
    dict
        The features of the SMARTS query atom
    """
    # Get the atom properties from the descriptor
    feats = {}
    _descriptors = qatom.DescribeQuery()

    if wish_list is None:
        wish_list = ["#", "A", "a", "H", "D", "X", "+-"]

    # Atomic number and atom type
    if "#" in wish_list:

        # Atomic number
        _match = re.search(r"AtomAtomicNum (?P<value>\d+) = val", _descriptors)
        if _match:
            feats["#"] = int(_match.group("value"))

        # Atom Type (see getAtomListQueryVals from rdkit/Code/GraphMol/QueryOps.cpp)
        _match = re.search(r"AtomType (?P<value>\d+) = val", _descriptors)
        if _match:
            if int(_match.group("value")) > 1000:  # Atom is aromatic
                _atom_number = int(_match.group("value")) - 1000
                if "#" in feats:
                    assert feats["#"] == _atom_number
                feats["#"] = _atom_number
                feats["a"] = 1
            else:  # Atom is aliphatic
                _atom_number = int(_match.group("value"))
                if "#" in feats:
                    assert feats["#"] == _atom_number
                feats["#"] = int(_match.group("value"))
                feats["A"] = 1

    # Hydrogens
    if "H" in wish_list:
        _match = re.search(r"AtomHCount (?P<value>\d) = val", _descriptors)
        if _match:
            feats["H"] = int(_match.group("value"))

    # # Aromatic
    if "a" in wish_list and "a" not in feats:
        _match = re.search(r"AtomIsAromatic (?P<value>\d) = val", _descriptors)
        if _match:
            feats["a"] = int(_match.group("value"))

    # Aliphatic
    if "A" in wish_list and "A" not in feats:
        _match = re.search(r"AtomIsAliphatic (?P<value>\d) = val", _descriptors)
        if _match:
            feats["A"] = int(_match.group("value"))

    # Degree
    if "D" in wish_list:
        _match = re.search(r"AtomExplicitDegree (?P<value>\d) = val", _descriptors)
        if _match:
            feats["D"] = int(_match.group("value"))

    # Connectivity
    if "X" in wish_list:
        _match = re.search(r"AtomTotalDegree (?P<value>\d) = val", _descriptors)
        if _match:
            feats["X"] = int(_match.group("value"))

    # Charge
    if "+-" in wish_list:
        _match = re.search(r"AtomFormalCharge (?P<value>-?\d) = val", _descriptors)
        if _match:
            feats["+-"] = int(_match.group("value"))

    return feats


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

    _PROP_SEP = ";"

    # Directly get features from the query atom
    if isinstance(atom, Chem.QueryAtom):
        feats = get_smarts_features(atom)
        _number = feats.get("#", 0)
        _symbol = Chem.GetPeriodicTable().GetElementSymbol(_number)
        _H_count = feats.get("H", 0)
        _connectivity = feats.get("X", 0)
        _degree = feats.get("D", 0)
        _formal_charge = feats.get("+-", 0)
        # _is_aromatic = feats.get("a", 0)
    else:
        _symbol = atom.GetSymbol()
        _H_count = atom.GetTotalNumHs()  # Total number of Hs, including implicit Hs
        _connectivity = atom.GetTotalDegree()  # Total number of connections, including H
        _degree = atom.GetDegree()  # Number of explicit connections, hence excluding H if Hs are implicits
        _formal_charge = atom.GetFormalCharge()
        # _is_aromatic = atom.GetIsAromatic()

    # Special case for dummies
    if atom.GetAtomicNum() == 0:
        return "*"

    # Refine symbols
    if atom.GetIsAromatic():
        _symbol = _symbol.lower()
    elif atom.GetAtomicNum() == 1:
        _symbol = "#1"  # otherwise, H is not recognized

    # Assemble the SMARTS
    smarts = f"[{_symbol}"
    smarts += f"{_PROP_SEP}H{_H_count}"
    smarts += f"{_PROP_SEP}D{_degree}"
    smarts += f"{_PROP_SEP}X{_connectivity}"
    # if _is_aromatic:
    #     smarts += f"{_PROP_SEP}a"
    # else:
    #     smarts += f"{_PROP_SEP}A"
    if _formal_charge > 0:
        if _formal_charge == 1:
            smarts += f"{_PROP_SEP}+"
        else:
            smarts += f"{_PROP_SEP}+{_formal_charge}"
    elif _formal_charge < 0:
        if _formal_charge == -1:
            smarts += f"{_PROP_SEP}-"
        else:
            smarts += f"{_PROP_SEP}-{abs(_formal_charge)}"
    if atom_map != 0:
        smarts += f":{atom_map}"
    smarts += "]"

    return smarts


def canonical_map_fragment(
    mol: Chem.Mol,
    atoms_to_use: list,
    atoms_symbols: list = None,
) -> None:
    """Canonize the atom map numbers of a molecule fragment

    This function canonizes the atom map numbers of a molecule fragment.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to canonicalize the atom map numbers for.
    atoms_to_use : list
        The list of atom indexes to use in the fragment.

    Returns
    -------
    None
    """
    ranks = list(
        Chem.CanonicalRankAtomsInFragment(
            mol,
            atomsToUse=atoms_to_use,
            atomSymbols=atoms_symbols,
            includeAtomMaps=False
        )
    )
    for j, i in enumerate(ranks):
        if j in atoms_to_use:
            mol.GetAtomWithIdx(j).SetIntProp('molAtomMapNumber', i+1)


# =================================================================================================
# Molecule Signature
# =================================================================================================


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
        all_bits: bool = True,
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
        all_bits : bool
            Whether to use all the bits of the Morgan fingerprint within the atom signatures.
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

            if all_bits:
                # Prepare recipient to collect bits information
                bits_info = rdFingerprintGenerator.AdditionalOutput()
                bits_info.AllocateAtomToBits()

                # Compute Morgan bits
                rdFingerprintGenerator.GetMorganGenerator(
                    radius=radius,
                    fpSize=nbits,
                ).GetFingerprint(mol, additionalOutput=bits_info)

                # Get the Morgan bits per atom
                morgan_vect = bits_info.GetAtomToBits()

            else:
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
                morgan_vect = morgan_vect.tolist()

        else:
            morgan_vect = [None] * mol.GetNumAtoms()

        # Compute the signatures of all atoms
        for atom in mol.GetAtoms():
            _sig = AtomSignature(
                atom,
                radius,
                use_smarts,
                boundary_bonds,
                map_root,
                rooted_smiles,
                morgan_bit=morgan_vect[atom.GetIdx()],
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
            and self.root_minus == other.root_minus
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

    def to_list(self, neighbors=False, morgans=True) -> list:
        """Return the signature as a list of features.

        If neighbors is False, the signature of the root atum at full radius is
        used. If neighbors is True, the signature of the root atom at radius - 1
        is used, followed by the atom signature of the neighbors at radius - 1.

        Parameters
        ----------
        neighbors : bool
            Whether to include the neighbors in the list.

        Returns
        -------
        list
            The signature as a list
        """
        return [atom.to_string(neighbors=neighbors, morgans=morgans) for atom in self._atoms]

    def to_string(self, neighbors=False, morgans=True) -> str:
        """Return the signature as a string.

        If neighbors is False, the signature of the root atum at full radius is
        used. If neighbors is True, the signature of the root atom at radius - 1
        is used, followed by the atom signature of the neighbors at radius - 1.

        Parameters
        ----------
        neighbors : bool
            Whether to include the neighbors in the string.

        Returns
        -------
        str
            The signature as a string
        """
        return self._ATOM_SEP.join(self.to_list(neighbors=neighbors, morgans=morgans))

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

    def post_compute_neighbors(self, radius=2):
        """Compute the neighbors signature of the atoms from the signature of the root atom

        Parameters
        ----------
        signatures : list
            The list of atom signatures
        radius : int
            The radius of be used (usually radius - 1 compared to the root signature)

        Returns
        -------
        None
        """
        [_atom.post_compute_neighbors(radius=radius) for _atom in self._atoms]


# =================================================================================================
# Overall helper functions
# =================================================================================================


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
    ms = MoleculeSignature(mol, radius=2, neighbor=True, use_smarts=False, boundary_bonds=True, nbits=2048)  # noqa

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
