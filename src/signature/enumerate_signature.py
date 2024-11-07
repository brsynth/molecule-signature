########################################################################################################################
# This library enumerate molecules from signatures or morgan vector.
# Signatures must be computed using neighbor = True.
# cf. signature.py for signature format.
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Apr. 2023
########################################################################################################################


import copy
import re
import time
from itertools import chain, combinations

import networkx as nx
import numpy as np
from collections import Counter
from rdkit import Chem
from signature.enumerate_utils import (
    atomic_num_charge,
    get_constraint_matrices,
    signature_bond_type,
    update_constraint_matrices,
)
from signature.Signature import AtomSignature, MoleculeSignature
from signature.signature_alphabet import sanitize_molecule, signature_vector_to_string
from signature.solve_partitions import solve_by_partitions

########################################################################################################################
# MolecularGraph class used for smiles enumeration from signature.
########################################################################################################################


class MolecularGraph:
    """
    This class is used used to enumerate molecular graphs for atom signatures or molecules.
    It provides methods for manipulating the molecular graph and obtaining SMILES representations.

    Attributes
    ----------
    A : numpy.ndarray
        Adjacency matrix representing the molecular graph.
    B : numpy.ndarray
        Bond matrix representing the bonds between atoms.
    SA : numpy.ndarray
        Atom signature array.
    Alphabet : SignatureAlphabet
        SignatureAlphabet object containing signature information.
    max_nbr_recursion : float, optional
        Maximum number of recursions (default is 1e6).
    ai : int, optional
        Current atom number used when enumerating signature up (default is -1).
    max_nbr_solution : float, optional
        Maximum number of solutions to produce (default is float("inf")).
    nbr_component : int, optional
        Number of connected components (default is 1).

    Methods
    -------
    bond_type(i)
        Get the RDKit bond type for the bond at index i.
    get_component(ai, cc)
        Return the set of atoms attached to atom ai.
    valid_bond(i, j)
        Check if bond i, j can be created.
    candidate_bond(i)
        Search all bonds that can be connected to atom i.
    add_bond(i, j)
        Add a bond between atoms i and j.
    remove_bond(i, j)
        Remove a bond between atoms i and j.
    smiles(verbose=False)
        Get the SMILES representation of the molecule.
    end(i, enum_graph_dict, node_current, j_current, verbose)
        Check if the enumeration ends and get the SMILES representation of the molecular graph.
    """

    def __init__(
        self,
        A,  # Adjacency matrix
        B,  # Bond matrix
        SA,  # Atom signature
        Alphabet,  # SignatureAlphabet object
        max_nbr_recursion=1e6,  # Max nbr of recursion
        ai=-1,  # Current atom nbr used when enumerating signature up
        max_nbr_solution=float("inf"),  # to produce all solutions
        nbr_component=1,  # nbr connected components
    ):
        """
        Initialize the MolecularGraph object with the provided parameters.

        Parameters
        ----------
        A : numpy.ndarray
            Adjacency matrix.
        B : numpy.ndarray
            Bond matrix.
        SA : numpy.ndarray
            Atom signature.
        Alphabet : SignatureAlphabet
            SignatureAlphabet object.
        max_nbr_recursion : float, optional
            Maximum number of recursions (default is 1e6).
        ai : int, optional
            Current atom number used when enumerating signature up (default is -1).
        max_nbr_solution : float, optional
            Maximum number of solutions to produce (default is float("inf")).
        nbr_component : int, optional
            Number of connected components (default is 1).
        """

        self.A, self.B, self.SA, self.Alphabet = A, B, SA, Alphabet
        self.max_nbr_solution = max_nbr_solution
        self.M = self.B.shape[1]  # number of bounds
        self.K = int(self.B.shape[1] / self.SA.shape[0])  # nbr of bound/atom
        self.ai = ai  # current atom for which signature is expanded
        self.nbr_recursion = 0  # Nbr of recursion
        self.max_nbr_recursion = max_nbr_recursion
        self.nbr_component = nbr_component
        self.recursion_timeout = False  # True if max_nbr_recursion is reached
        rdmol = Chem.Mol()
        rdedmol = Chem.EditableMol(rdmol)
        for sa in self.SA:
            num, charge = atomic_num_charge(sa, Alphabet.use_smarts)
            if num < 1:
                print(sa)
            rdatom = Chem.Atom(num)
            rdatom.SetFormalCharge(int(charge))
            # Get the root
            sa_root = atom_sig_to_root(sa)
            # Get explicit H, degree and valency values from the root
            h_value, d_value, x_value = get_h_x_d_value_regex(sa_root)
            implicit_hydrogens = x_value - (h_value + d_value) - int(charge)
            # Set explicit and implicit hydrogens
            rdatom.SetNumExplicitHs(h_value)
            if implicit_hydrogens <= 0:
                rdatom.SetNoImplicit(True)
            rdedmol.AddAtom(rdatom)
        self.mol = rdedmol
        self.imin, self.imax = 0, self.M

    def bond_type(self, i):
        """
        Get the RDKit bond type for bond i from its signature.

        Parameters
        ----------
        i : int
            The index of the bond.

        Returns
        -------
        str
            The RDKit bond type for the specified bond.
        """

        ai = int(i / self.K)
        sai, iai = self.SA[ai], i % self.K
        nai = sai.split(" && ")[iai + 1]  # the right neighbor
        return str(nai.split(" <> ")[0])

    def get_component(self, ai, cc):
        """
        Return the set of atoms attached to ai.

        Parameters
        ----------
        ai : int
            The index of the atom.
        cc : set
            The set of atoms.

        Returns
        -------
        cc : set
            The set of atoms attached to atom ai.
        """

        cc.add(ai)
        J = np.transpose(np.argwhere(self.A[ai] > 0))[0]
        for aj in J:
            if aj not in cc:  # not yet visited and bonded to ai
                cc = self.getcomponent(aj, cc)
        return cc

    def valid_bond(self, i, j):
        """
        Check if bond i, j can be created.

        Parameters
        ----------
        i : int
            The index of the first atom.
        j : int
            The index of the second atom.

        Returns
        -------
        valid : bool
            True if bond i, j can be created, False otherwise.
        """

        ai, aj = int(i / self.K), int(j / self.K)
        if j < i or self.A[ai, aj]:
            return False
        if self.nbr_component > 1:
            return True
        # check the bond does not create a saturated component
        self.add_bond(i, j)
        I = list(self.get_component(ai, set()))
        A = np.copy(self.A[I, :])
        A = A[:, I]
        valid = False
        if A.shape[0] == self.A.shape[0]:
            # component has all atoms
            valid = True
        else:
            Ad = np.diagonal(A)
            Ab = np.sum(A, axis=1) - Ad
            if np.array_equal(Ad, Ab) is False:
                valid = True  # not saturated
        self.remove_bond(i, j)
        return valid

    def candidate_bond(self, i):
        """
        Search all bonds that can be connected to atom i according to self.B (bond matrix).

        Parameters
        ----------
        i : int
            The index of the atom for which candidate bonds are searched.

        Returns
        -------
        list of int
            A list of indices representing candidate atoms to which atom i can form bonds.
        """

        if self.B[self.M, i] == 0:
            return []  # The bond is not free
        F = np.multiply(self.B[i], self.B[self.M])
        J = np.transpose(np.argwhere(F != 0))[0]
        J = [j for j in J if self.valid_bond(i, j)]
        np.random.shuffle(J)
        return J

    def add_bond(self, i, j):
        """
        Add a bond between atoms i and j.

        Parameters
        ----------
        i : int
            Index of the first atom.
        j : int
            Index of the second atom.
        """

        self.B[i, j], self.B[j, i] = 2, 2  # 0: forbiden, 1: candidate, 2: formed
        ai, aj = int(i / self.K), int(j / self.K)
        self.A[ai, aj], self.A[aj, ai] = self.A[ai, aj] + 1, self.A[aj, ai] + 1
        self.B[self.M, i], self.B[self.M, j] = 0, 0  # i and j not free
        bt = self.bond_type(i)
        self.mol.AddBond(int(ai), int(aj), signature_bond_type(bt))

    def remove_bond(self, i, j):
        """
        Remove a bond between atoms i and j.

        Parameters
        ----------
        i : int
            Index of the first atom.
        j : int
            Index of the second atom.
        """

        self.B[i, j], self.B[j, i] = 1, 1
        ai, aj = int(i / self.K), int(j / self.K)
        self.A[ai, aj], self.A[aj, ai] = self.A[ai, aj] - 1, self.A[aj, ai] - 1
        self.B[self.M, i], self.B[self.M, j] = 1, 1
        self.mol.RemoveBond(ai, aj)

    def smiles(self, verbose=False):
        """
        Get the SMILES representation of the molecule.

        Parameters
        ----------
        verbose : bool, optional
            If True, print verbose output (default is False).

        Returns
        -------
        set
            A set containing the SMILES representation of the molecule.
        """

        mol = self.mol.GetMol()
        # We correct the [nH] for aromatic nitrogen atoms
        smis = correction_nitrogen(mol)
        smis_san = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            _, smi_san = sanitize_molecule(
                mol,
                kekuleSmiles=self.Alphabet.kekuleSmiles,
                allHsExplicit=self.Alphabet.allHsExplicit,
                isomericSmiles=self.Alphabet.isomericSmiles,
                formalCharge=self.Alphabet.formalCharge,
                atomMapping=self.Alphabet.atomMapping,
                verbose=verbose,
            )
            smis_san.append(smi_san)
        return set(smis_san)

    def end(self, i, enum_graph_dict, node_current, j_current, verbose):
        """
        Check if the enumeration ends and get the SMILES representation of the molecular graph.
        Making sure all atoms are connected.

        Parameters
        ----------
        i : int
            Current index in the enumeration.
        enum_graph_dict : dict
            Dictionary containing additional information about nodes in the graph.
        node_current : int
            Index of the current node in the graph.
        j_current : int
            Index of the current bond in the graph.
        verbose : bool
            If True, print verbose output (default is False).

        Returns
        -------
        boolean
            True if the enumeration ends. False otherwise.
        set of str
            A set containing the SMILES representation(s) of the molecular graph.
        """

        if self.nbr_recursion > self.max_nbr_recursion:
            self.recursion_timeout = True
            if verbose:
                print("recursion exceeded for enumeration")
            return True, set()
        if i < self.imax:
            return False, set()
        # we are at the end all atoms must be saturated
        Ad = np.diagonal(self.A)
        Ab = np.sum(self.A, axis=1) - Ad
        if np.array_equal(Ad, Ab) is False:
            if verbose:
                print(f"sol not saturated\nDiag: {Ad}\nBond: {Ab}")
            return True, set()
        smi = self.smiles(verbose=verbose)
        if verbose == 2:
            print(f"smi sol found at {int(self.nbr_recursion)}: {smi}")
        # get the smiles
        enum_graph_dict[node_current][1] = True
        return True, smi


def atom_sig_to_root(sa):
    """
    Extract the root of an atomic signature.

    Parameters
    ----------
    sa : str
        An atomic signature.

    Returns
    -------
    str
        Root of the atomic signature sa.
    """

    sa = sa.split(" && ")[0]
    for x in sa.split("]"):
        for y in x.split("["):
            if ":1" in y:
                root = "[" + y + "]"
    return root


def get_h_x_d_value_regex(sa):
    """
    Extract the explicit H, degree and valency values from the root.

    Parameters
    ----------
    sa : str
        An atomic signature.

    Returns
    -------
    int
        Explicit H.
    int
        Degree.
    int
        Valency.
    """

    # Use regex to find the pattern
    match = re.search(r'H(\d+)', sa)
    h_value = int(match.group(1))
    match = re.search(r'X(\d+)', sa)
    x_value = int(match.group(1))
    match = re.search(r'D(\d+)', sa)
    d_value = int(match.group(1))
    return h_value, x_value, d_value


########################################################################################################################
# Enumerate molecule(s) (smiles) from signature.
########################################################################################################################


def correction_nitrogen(mol):
    """
    Perform correction of the [nH] valence problems on a molecule by increasing
    the number of explicit hydrogens on aromatic nitrogen atoms.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule object that needs correction.

    Returns
    -------
    smis_san : list of str
        List of canonical SMILES strings after applying all possible corrections.
    """

    smi = Chem.MolToSmiles(mol)
    smis = [smi]
    list_N = []
    # Identify aromatic nitrogen atoms
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetIsAromatic():
            list_N.append(atom.GetIdx())
    # Generate all possible corrections by incrementing explicit hydrogens on identified atoms
    if len(list_N) > 0:
        lists_atoms_N_to_incr = [
            list(l) for l in chain.from_iterable(combinations(list_N, r + 1) for r in range(len(list_N)))
        ]
        for atoms_N in lists_atoms_N_to_incr:
            new_mol = copy.deepcopy(mol)
            for atom in new_mol.GetAtoms():
                if atom.GetIdx() in atoms_N:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            new_mol_smi = Chem.MolToSmiles(new_mol)
            smis.append(new_mol_smi)
    return smis


def save_mol_plot(mol, name):
    """
    Save a molecule as SVG file.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule object.
    """

    # Compute 2D coordinates
    Chem.rdDepictor.Compute2DCoords(mol)
    # Create an SVG drawer
    dr = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    # Set drawing options (optional, here we disable clearing background)
    opts = dr.drawOptions()
    opts.clearBackground = False
    # Draw the molecule
    dr.DrawMolecule(mol)
    dr.FinishDrawing()
    # Get the SVG string
    svg = dr.GetDrawingText()
    # Save the SVG to a file
    output_path = "C:/Users/meyerp/Documents/INRAE/Diophantine/Enumération/github/output/"
    with open(output_path + name + ".svg", "w") as f:
        f.write(svg)
    # Optionally display the SVG (for Jupyter or inline display environments)
    print(svg)  # This would display the raw SVG in a Jupyter environment or output


def enumeration(MG, index, enum_graph, enum_graph_dict, node_previous, j_current, verbose=False, plot_mol=False, len_J=1):
    """
    Local function that build a requested number of molecules (in MG.max_nbr_solution) matching the matrices in the
    molecular graph MG.

    Parameters
    ----------
    MG : MolecularGraph
        The molecular graph.
    index : int
        The bond number to be connected.
    enum_graph : networkx graph
        Graph of the enumeration.
    enum_graph_dict : dictionary
        Dictionary of the enumeration.
    node_previous : int
        Index of the previous node.
    j_current : int
        Index of the current bond.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.
    plot_mol : bool optional
        If True, save the molecule in SVG at each reconstruction step. Defaults to False.
    len_J : int
        Length of the candidate bonds. Only used to know when to save the molecule in SVG. Defaults to 1.
    
    Returns
    -------
    sol : list
        List of smiles.
    """

    if plot_mol and len_J > 0:
        mol_to_plot =  copy.deepcopy(MG.mol.GetMol())
        for atom in mol_to_plot.GetAtoms():
            atom.SetNoImplicit(True)
        save_mol_plot(mol_to_plot, "test" + str(int(MG.nbr_recursion)))
    # start
    if index < 0:
        index = MG.imin
    MG.nbr_recursion += 1
    # check if the enumeration has already gone through this graph branch
    nodes_connected = [link[1] for link in enum_graph.edges(node_previous)]
    if j_current in [enum_graph_dict[node][0] for node in nodes_connected]:
        node_current = int([node for node in nodes_connected if enum_graph_dict[node][0] == j_current][0])
        if enum_graph_dict[node_current][1]:
            return set()
    else:
        node_current = int(MG.nbr_recursion)
        enum_graph.add_edge(node_previous, node_current)
        enum_graph_dict[node_current] = [j_current, False]
    # check if the enumeration has to end
    end, sol = MG.end(index, enum_graph_dict, node_current, j_current, verbose=verbose)
    if end:
        return sol
    # search all bonds that can be attached to i
    J = MG.candidate_bond(index)
    if len(J) == 0:
        Sol2 = enumeration(MG, index + 1, enum_graph, enum_graph_dict, node_current, -1, verbose, plot_mol=plot_mol, len_J=0)
        tmp = []
        for node in [link[1] for link in enum_graph.edges(node_current)]:
            tmp.append(enum_graph_dict[node][1])
        if False not in tmp:
            enum_graph_dict[node_current][1] = True
        return Sol2
    # enumeration through all valid bonds
    sol = set()
    for j in J:
        MG.add_bond(index, j)
        sol2 = enumeration(MG, index + 1, enum_graph, enum_graph_dict, node_current, j, verbose=verbose, plot_mol=plot_mol, len_J=1)
        sol = sol | sol2
        if MG.nbr_recursion > MG.max_nbr_recursion:
            MG.recursion_timeout = True
            break  # time exceeded
        MG.remove_bond(index, j)
    # update the enumeration graph
    tmp = []
    for node in [link[1] for link in enum_graph.edges(node_current)]:
        tmp.append(enum_graph_dict[node][1])
    if False not in tmp:
        enum_graph_dict[node_current][1] = True
    return sol


def atomic_sig_to_smiles(sa):
    """
    Transform a single atom atomic signature into a smiles.

    Parameters
    ----------
    sa : str
        The atomic signature.

    Returns
    -------
    simplified_smiles : str
        SMILES string of the molecule.
    """

    sa_root = atom_sig_to_root(sa)
    rdmol = Chem.Mol()
    rdedmol = Chem.EditableMol(rdmol)
    m = Chem.MolFromSmarts(sa_root)
    for a in m.GetAtoms():
        if a.GetAtomMapNum() == 1:
            formal_charge_str = a.DescribeQuery().split("AtomFormalCharge")[-1].split(" ")[1]
            if len(formal_charge_str) == 0:
                formal_charge = 0
            else:
                formal_charge = int(formal_charge_str)
    num = a.GetAtomicNum()
    rdatom = Chem.Atom(num)
    rdatom.SetFormalCharge(formal_charge)
    h_value = get_h_x_d_value_regex(sa_root)[0] # explicit H
    rdatom.SetNumExplicitHs(h_value)  # Ensure no explicit hydrogens
    rdatom.SetNoImplicit(True)
    rdedmol.AddAtom(rdatom)
    mol = rdedmol.GetMol()
    Chem.SanitizeMol(mol)
    simplified_smiles = Chem.MolToSmiles(mol)
    return simplified_smiles


def enumerate_molecule_from_signature(
    sig,
    Alphabet,
    smi,
    max_nbr_recursion=int(1e5),
    max_nbr_solution=float("inf"),
    repeat=1,
    verbose=False,
):
    """
    Build a molecule matching a provided signature.

    Parameters
    ----------
    sig : str
        The signature (with neighbor) of a molecule.
    Alphabet : object
        The alphabet object.
    smi : str
        The smiles representation of the molecule.
    max_nbr_recursion : int, optional
        Constant used in signature_enumerate. Defaults to 1e5.
    max_nbr_solution : float, optional
        Maximum number of solutions returned. Defaults to infinity.
    nbr_component : int, optional
        Number of connected components. Defaults to 1.
    repeat : int, optional
        Number of repetitions. Defaults to 1.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    list(SMIsig) : list
        The list of smiles representing the molecules matching the provided signature.

    recursion_timeout : bool
        True if recursion timeout occurred, False otherwise.
    """

    recursion_timeout = False
    # If we want to save the plot of the molecule in SVG
    plot_mol = False
    # Handle signature coming from a single atom
    if len(sig) == 1:
        smi = atomic_sig_to_smiles(sig[0])
        return [smi], False
    # initialization of the enumeration graph
    enum_graph = nx.DiGraph()
    enum_graph.add_node(0)
    enum_graph_dict = dict()
    enum_graph_dict[0] = [-1, False]
    # initialization of the solution sets
    S, nS, n_nS, max_nS = set(), 0, 0, 3
    r = 0
    while r == 0 or (recursion_timeout and r < repeat):
        if verbose:
            print(f"repeat {r}")
        # Get initial molecule
        AS, NAS, Deg, A, B, C = get_constraint_matrices(sig, unique=False, verbose=verbose)
        MG = MolecularGraph(
            A,
            B,
            AS,
            Alphabet,
            ai=-1,
            max_nbr_recursion=(r + 1) * max_nbr_recursion,
            max_nbr_solution=max_nbr_solution,
        )
        MG.nbr_component = float("inf")
        MG.max_nbr_solution = 1
        MG.nbr_recursion = r * max_nbr_recursion
        SMI = enumeration(MG, -1, enum_graph, enum_graph_dict, 0, -1, verbose=verbose, plot_mol=plot_mol)
        recursion_timeout = MG.recursion_timeout
        S = S | set(SMI)
        n_nS = n_nS + 1 if len(S) == nS else 0
        # break if no new solutions in max_nS repeats
        if n_nS == max_nS:
            break
        nS = len(S)
        r += 1
    # retain solutions having a signature = provided sig
    Alphabet.nBits = 0
    SMIsig = set()
    S = [smi for smi in S if smi != "" and "." not in smi and Chem.MolFromSmiles(smi) is not None]
    if len(S) > int(max_nbr_solution / 10):
        recursion_timeout = True
    S = S[: int(max_nbr_solution / 10)]
    for smi in S:
        mol = Chem.MolFromSmiles(smi)
        ms = MoleculeSignature(
            mol,
            radius=Alphabet.radius,
            use_smarts=Alphabet.use_smarts,
            nbits=0,
            boundary_bonds=Alphabet.boundary_bonds,
            map_root=True,
        )
        ms.post_compute_neighbors()
        sigsmi = sorted([atom.to_string(neighbors=True) for atom in ms.atoms])
        if sigsmi == sig:
            SMIsig.add(smi)
    if verbose:
        print(f"retain solutions having a signature = provided sig {len(S)}, {len(SMIsig)}")
    return list(SMIsig), recursion_timeout


########################################################################################################################
# Enumerate signature(s) from Morgan vector.
########################################################################################################################


def is_counted_subset(sublist, mainlist):
    # Create Counters for both lists
    counter_sublist = Counter(sublist)
    counter_mainlist = Counter(mainlist)

    # Check if each element in the sublist is in the mainlist with enough count
    for element, count in counter_sublist.items():
        if counter_mainlist[element] < count:
            return False
    return True


def signature_set(AS, occ):
    """
    Generate a set of signature strings based on the provided signature and occurrence arrays.

    Parameters
    ----------
    AS : numpy.ndarray
        An array containing atomic signatures.
    occ : numpy.ndarray
        An array containing occurrences.

    Returns
    -------
    S : set of str
        A set containing unique signature strings.
    """

    sol = []
    for v in occ:
        s = []
        for i in range(len(v)):
            for count in range(v[i]):
                s.append(AS[i])
        sol.append(sorted(s))
    return sol


def enumerate_signature_from_morgan(morgan, Alphabet, max_nbr_partition=int(1e5), verbose=False):
    """
    Compute all possible signatures having the same Morgan vector as the provided one.

    Parameters
    ----------
    morgan : numpy.ndarray
        The Morgan vector.
    Alphabet : dictionary
        The alphabet dictionary.
    max_nbr_partition : int, optional
        Maximum number of partitions. Defaults to 1e5.
    verbose : bool, optional
        If True, print detailed information during the operation. Defaults to False.

    Returns
    -------
    list(sol) : list of str
        The list of signature strings matching the Morgan vector.
    bool_timeout : bool
        True if timeout occurred during solving, False otherwise.
    ct_solve : float
        Time taken to solve the problem.
    """

    morgan_indices = [i for i in range(len(morgan)) for _ in range(morgan[i]) if morgan[i] != 0]
    morgan_indices_unique = list(set(morgan_indices))
    morgan_non_zero = [morgan[i] for i in morgan_indices_unique]
    # Selection of the atomic signatures of the alphabet having morgan bits included in the morgan vector input
    AS, MIN, MAX, IDX, I = {}, {}, {}, {}, 0
    for sig in Alphabet.Dict.keys():
        mbits, sa = sig.split(" ## ")[0], sig.split(" ## ")[1]
        mbits = [int(x) for x in mbits.split("-")]
        if is_counted_subset(mbits, morgan_indices):
            mbit = mbits[-1]
            maxi = morgan[mbit]
            mini = 0 if len(sig) > 1 else maxi
            AS[I], MIN[I], MAX[I], IDX[I] = sa, mini, maxi, mbits
            I += 1
    # Compute neighbors of selected fragments
    for i in range(len(AS)):
        _as = AS[i]
        _as_neigh = AtomSignature.from_string(_as)
        _as_neigh.post_compute_neighbors()
        _as_neigh_string = _as_neigh.to_string(True)
        AS[i] = _as_neigh_string
    # Get Matrices for enumeration
    AS = np.asarray(list(AS.values()))
    MIN = np.asarray(list(MIN.values()))
    MAX = np.asarray(list(MAX.values()))
    Deg = np.asarray([len(AS[i].split(" && ")) - 1 for i in range(AS.shape[0])])
    AS, IDX, MIN, MAX, Deg, C = update_constraint_matrices(AS, IDX, MIN, MAX, Deg, verbose=verbose)
    C = C.astype(int)
    if AS.shape[0] == 0:
        return [], False, 0
    # Creation of the diophantine system
    P = np.zeros((len(morgan_indices_unique), len(AS)), dtype=int)
    for i in range(len(IDX)):
        mbits = IDX[i]
        for mbit in mbits:
            mbit_index = morgan_indices_unique.index(mbit)
            P[mbit_index, i] += 1
    # Solving the diophantine system
    st = time.time()
    S, bool_timeout = solve_by_partitions(P, morgan_non_zero, C, max_nbr_partition=max_nbr_partition, verbose=verbose)
    ct_solve = time.time() - st
    occ = np.array(S)
    if occ.shape[0] == 0:
        return [], bool_timeout, ct_solve
    occ = occ[:, : AS.shape[0]]
    sol = signature_set(AS, occ)
    sol = list(map(list, set(map(tuple, sol))))
    return sol, bool_timeout, ct_solve
