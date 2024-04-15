###############################################################################
# This library enumerate molecules from signatures or morgan vector
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format 
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Apr. 2023
###############################################################################

# packages
from .imports import *
os.chdir(os.path.dirname(os.path.dirname(__file__)))
from src.enumerate_utils import BondMatrices, ConstraintMatrix, GetConstraintMatrices, UpdateConstraintMatrices
from src.signature import SanitizeMolecule, SignatureBondType, SignatureNeighbor
from src.signature_alphabet import SignatureAlphabetFromMorganBit, SignatureFromSmiles, SignatureVectorToString
from src.solve_partitions import SolveByPartitions

###############################################################################
# MolecularGraph local object used for smiles enumeration from signature
###############################################################################

class MolecularGraph:
    # A local object used to enumerate molecular graphs
    # for atom signatures or molecules
    def __init__(self, 
                 A, # Adjacency matrix
                 B, # Bond matrix
                 SA, # Atom signature
                 Alphabet, # SignatureAlphabet object
                 max_nbr_recursion=1.0e6, # Max nbr of recursion
                 ai=-1, # Current atom nbr used when enumerating signature up
                 max_nbr_solution=float('inf'), # to produce all solutions
                 nbr_component=1 # nbr connected components
                ):
        
        global recursion_timeout
        
        def AtomicNumCharge(sa):
        # return the atomic number of the root of sa
            sa = sa.split('.')[0] # the root
            sa = sa.split(',')[1] if len(sa.split(',')) > 1 else sa
            m = Chem.MolFromSmiles(sa)
            for a in m.GetAtoms():
                if a.GetAtomMapNum() == 1:
                    return a.GetAtomicNum(), a.GetFormalCharge()
            return -1, 0
        
        self.A, self.B, self.SA, self.Alphabet = A, B, SA, Alphabet
        self.max_nbr_solution = max_nbr_solution
        self.M = self.B.shape[1] # number of bounds
        self.K = int(self.B.shape[1] / self.SA.shape[0]) # nbr of bound/atom
        self.ai = ai # current atom for which signature is expanded
        self.nbr_recursion = 0 # Nbr of recursion
        self.max_nbr_recursion = max_nbr_recursion
        self.nbr_component = nbr_component
                
        rdmol = Chem.Mol()
        rdedmol = Chem.EditableMol(rdmol)
        for sa in self.SA:
            num, charge = AtomicNumCharge(sa)
            if num < 1:
                print(sa)
            rdatom = Chem.Atom(num)
            rdatom.SetFormalCharge (int(charge))
            rdedmol.AddAtom(rdatom)
        self.mol = rdedmol
        self.imin, self.imax = 0, self.M
    
    def bondtype(self, i):
        # Get the RDKit bond type for bond i from its signature
        ai = int(i/self.K)
        sai, iai = self.SA[ai], i % self.K
        nai = sai.split('.')[iai+1] # the right neighbor
        return str(nai.split('|')[0])

    def getcomponent(self, ai, CC):
        # Return the set of atoms attached to ai
        CC.add(ai)
        J = np.transpose(np.argwhere(self.A[ai] > 0))[0] 
        for aj in J:
            if aj not in CC: # not yet visited and bonded to ai
                CC = self.getcomponent(aj, CC)
        return CC

    def validbond(self, i, j):
        # Check if bond i, j can be created
        ai, aj = int(i/self.K), int(j/self.K)
        if j < i or self.A[ai, aj]:
            return False
        if self.nbr_component > 1:
            return True 
        # check the bond does not create a saturated component
        self.addbond(i, j)
        I = list(self.getcomponent(ai, set()))
        A = np.copy(self.A[I,:])
        A = A[:,I]
        valid = False
        if A.shape[0] == self.A.shape[0]: 
            # component has all atoms
            valid = True
        else:
            Ad = np.diagonal(A)
            Ab = np.sum(A, axis=1) - Ad
            if np.array_equal(Ad, Ab) == False:
                valid = True # not saturated
        self.removebond(i,j)
        return valid

    def candidatebond(self, i):
        # Search all bonds that can be connected to i 
        # according to self.B (bond matrix)
        if self.B[self.M,i] == 0:
            return [] # The bond is not free
        F = np.multiply(self.B[i], self.B[self.M]) 
        J = np.transpose(np.argwhere(F != 0))[0]
        J = [j for j in J if self.validbond(i, j)]
        np.random.shuffle(J)
        return J
    
    def addbond(self, i, j):
        # add a bond 
        self.B[i,j], self.B[j,i] = 2, 2 # 0: forbiden, 1: candidate, 2: formed
        ai, aj = int(i/self.K), int(j/self.K)
        self.A[ai,aj], self.A[aj,ai] = self.A[ai,aj]+1, self.A[aj,ai]+1
        self.B[self.M,i], self.B[self.M,j] = 0, 0 # i and j not free
        bt = self.bondtype(i)
        self.mol.AddBond(int(ai), int(aj), SignatureBondType(bt))
            
    def removebond(self, i, j):
        # delete a bond
        self.B[i,j], self.B[j,i] = 1, 1 
        ai, aj = int(i/self.K), int(j/self.K)
        self.A[ai,aj], self.A[aj,ai] = self.A[ai,aj]-1, self.A[aj,ai]-1
        self.B[self.M,i], self.B[self.M,j] = 1, 1 
        self.mol.RemoveBond(ai, aj)

    def smiles(self, verbose=False):
        # get smiles with rdkit
        mol = self.mol.GetMol()
        mol, smi = SanitizeMolecule(mol, 
                                    kekuleSmiles=self.Alphabet.kekuleSmiles,
                                    allHsExplicit=self.Alphabet.allHsExplicit,
                                    isomericSmiles=self.Alphabet.isomericSmiles,
                                    formalCharge=self.Alphabet.formalCharge,
                                    atomMapping=self.Alphabet.atomMapping,
                                    verbose=verbose)
        if 1 == 0:
            smis = CorrectionNitrogen(mol)
            return set(smis)
        else:
            return set([smi])
     
    def end(self, i, G, dict_G, node_current, j_current, verbose):
        # check if the enumeration ends
        # Get the smiles corresponding to the molecular graph
        # make sure all atoms are connected
        global recursion_timeout
        if self.nbr_recursion > self.max_nbr_recursion:
            recursion_timeout = True
            if verbose: print(f'recursion exceeded for enumeration')
            return True, set()
        if i < self.imax:
            return False, set()
        # we are at the end all atoms must be saturated
        Ad = np.diagonal(self.A)
        Ab = np.sum(self.A, axis=1) - Ad
        if np.array_equal(Ad, Ab) == False:
            if verbose == True:
                print(f'sol not saturated\nDiag: {Ad}\nBond: {Ab}')
            return True, set() 
        if verbose == 2: print(f'smi sol found at', self.nbr_recursion)
        # get the smiles
        dict_G[node_current][1] = True
        return True, self.smiles(verbose=verbose)

def AtomSignatureMod(sa):
    # Local function
    # return rsa, a modified atom signature where '.' is change by '_'
    # and '|' is change by '§'
    rsa = copy.copy(sa)
    rsa = rsa.replace(".", "_" )
    rsa = rsa.replace("|", "§" )
    return rsa

###############################################################################
# Enumerate Molecules (smiles) from Signature
###############################################################################

def CorrectionNitrogen(mol):
    # Function to correct the valence problem that
    # appears with [nH]
    # ARGUMENTS:
    # mol: a molecule
    # RETURNS:
    # list of smiles

    smii = Chem.MolToSmiles(mol)    
    mols = [mol]
    list_N = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() and atom.GetTotalDegree() != 3:
            list_N.append(atom.GetIdx())
    if len(list_N) > 0:
        lists_atoms_N_to_incr = [list(l) for l in chain.from_iterable(combinations(list_N, r+1) for r in range(len(list_N)))]
        for atoms_N in lists_atoms_N_to_incr:
            new_mol = copy.deepcopy(mol)
            for atom in new_mol.GetAtoms():
                if atom.GetSymbol() == "N" and atom.GetIsAromatic() and atom.GetTotalDegree() != 3 and atom.GetIdx() in atoms_N:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs()+1)
                mols.append(new_mol)
            smii = Chem.MolToSmiles(mol)
    smis = [smii]
    for mol_cur in mols:
        smi = Chem.MolToSmiles(mol_cur)
        smis.append(smi)  
    return smis

def Enumerate(MG, index, G, dict_G, node_previous, j_current, verbose=False):
    # Local function that build a requested number of
    # molecules (in MG.max_nbr_solution)
    # matching the matrices in the molecular graph MG
    # ARGUMENTS:
    # i: the bond number to be connected
    # MG: the molecular graph
    # RETURNS:
    # Sol: a list of smiles

    global recursion_timeout
    # start
    if index < 0:
        index = MG.imin
    MG.nbr_recursion += 1

    # check if the enumeration has already gone through this branch
    nodes_connected = [link[1] for link in G.edges(node_previous)]
    if j_current in [dict_G[node][0] for node in nodes_connected]:
        node_current = int([node for node in nodes_connected if dict_G[node][0]  == j_current][0])
        if dict_G[node_current][1]:
            return set()
    else:
        node_current = int(MG.nbr_recursion)
        G.add_edge(node_previous, node_current)
        dict_G[node_current] = [j_current, False]

    # check if the enumeration has to end
    end, Sol = MG.end(index, G, dict_G, node_current, j_current, verbose=verbose)
    if end:
        return Sol

    # search all bonds that can be attached to i
    J = MG.candidatebond(index)
    if len(J) == 0:
        Sol2 = Enumerate(MG, index+1, G, dict_G, node_current, -1, verbose)
        tmp = []
        for node in [link[1] for link in G.edges(node_current)]:
            tmp.append(dict_G[node][1])
        if False not in tmp:
            dict_G[node_current][1] = True
        return Sol2

    Sol = set()
    for j in J:
        MG.addbond(index, j)
        sol = Enumerate(MG, index+1, G, dict_G, node_current, j, verbose=verbose)
        Sol = Sol | sol
        if MG.nbr_recursion > MG.max_nbr_recursion:
            recursion_timeout = True
            break # time exceeded
        MG.removebond(index, j)

    tmp = []
    for node in [link[1] for link in G.edges(node_current)]:
        tmp.append(dict_G[node][1])
    if False not in tmp:
        dict_G[node_current][1] = True
    
    return Sol

def EnumerateMoleculeFromSignature(sig, Alphabet, smi,
                                   max_nbr_recursion=int(1e5), 
                                   max_nbr_solution=float('inf'),
                                   nbr_component=1,
                                   repeat=1,
                                   verbose=False):
    # Callable function
    # Build a molecule matching a provided signature
    # ARGUMENTS:
    # sig: signature (with neighbor) of a molecule 
    # max_nbr_solution: maximum nbr of solutions returned 
    # max_nbr_recursion: constant used in signature_enumerate
    # nbr_component: nbr connected components
    # RETURNS:
    # The list of smiles
    
    global recursion_timeout
    recursion_timeout = False
    sign = SignatureNeighbor(sig)

    # initialization of the enumeration graph
    G = nx.DiGraph()
    G.add_node(0)
    dict_G = dict()
    dict_G[0] = [-1, False]

    S, nS, n_nS, max_nS = set(), 0, 0, 3
    r = 0
    while r == 0 or (recursion_timeout and r < repeat):
        if verbose: print(f'repeat {r}')
        recursion_timeout = False
        # Get initial molecule
        AS, NAS, Deg, A, B, C = GetConstraintMatrices(sign, 
                                                    unique=False,
                                                    verbose=verbose)
        MG = MolecularGraph(A, B, AS, Alphabet, ai=-1, 
                max_nbr_recursion=(r+1)*max_nbr_recursion, 
                max_nbr_solution=max_nbr_solution)
        MG.nbr_component = float('inf')
        MG.max_nbr_solution = 1
        MG.nbr_recursion = r*max_nbr_recursion
        SMI = Enumerate(MG, -1, G, dict_G, 0, -1, verbose=verbose)
        S = S | set(SMI)
        n_nS = n_nS+1 if len(S) == nS else 0
        if n_nS == max_nS: # no new solutions in max_nS repeats
            break
        nS = len(S)
        r +=1

    # retain solutions having a signature = provided sig 
    Alphabet.nBits = 0
    SMIsig = set()
    for smi in S:
        if smi != '' and '.' not in smi:
            sigsmi, mol, smisig = SignatureFromSmiles(smi, Alphabet, neighbor=True)
            sigsmi = SignatureNeighbor(sigsmi)
            if sigsmi == sig:
                SMIsig.add(smisig)
    if verbose: print(f'retain solutions having a signature = provided sig {len(S)}, {len(SMIsig)}')
    return list(SMIsig), recursion_timeout

###############################################################################
# Enumerate Signatures from Morgan vector
###############################################################################

def SignatureSet(Sig, Occ):
    # Return a set of signature string
    S = set()
    for i in range(Sig.shape[0]): # get rid of Morgan bit
        if ',' in Sig[i]:
            Sig[i] = Sig[i].split(',')[1]
    for i in range(Occ.shape[0]):
        if len(Occ[i]):
            S.add(SignatureVectorToString(Occ[i], Sig))
    return S  

def handle_timeout(sig, frame):
    raise TimeoutError('took too long')

def EnumerateSignatureFromMorgan(morgan, Alphabet, max_nbr_partition=int(1e5), verbose=False):
    # Callable function
    # Compute all possible signature having a the same Morgan vector 
    # than the provided one. Make use of a Python (sympy) diophantine solver
    # ARGUMENTS:
    # morgan: the Morgan vector
    # RETURNS:
    # The list of signature strings matching the Morgan vector
        
    # Get alphabet signatures in AS along with their
    # minimum and maximum occurence numbers
    # randomize the list of indices
    AS, MIN, MAX, IDX, I = {}, {}, {}, {}, 0
    L = np.arange(morgan.shape[0])
    #np.random.shuffle(L)
    for i in list(L):
        if morgan[i] == 0:
            continue
        # get all signature neighbor in Alphabet having MorganBit = i 
        sig = SignatureAlphabetFromMorganBit(i, Alphabet)
        sig = [s.split('&')[1] for s in sig]        
        if verbose: 
            print(f'MorganBit {i}:{int(morgan[i])}, Nbr in alphabet {len(sig)}')
        (maxi, K) = (morgan[i], 1)
        mini = 0 if len(sig) > 1 else maxi
        for j in range(len(sig)):
            for k in range(int(K)):
                AS[I], MIN[I], MAX[I], IDX[I] = sig[j], mini, maxi, i
                I += 1
    # Get Matrices for enumeration
    AS = np.asarray(list(AS.values()))
    IDX = np.asarray(list(IDX.values()))
    MIN = np.asarray(list(MIN.values()))
    MAX = np.asarray(list(MAX.values()))
    Deg = np.asarray([len(AS[i].split('.'))-1 for i in range(AS.shape[0])])
    n1 = AS.shape[0]
    AS, IDX, MIN, MAX, Deg, C, AS, BS = \
    UpdateConstraintMatrices(AS, IDX, MIN, MAX, Deg, verbose=verbose)
    n2 = AS.shape[0]
    if verbose:
        print(f'AS reduction {n1}, {n2}')
    # Get matrix A and vector b for diophantine solver
    A, b, m = C, np.zeros(C.shape[0]), -1
    for i in range(AS.shape[0]):
        mi = IDX[i]  # morgan bit (modification JLF 241903)
        if mi != m:
            A = np.concatenate((A, P), axis=0) if m != -1 else A
            b = np.concatenate((b, [morgan[m]]), axis=0) if m != -1 else b
            P, m = np.zeros(A.shape[1]).reshape(1,A.shape[1]), mi
        P[0,i] = 1
    A = np.concatenate((A, P), axis=0) if m != -1 else A
    b = np.concatenate((b, [morgan[m]]), axis=0) if m != -1 else b    
    A = A.astype("int")
    b = b.astype("int")

    if verbose:
        print(f'A: {A.shape} b: {b.shape}')
    if verbose == 2:
        print(f'A = {A}\nb = {b}')

    # Solve
    if 1 == 0: # diophantine
        A, b = Matrix(A.astype(int)), Matrix(b.astype(int))
        OCC = np.asarray(list(solve(A, b)))
    else:
        st = time.time()
        OCC, bool_timeout = SolveByPartitions(A, b, verbose=verbose, max_nbr_partition=max_nbr_partition) 
        ct_solve = time.time() - st
    if OCC.shape[0] == 0:
        return [], bool_timeout, ct_solve
    OCC = OCC.reshape(OCC.shape[0], OCC.shape[1])
    OCC = OCC[:, :AS.shape[0]]
    Sol = SignatureSet(AS, OCC) 
    return list(Sol), bool_timeout, ct_solve
