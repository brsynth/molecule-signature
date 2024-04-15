###############################################################################
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
###############################################################################

from .imports import *

def SignatureBondType(bt='UNSPECIFIED'):
    # Callable function
    # Necessary because RDKit functions
    # GetBondType (string) != Chem.BondType (RDKit object)
    # Must be updated with new RDKit release !!!
    BondType = {
    'UNSPECIFIED': Chem.BondType.UNSPECIFIED,
    'SINGLE':  Chem.BondType.SINGLE,
    'DOUBLE':  Chem.BondType.DOUBLE,
    'TRIPLE':  Chem.BondType.TRIPLE,
    'QUADRUPLE':  Chem.BondType.QUADRUPLE,
    'QUINTUPLE':  Chem.BondType.QUINTUPLE,
    'HEXTUPLE':  Chem.BondType.HEXTUPLE,
    'ONEANDAHALF':  Chem.BondType.ONEANDAHALF,
    'TWOANDAHALF':  Chem.BondType.TWOANDAHALF,
    'THREEANDAHALF':  Chem.BondType.THREEANDAHALF,
    'FOURANDAHALF':  Chem.BondType.FOURANDAHALF,
    'FIVEANDAHALF':  Chem.BondType.FIVEANDAHALF,
    'AROMATIC':  Chem.BondType.AROMATIC,
    'IONIC':  Chem.BondType.IONIC,
    'HYDROGEN':  Chem.BondType.HYDROGEN,
    'THREECENTER':  Chem.BondType.THREECENTER,
    'DATIVEONE':  Chem.BondType.DATIVEONE,
    'DATIVE':  Chem.BondType.DATIVE,
    'DATIVEL':  Chem.BondType.DATIVEL,
    'DATIVER':  Chem.BondType.DATIVER,
    'OTHER':  Chem.BondType.OTHER,
    'ZERO':  Chem.BondType.ZERO }
    return BondType[bt]

def AtomSignature(atm,
                  radius=2,
                  isomericSmiles=False,
                  allHsExplicit=False,
                  verbose=False):
    # Local function
    # ARGUMENTS:
    # cf. GetMoleculeSignature
    # RETURNS:
    # A signature (SMILES string) where the root has label 1

    signature = ''
    if atm is None:
        return signature  
    if allHsExplicit == False: # one keep charged hydrogen
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
    env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atmidx,useHs=True)
    while len(env) == 0 and radius > 0:
        radius = radius - 1
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atmidx,useHs=True)
    if radius>0:
        atoms=set()
        for bidx in env:
            atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        atomsToUse=list(atoms)
    else:
        atomsToUse = [atmidx]
        env=None
        
    # Now we get to the business of computing the atom signature        
    atm.SetAtomMapNum(1)
    try:
        signature  = Chem.MolFragmentToSmiles(mol,
                                              atomsToUse,bondsToUse=env,
                                              rootedAtAtom=atmidx,
                                              isomericSmiles=isomericSmiles,
                                              kekuleSmiles=True,
                                              canonical=True,
                                              allBondsExplicit=True,
                                              allHsExplicit=allHsExplicit)
        # Chem.MolFragmentToSmiles canonicalizes the rooted fragment 
        # but does not do the job properly.
        # To overcome the issue the atom is mapped to 1, and the smiles 
        # is canonicalized via Chem.MolToSmiles
        signature = Chem.MolFromSmiles(signature)            
        if allHsExplicit:
            signature = Chem.rdmolops.AddHs(signature)
        signature = Chem.MolToSmiles(signature)
        if verbose == 2:
            print(f'signature for {atm.GetIdx()}: {signature}')

    except:
        if verbose:
            print(f'WARNING cannot compute atom signature for: \
atom num: {atmidx} {atm.GetSymbol()} radius: {radius}') 
        signature =  ''
    atm.SetAtomMapNum(0)
        
    return signature

###############################################################################
# Signature Callable functions
###############################################################################

def SanitizeMolecule(mol, 
                     kekuleSmiles=False,
                     allHsExplicit=False,
                     isomericSmiles=False,
                     formalCharge=False,
                     atomMapping=False,
                     verbose=False):
    # Callable function
    # ARGUMENTS:
    # mol: a RDkit mol object
    # kekuleSmiles: if True remove aromaticity.  
    # allHsExplicit: if true, all H counts will be explicitly 
    #                indicated in the output SMILES.
    # isomericSmiles: include information about stereochemistry  
    # formalCharge: if False remove charges
    # atomMapping: if False remove atom map numbers
    # RETURNS:
    # The sanitized molecule and the corresponding smiles
    verbose = False
    try: 
        Chem.SanitizeMol(mol)
    except:
        if verbose: 
            print(f'WARNING SANITIZATION: molecule cannot be sanitized')
        return None, ''

    if kekuleSmiles:    
        try:
            Chem.Kekulize(mol)
        except:
            if verbose:
                print(f'WARNING SANITIZATION: molecule cannot be kekularized')
            return None, ''

    try:
        mol = Chem.RemoveHs(mol)
    except:
        if verbose:
            print(f'WARNING SANITIZATION: hydrogen cannot be removed)')
        return None, ''

    if allHsExplicit:    
        try:
            mol = Chem.rdmolops.AddHs(mol)
        except:
            if verbose:
                print(f'WARNING SANITIZATION: hydrogen cannot be added)')
            return None, ''
    
    if isomericSmiles == False:
        try:
            Chem.RemoveStereochemistry(mol)
        except:
            if verbose:
                print(f'WARNING SANITIZATION: stereochemistry cannot be removed')
            return None, ''

    if formalCharge == False:
        [a.SetFormalCharge(0) for a in mol.GetAtoms()]
        
    if atomMapping == False:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        
    smi = Chem.MolToSmiles(mol)    
    return mol, smi

def GetAtomSignature(atm,
                     radius=2,
                     neighbor=False,
                     isomericSmiles=False,
                     allHsExplicit=False,
                     verbose=False):
    # Callable function
    # ARGUMENTS:
    # cf. GetMoleculeSignature
    #
    # RETURNS:
    # A signature (string) where atom signatures are sorted in lexicographic 
    # order and separated by ' '
    # see GetAtomSignature for atom signatures format

    signature, temp_signature = '', []
    if neighbor and radius < 1:
        return ''
   
    # We compute atom signature for atm
    signature = AtomSignature(atm, radius=radius, 
                              isomericSmiles=isomericSmiles,
                              allHsExplicit=allHsExplicit,
                              verbose=verbose)
    if neighbor == False:
        return signature
        
    # We compute atm signature at radius-1
    radius = radius-1
    s = AtomSignature(atm, radius=radius, 
                      isomericSmiles=isomericSmiles,
                      allHsExplicit=allHsExplicit,
                      verbose=verbose)
    if s == '':
        return ''
    signature = signature + '&' + s
    
    # We compute atom signatures for all neighbor at radius-1
    mol = atm.GetOwningMol()
    atmset = atm.GetNeighbors() 
    sig_neighbor, temp_sig = '', []
    for a in atmset:
        s = AtomSignature(a, radius=radius, 
                          isomericSmiles=isomericSmiles,
                          allHsExplicit=allHsExplicit,
                          verbose=verbose)
        if s != '': 
            bond = mol.GetBondBetweenAtoms(atm.GetIdx(),a.GetIdx())
            s = str(bond.GetBondType()) + '|' + s
            temp_sig.append(s)
            
    if len(temp_sig) < 1:
        return '' # no signature because no neighbor
    temp_sig = sorted(temp_sig)
    sig_neighbor = '.'.join(s for s in temp_sig)  
    signature = signature + '.' + sig_neighbor
        
    return  signature
    
def GetMoleculeSignature(mol, radius=2, 
                         neighbor=False, 
                         nBits=0, 
                         isomericSmiles=False,
                         allHsExplicit=False,
                         verbose=False):
    # Callable function
    # ARGUMENTS:
    # mol: the molecule in rdkit format
    # radius: the raduis of the signature, when radius < 0 the radius is set 
    #         to the size the molecule
    # nBits: number of bits for Morgan bit vector, when = 0 (default)
    #        Morgan bit vector is not computed
    # isomericSmiles: include information about stereochemistry 
    #                 in the SMILES.  
    # allHsExplicit: if true, all H counts will be explicitly 
    #                indicated in the output SMILES.
    # RETURNS:
    # A signature (string) where atom signatures are sorted in lexicographic 
    # order and separated by ' '
    # see GetAtomSignature for atom signatures format

    signature, temp_signature, morgan = '', [], []
    
    # First get radius and Morgan bits for all atoms 
    if nBits:
        bitInfo = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                               nBits=nBits, 
                                               bitInfo=bitInfo,
                                               useChirality=isomericSmiles,
                                               useFeatures=False)
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
        sig = GetAtomSignature(atm,
                               radius=radius,
                               neighbor=neighbor,
                               isomericSmiles=isomericSmiles,
                               allHsExplicit=allHsExplicit,
                               verbose=verbose)
        if sig != '':
            if nBits: # Add morgan bit if any
                sig = str(int(morgan[atm.GetIdx()])) + ',' + sig
            temp_signature.append(sig)
            
    # collect the signature for all atoms
    if len(temp_signature) < 1:
        return signature
    temp_signature = sorted(temp_signature)
    signature = ' '.join(sig for sig in temp_signature) 
    
    return  signature

def SignatureNeighbor(sig):
    # Get rid ot Morgan and regular signature
    L = sig.split(' ')
    for i in range(len(L)):
        s = L[i]
        if ',' in s:
            s = s.split(',')[1]
        if '&' in s:
            s = s.split('&')[1]
        L[i] = s
    L = sorted(L)
    signature = ' '.join(s for s in L)
    return signature
