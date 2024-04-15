###############################################################################
# This library solve a diophantine system associated to the enumeration
# of signatures from a Morgan vector via a partition method
# Oct. 2023
###############################################################################

# packages
import collections
import itertools
import math
import numpy as np
import pickle
import scipy.linalg.blas as blas

from functools import reduce
from operator import mul
from sympy.utilities.iterables import multiset_permutations

def SizedPartitions(n, k, m=None):
  # Partition n into k parts with a max part of m.
  # Yield non-increasing lists.  m not needed to create generator.
  # ARGUMENTS:
  # n: the integer to be partitioned
  # k: the number of partitions of n
  # RETURNS:
  # a generator of partitions of n in k parts
  if k == 1:
    yield [n]
    return
  for f in range(n-k+1 if (m is None or m > n-k+1) else m, (n-1)//k, -1):
    for p in SizedPartitions(n-f, k-1, f): yield [f] + p

def Partitions(n, k):
    # Partitions of n into at most k parts
    # ARGUMENTS:
    # n: the integer to be partitioned
    # k: the number of partitions of n at most
    # RETURNS:
    # a list of all partitions of n in at most k parts
    return ([part for i in range(1, min(k,n)+1) for part in SizedPartitions(n, i)])

def Compatibility(sol1, sol2):
    # Intern function to solve AX=b
    # Check compatibility between two solutions
    # ARGUMENTS:
    # sol1: a local solution
    # sol2: a local solution
    # RETURNS:
    # True if the two solutions are compatible, False if not compatible
    i = 0
    while i < len(sol1) and (-1 in [sol1[i], sol2[i]] or sol1[i] == sol2[i]):
        i = i+1
    return i == len(sol1)

def NbPerm(l):
    # ARGUMENTS:
    # l: a list
    # RETURNS:
    # The number of permutations without repetitions of the list
    nb_perm = math.factorial(len(l))
    counter = collections.Counter(l)
    for x in counter:
        nb_perm = nb_perm/math.factorial(counter[x])
    return int(nb_perm)

def SolveByPartitions(A, b, max_nbr_partition=100000, verbose=False):
    # Solve the dioph syst AX=b on N with the partitions algorithm
    # ARGUMENTS:
    # A: np.array of shape (m,n)
    # b: np.array of shape (m,1)
    # max_nbr_partition: a bound integer for the number of partitions to use
    # RETURNS:
    # a list of all the solutions of AX=b
    # a boolean that is True if the max_nbr_partition has been reached, False otherwise
    
    bool_timeout = False

    if 1 == 0:
        print("A")
        print(repr(A))
        print("b")
        print(repr(b))

    # We start by extracting the matrices P, C and the N vector
    j1 = [i for i in range(len(A[:, 0])) if A[:, 0][i] == 1][-1]
    j2 = [i for i in range(len(A[-1, :])) if A[-1, :][i] == 1][-1] + 1
    P = A[j1:, :j2]
    N = b[-P.shape[0] :]
    C = A[:j1, :]
    indices_parity = [sum(C[i, j2:] == [0] * (A.shape[1] - j2)) != (A.shape[1] - j2) for i in range(C.shape[0])]
    C = C[:, :j2]
    graph_line = C[-1:, :]

    # We compute the partitions of each element of N wrt P
    if verbose: print(f'We compute the partitions of each Ni')
    tups = []
    dict_partitions = {}
    nb_parts_tot, nb_parts_loc = 0, 0
    for i in range(P.shape[0]):
        tups.append((list(P[i, :]).index(1), list(P[i, :]).index(1) + sum(P[i, :])))
        All_parts = list(Partitions(N[i], sum(P[i, :])))
        dict_partitions[i] = []
        for x in All_parts:
            if NbPerm(x + [0] * (sum(P[i, :]) - len(x))) > max_nbr_partition:
                bool_timeout = True
                dict_partitions[i] = dict_partitions[i] + list(itertools.islice(multiset_permutations(x + [0] * (sum(P[i, :]) - len(x))), max_nbr_partition))
                nb_parts_loc += max_nbr_partition
            else:
                dict_partitions[i] = dict_partitions[i] + list(multiset_permutations(x + [0] * (sum(P[i, :]) - len(x))))
                nb_parts_loc += NbPerm(x + [0] * (sum(P[i, :]) - len(x)))
            nb_parts_tot += NbPerm(x + [0] * (sum(P[i, :]) - len(x)))

        if verbose: print(f'Part {i}. Nb parts of {N[i]} in {tups[i][1]-tups[i][0]} parts: {len(dict_partitions[i])}')
    percent_parts_used = 100 * nb_parts_loc/nb_parts_tot
    if verbose and bool_timeout: print("T I M E O U T.")

    # We suppress eqs trivially satisfied since constants on partitions
    indices = []
    for j in range(C.shape[0]):
        i = 0
        prod = 0
        while i < len(tups) and len(set(C[j, tups[i][0] : tups[i][1]])) == 1:
            prod = prod + N[i] * C[j, tups[i][0] : tups[i][1]][0]
            i = i + 1
        if j == j1-1:
            graph = prod % 2 == 0
            if verbose: print(f'\nGraphicality satisfied directly? {graph}')
        else:
            if i == len(tups):
                if (indices_parity[j] == False and prod != 0) or (indices_parity[j] and prod % 2 != 0):
                    indices.append(j)
            else:
                indices.append(j)
    if verbose: print(f'Nb of lines deleted since trivially satisfied: {C.shape[0]-len(indices)}')
    C = C[indices, :]
    indices_parity = [indices_parity[i] for i in indices]
    
    if C.shape[0] == 0:
        l = [dict_partitions[i] for i in dict_partitions]
        l = [list(itertools.chain(*x)) for x in itertools.product(*l)]
        S = np.array(l)        
        indices_tmp = np.where(np.dot(S, np.transpose(graph_line)) % 2 == 0)[0]
        S = S[indices_tmp, :]
        #return S, percent_parts_used
        return S, bool_timeout
    
    # We compute partitions involved in each eq
    partitions_involved = []
    for j in range(C.shape[0]):
        partitions_involved.append(tuple([i for i in range(len(tups)) if set(C[j, tups[i][0]: tups[i][1]]) != {0}]))
    
    # We sort the lines of C following the number of partitions involved
    indices_tmp = sorted(range(len(partitions_involved)), key=lambda k: len(partitions_involved[k]))
    C_tmp = C.copy()
    for i in range(C.shape[0]):
        C_tmp[i, ] = C[indices_tmp[i], ]
    C = C_tmp
    indices_parity = [indices_parity[i] for i in indices_tmp]
    partitions_involved = [partitions_involved[i] for i in indices_tmp]

    # We clean the partitions where only one parition is involved
    # not necessary since the sorting?
    if 1 == 0:
        if verbose: print(f'We restrict local solutions by the lines where only one partition is involved')
        indices = []
        for j in range(C.shape[0]):
            parts = partitions_involved[j]
            if len(parts) == 1:  
                l = np.array(dict_partitions[parts[0]])
                if verbose: print(f'Parts {parts}, len l: {len(l)}')
                if indices_parity[j] == False:
                    indices_tmp = np.where(blas.dgemm(alpha=1.0, a=l, b=np.transpose(C[j, tups[parts[0]][0]:(tups[parts[0]][1])])) == 0)[0]
                else:
                    indices_tmp = np.where(blas.dgemm(alpha=1.0, a=l, b=np.transpose(C[j, tups[parts[0]][0]:(tups[parts[0]][1])])) % 2 == 0)[0]
                if verbose: print(f'after restriction: {len(indices_tmp)}')
                dict_partitions[parts[0]] = l[indices_tmp, :].tolist()
            else:
                indices.append(j)
        if verbose: print(f'Nb of lines where only one partition is involved: {C.shape[0]-len(indices)}')
        C = C[indices, :]
        indices_parity = [indices_parity[i] for i in indices]           
        partitions_involved = [partitions_involved[i] for i in indices]
            
    if verbose: print(f'The couple of partitions involved: {partitions_involved} \n')
        
    # We compute the solutions for each line
    if verbose: print(f'We compute the solutions of each line')
    dict_sols_par_lignes = {}
    for j in range(C.shape[0]):
        parts = partitions_involved[j]
        if verbose: print(f'\nLine {j}: \nThe partitions involved are {parts}')
        if parts in dict_sols_par_lignes:
            if verbose: print(f'These partitions have already been involved')
            l = np.array(dict_sols_par_lignes[parts])
        else:
            if verbose: print(f'These partitions have not been involved')
            l =  []
            for i in range(len(tups)):
                if i in parts:
                    l.append(dict_partitions[i])
                else:
                    l.append([[-1]*len(dict_partitions[i][0])])
            local_bool_timeout = False
            while reduce(mul, [len(x) for x in l]) > max_nbr_partition: # timeout
                bool_timeout, local_bool_timeout = True, True
                borne = max([len(x) for x in l])
                l = [x[:borne-1] for x in l]
            if local_bool_timeout : print("T I M E O U T.", [len(x) for x in l])
            l = np.array([list(itertools.chain(*x)) for x in itertools.product(*l)])
        if verbose: print(f'Local solutions len: {l.shape[0]}')
        if verbose: print(f'We restrict local solutions by the current line')
        if indices_parity[j]:
            indices_tmp = np.where(blas.dgemm(alpha=1.0, a=l, b=np.transpose(C[j, :])) % 2 == 0)[0]
        else:
            indices_tmp = np.where(blas.dgemm(alpha=1.0, a=l, b=np.transpose(C[j, :])) == 0)[0]
        l = l[indices_tmp, :]
        if l.shape[0] == 0: return np.array([]), bool_timeout

        if verbose: print(f'We clean partitions of dict_partitions by the local solutions')
        for i in parts:
            l_proj = l[:,tups[i][0]:(tups[i][1])].tolist()
            l_proj.sort()
            l_proj = list(k for k,_ in itertools.groupby(l_proj))
            if verbose: print(f'part {i}. dict_partitions[part] x S_proj: {len(dict_partitions[i])} x {len(l_proj)}')
            dict_partitions[i] = [list(x) for x in set(tuple(x) for x in dict_partitions[i]).intersection(set(tuple(x) for x in l_proj))]
            if verbose: print(f'after inter: {len(dict_partitions[i])}')

        dict_sols_par_lignes[parts] = l.tolist()
    
    # We clean local solutions    
    if verbose: print(f'\nWe clean each local sol with the partitions cleaned during the process')
    for parts in dict_sols_par_lignes:
        l =  []
        for i in range(len(tups)):
            if i in parts:
                l.append(dict_partitions[i])
            else:
                l.append([[-1]*len(dict_partitions[i][0])])
        l = [list(itertools.chain(*x)) for x in itertools.product(*l)]
        dict_sols_par_lignes[parts] = [list(x) for x in set(tuple(x) for x in dict_sols_par_lignes[parts]).intersection(set(tuple(x) for x in l))]

    if verbose: print(f'All the partitions involved: {set(itertools.chain.from_iterable(partitions_involved))}')
    missing_parts = set(range(len(tups))) - set(itertools.chain.from_iterable(partitions_involved))
    for j in missing_parts:
        l = []
        for i in range(len(tups)):
            if i == j:
                l.append(dict_partitions[i])
            else:
                l.append([[-1]*len(dict_partitions[i][0])])
        l = np.array([list(itertools.chain(*x)) for x in itertools.product(*l)])
        dict_sols_par_lignes[tuple([j])] = l.tolist()
    if verbose: print(f'We add {len(missing_parts)} missing partitions')
    
    # We compute solutions per group
    partitions_involved = list(dict_sols_par_lignes.keys())
    parts_groups = []
    while len(partitions_involved) > 0 :
        group = set(partitions_involved[0])
        partitions_involved.remove(partitions_involved[0])
        find_one = True
        while find_one:
            find_one = False
            i = 0
            while i < len(partitions_involved) :
                if len(set.intersection(set(partitions_involved[i]), group)) > 0:
                    group = group | set(partitions_involved[i])
                    partitions_involved.remove(partitions_involved[i])
                    find_one = True
                else:
                    i = i+1
        parts_groups.append(group)
    if verbose: print(f'The disjoint groups of partitions: {parts_groups}\n')
                      
    # We compute the product of solutions in each group
    if verbose: print(f'We compute solutions per group (compatibility + intersection)')
    S_groups = []
    for group in parts_groups:
        if verbose: print(f'Group {group}\n{[parts for parts in sorted(dict_sols_par_lignes, key=lambda parts: len(dict_sols_par_lignes[parts]), reverse=False) if parts[0] in group]}')
        S_group = [[-1]*C.shape[1]]
        dtmp = dict()
        for parts in dict_sols_par_lignes:
            if parts[0] in group:
                dtmp[parts] = len(dict_sols_par_lignes[parts])
        dtmp = {k: v for k, v in sorted(dtmp.items(), key=lambda item: item[1])}
        group2 = list(dtmp.keys())
        if verbose: print(f'Group sorted {group2}')
        group_impliq = []
        while len(group2) > 0 :
            find_one = False
            j = 0
            while find_one == False:
                parts = group2[j]
                if group_impliq == [] or len(set.intersection(set(parts), set(group_impliq))) > 0:
                    find_one = True
                    group_impliq = group_impliq + list(parts)
                    group2.remove(parts)
                else: 
                    j = j+1
            if verbose: print(f'Parts {parts}.\ndict_sols[parts] x S_tmp: {len(dict_sols_par_lignes[parts])} x {len(S_group)}')
            S_group = [list(np.maximum(i, k)) for k in dict_sols_par_lignes[parts] for i in S_group if Compatibility(i, k)]
        S_groups.append((S_group))

    # We compute the product of solutions between groups    
    if verbose: print(f'\nWe compute sol between the distinct groups (intersection)')
    S = [[-1]*C.shape[1]]
    for S_group in S_groups:
        if verbose: print(f'S_tmp x S_group: {len(S)} x {len(S_group)}')
        S = [list(np.maximum(i, k)) for k in S_group for i in S]
        if verbose: print(f'after inter: {len(S)}')
    S = np.array(S)
    
    if graph == False:
        indices_tmp = np.where(np.dot(S, np.transpose(graph_line)) % 2 == 0)[0]
        S = S[indices_tmp, :]

    return S, bool_timeout
    #return S, percent_parts_used
