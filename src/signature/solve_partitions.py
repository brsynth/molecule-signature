########################################################################################################################
# This library solve a diophantine system associated to the enumeration
# of signatures from a Morgan vector via a partition method
# Oct. 2023
########################################################################################################################


import collections
import itertools
import math
from functools import reduce
from operator import mul

import numpy as np
import scipy.linalg.blas as blas
from sympy.utilities.iterables import multiset_permutations


########################################################################################################################
# Local functions
########################################################################################################################


def sized_partitions(n, k, m=None):
    """
    Partition n into k parts with a maximum part size of m. Yield non-increasing lists.

    Parameters
    ----------
    n : int
        The integer to be partitioned.
    k : int
        The number of partitions of n.
    m : int, optional
        The maximum part size (default is None).

    Yields
    ------
    list of int
        A partition of n into k parts.
    """

    if k == 0:
        return
    if k == 1:
        yield [n]
        return
    for f in range(n - k + 1 if (m is None or m > n - k + 1) else m, (n - 1) // k, -1):
        for p in sized_partitions(n - f, k - 1, f):
            yield [f] + p


def partitions(n, k):
    """
    Generate partitions of n into at most k parts.

    Parameters
    ----------
    n : int
        The integer to be partitioned.
    k : int
        The number of partitions of n at most

    Returns
    -------
    list of lists of int
        A list of all partitions of n into at most k parts.
    """

    return [part for i in range(1, min(k, n) + 1) for part in sized_partitions(n, i)]


def nb_permutations(l):
    """
    Calculate the number of permutations without repetitions of a list.

    Parameters
    ----------
    l : list
        The input list.

    Returns
    -------
    nb_perm : int
        The number of permutations without repetitions of the list.
    """

    nb_perm = math.factorial(len(l))
    counter = collections.Counter(l)
    for x in counter:
        nb_perm = nb_perm / math.factorial(counter[x])
    return int(nb_perm)


def extract_matrices_C_P(A, b):
    """
    Extract matrices C and P from A and b.

    Parameters
    ----------
    A : np.array
        Matrix.
    b : np.array
        Vector.

    Returns
    -------
    C : np.array
        Matrix C.
    P : np.array
        Matrix P.
    N : np.array
        Vector N.
    parity_indices : list
        List of equations that require parity.
    graph_line : np.array
        Graphicality equation.
    graph_index : int
        Index of the graphicality equation in the matrix C.
    """

    j1 = [i for i in range(len(A[:, 0])) if A[:, 0][i] == 1][-1]
    j2 = [i for i in range(len(A[-1, :])) if A[-1, :][i] == 1][-1] + 1
    P = A[j1:, :j2]
    N = b[-P.shape[0] :]
    C = A[:j1, :]
    parity_indices = [
        sum(C[i, j2:] == [0] * (A.shape[1] - j2)) != (A.shape[1] - j2)
        for i in range(C.shape[0])
    ]
    C = C[:, :j2]
    graph_line = C[-1:, :]
    graph_index = j1 - 1
    return C, P, N, parity_indices, graph_line, graph_index


def partitions_P_N(P, N, max_nbr_partition, bool_timeout):
    """
    Compute partitions of each element of N with respect to P.

    Parameters
    ----------
    P : np.array
        Matrix.
    N : np.array
        Vector.
    max_nbr_partition : int
        Maximum number of partitions.
    bool_timeout : bool
        Timeout flag.

    Returns
    -------
    dict_partitions : dictionary
        Dictionary of the partitions of N with respect to P.
    tups : list of tuples
        List of tuples indicating column indices for each partition.
    bool_timeout : bool
        Timeout flag.
    """

    tups = []
    dict_partitions = {}
    # nb_parts_tot, nb_parts_loc = 0, 0
    for i in range(P.shape[0]):
        tups.append((list(P[i, :]).index(1), list(P[i, :]).index(1) + sum(P[i, :])))
        All_parts = list(partitions(N[i], sum(P[i, :])))
        dict_partitions[i] = []
        for x in All_parts:
            nb_perm = nb_permutations(x + [0] * (sum(P[i, :]) - len(x)))
            if nb_perm > max_nbr_partition:
                bool_timeout = True
                dict_partitions[i] = dict_partitions[i] + list(
                    itertools.islice(
                        multiset_permutations(x + [0] * (sum(P[i, :]) - len(x))),
                        max_nbr_partition,
                    )
                )
                # nb_parts_loc += max_nbr_partition
            else:
                dict_partitions[i] = dict_partitions[i] + list(
                    multiset_permutations(x + [0] * (sum(P[i, :]) - len(x)))
                )
                # nb_parts_loc += nb_perm
            # nb_parts_tot += nb_perm
    # percent_parts_used = 100 * nb_parts_loc / nb_parts_tot
    return dict_partitions, tups, bool_timeout


def equations_trivially_satisfied(C, N, tups, parity_indices, graph_index):
    """
    Identify equations trivially satisfied since constants on partitions.

    Parameters
    ----------
    C : np.array
        Matrix.
    N : np.array
        Vector.
    tups : list of tuples
        List of tuples representing indices of partitions.
    parity_indices : list of bool
        List indicating parity of indices.
    graph_index : int
        Index of the graph.

    Returns
    -------
    indices : list
        Indices of equations not trivially satisfied.
    graph : bool
        Boolean indicating if graphicality is trivially satisfied or not.
    """

    indices = []
    graph = False
    for j in range(C.shape[0]):
        i = 0
        prod = 0
        while i < len(tups) and len(set(C[j, tups[i][0] : tups[i][1]])) == 1:
            prod = prod + N[i] * C[j, tups[i][0] : tups[i][1]][0]
            i = i + 1
        if i == len(tups):
            if j == graph_index:
                graph = prod % 2 == 0
            elif (parity_indices[j] == False and prod != 0) or (
                parity_indices[j] and prod % 2 != 0
            ):
                indices.append(j)
        else:
            indices.append(j)
    return indices, graph


def sort_C_wrt_partitions_involved(C, tups, parity_indices):
    """
    Sort the lines of C following the number of partitions involved per equation.

    Parameters
    ----------
    C : np.array
        Matrix.
    tups : list of tuples
        Tuple of tuples representing indices of partitions.
    parity_indices : list of bool
        List of equations that require parity.

    Returns
    -------
    C : np.array
        Matrix C sorted.
    parity_indices : list
        List of equations that require parity.
    partitions_involved : list of tuples
        List of tuples indicating the partitions involved for each equation.
    """

    partitions_involved = []
    for j in range(C.shape[0]):
        partitions_involved.append(
            tuple(
                [
                    i
                    for i in range(len(tups))
                    if set(C[j, tups[i][0] : tups[i][1]]) != {0}
                ]
            )
        )
    # We sort the lines of C following the number of partitions involved
    indices_tmp = sorted(
        range(len(partitions_involved)), key=lambda k: len(partitions_involved[k])
    )
    C_tmp = C.copy()
    for i in range(C.shape[0]):
        C_tmp[i,] = C[indices_tmp[i],]
    C = C_tmp
    parity_indices = [parity_indices[i] for i in indices_tmp]
    partitions_involved = [partitions_involved[i] for i in indices_tmp]
    return C, parity_indices, partitions_involved


def intersection_of_lists_of_lists(S1, S2):
    """
    Find the intersection of two lists of lists.

    Parameters
    ----------
    S1 : list of lists
        First list of lists.
    S2 : list of lists
        Second list of lists.

    Returns
    -------
    list of lists
        Intersection of the two input lists of lists.
    """

    S_inter = [
        list(x)
        for x in set(tuple(x) for x in S1).intersection(set(tuple(x) for x in S2))
    ]
    return S_inter


def solutions_per_line(
    C,
    tups,
    parity_indices,
    dict_partitions,
    partitions_involved,
    max_nbr_partition,
    bool_timeout,
    verbose=False,
):
    """
    Compute the solutions for each line.

    Parameters
    ----------
    C : np.array
        Matrix.
    tups : list of tuples
        List of tuples representing indices of partitions.
    parity_indices : list of bool
        List of equations that require parity.
    dict_partitions : dict
        Dictionary containing partitions.
    partitions_involved : list
        List of partitions involved.
    max_nbr_partition : int
        Maximum number of partitions.
    bool_timeout : bool
        Boolean indicating timeout.
    verbose : bool, optional
        If True, print verbose output (default is False).

    Returns
    -------
    dict_sols_per_eq : dictionary
        Dictionary containing solutions per line.
    dict_partitions : dictionary
        Updated dictionary of partitions.
    bool_timeout : bool
        Timeout flag.
    """

    dict_sols_per_eq = {}
    for j in range(C.shape[0]):
        parts = partitions_involved[j]
        if verbose:
            print(f"\nLine {j}: \nThe partitions involved are {parts}")
        if parts in dict_sols_per_eq:
            if verbose:
                print(f"These partitions have already been involved")
            l = np.array(dict_sols_per_eq[parts])
        else:
            if verbose:
                print(f"These partitions have not been involved")
            l = []
            for i in range(len(tups)):
                if i in parts:
                    l.append(dict_partitions[i])
                else:
                    l.append([[-1] * len(dict_partitions[i][0])])
            local_bool_timeout = False
            while reduce(mul, [len(x) for x in l]) > max_nbr_partition:  # timeout
                bool_timeout, local_bool_timeout = True, True
                bound = max([len(x) for x in l])
                l = [x[: bound - 1] for x in l]
            if local_bool_timeout:
                print("T I M E O U T.", [len(x) for x in l])
            l = np.array([list(itertools.chain(*x)) for x in itertools.product(*l)])
        if verbose:
            print(f"Local solutions len: {l.shape[0]}")
        if verbose:
            print(f"We restrict local solutions by the current line")
        if parity_indices[j]:
            indices_tmp = np.where(
                blas.dgemm(alpha=1.0, a=l, b=np.transpose(C[j, :])) % 2 == 0
            )[0]
        else:
            indices_tmp = np.where(
                blas.dgemm(alpha=1.0, a=l, b=np.transpose(C[j, :])) == 0
            )[0]
        l = l[indices_tmp, :]
        if l.shape[0] == 0:
            return dict(), dict(), bool_timeout
        if verbose:
            print(f"We clean partitions of dict_partitions by the local solutions")
        for i in parts:
            l_proj = l[:, tups[i][0] : (tups[i][1])].tolist()
            l_proj.sort()
            l_proj = list(k for k, _ in itertools.groupby(l_proj))
            if verbose:
                print(
                    f"part {i}. dict_partitions[part] x S_proj: {len(dict_partitions[i])} x {len(l_proj)}"
                )
            dict_partitions[i] = intersection_of_lists_of_lists(
                dict_partitions[i], l_proj
            )
            if verbose:
                print(f"after inter: {len(dict_partitions[i])}")
        dict_sols_per_eq[parts] = l.tolist()
    return dict_sols_per_eq, dict_partitions, bool_timeout


def clean_local_solutions(tups, dict_sols_per_eq, dict_partitions):
    """
    Clean local solutions stored in dict_sols_per_eq.

    Parameters
    ----------
    tups : list of tuples
        List of tuples representing indices of partitions.
    dict_sols_per_eq : dict
        Dictionary containing solutions per line.
    dict_partitions : dict
        Dictionary containing partitions.

    Returns
    -------
    dict_sols_per_eq : dict
        Dictionary containing cleaned local solutions.
    """

    for parts in dict_sols_per_eq:
        l = []
        for i in range(len(tups)):
            if i in parts:
                l.append(dict_partitions[i])
            else:
                l.append([[-1] * len(dict_partitions[i][0])])
        l = [list(itertools.chain(*x)) for x in itertools.product(*l)]
        dict_sols_per_eq[parts] = intersection_of_lists_of_lists(
            dict_sols_per_eq[parts], l
        )
    return dict_sols_per_eq


def partitions_groups(partitions_involved):
    """
    Compute the groups of partitions involved in solutions.

    This function computes the groups of partitions involved in solutions, where a group is a set of partitions
    that are interconnected through their involvement in solutions.

    Parameters
    ----------
    dict_sols_per_eq : dict
        Dictionary containing the solutions for each line.

    Returns
    -------
    list
        List of sets representing the groups of partitions involved in solutions.
    """

    parts_groups = []
    while len(partitions_involved) > 0:
        group = set(partitions_involved[0])
        partitions_involved.remove(partitions_involved[0])
        find_one = True
        while find_one:
            find_one = False
            i = 0
            while i < len(partitions_involved):
                if len(set.intersection(set(partitions_involved[i]), group)) > 0:
                    group = group | set(partitions_involved[i])
                    partitions_involved.remove(partitions_involved[i])
                    find_one = True
                else:
                    i = i + 1
        parts_groups.append(group)
    return parts_groups


def compatibility(sol1, sol2):
    """
    Check compatibility between two solutions.

    Parameters
    ----------
    sol1 : list
        The first solution.
    sol2 : list
        The second solution.

    Returns
    -------
    bool
        True if the two solutions are compatible, False otherwise.
    """

    i = 0
    while i < len(sol1) and (-1 in [sol1[i], sol2[i]] or sol1[i] == sol2[i]):
        i = i + 1
    return i == len(sol1)


def intersection_of_solutions(S1, S2, test_compatibility):
    """
    Compute the intersection of solutions between groups.

    Parameters
    ----------
    S : list
        List of solutions.
    S_group : list
        List of solutions for a specific group.
    test_compatibility : bool
        If True, test compatibility between solutions.

    Returns
    -------
    S_inter : list of lists
        List of solutions.
    """

    if test_compatibility:
        S_inter = [
            list(np.maximum(sol1, sol2))
            for sol1 in S1
            for sol2 in S2
            if compatibility(sol1, sol2)
        ]
    else:
        S_inter = [list(np.maximum(sol1, sol2)) for sol1 in S1 for sol2 in S2]
    return S_inter


def sort_group_of_partitions(dict_sols_per_eq, group):
    """
    Sort the group of partitions based on the number of solutions per partition.

    Parameters
    ----------
    dict_sols_per_eq : dict
        Dictionary containing the solutions for each equation.
    group : set
        Set representing the group of partitions to be sorted.

    Returns
    -------
    group_sorted : list
        List of partitions sorted based on the number of solutions per partition.
    """

    dtmp = dict()
    for parts in dict_sols_per_eq:
        if parts[0] in group:
            dtmp[parts] = len(dict_sols_per_eq[parts])
    dtmp = {k: v for k, v in sorted(dtmp.items(), key=lambda item: item[1])}
    group_sorted = list(dtmp.keys())
    return group_sorted


def groups_of_solutions(dict_sols_per_eq, parts_groups, verbose=False):
    """
    Compute the product of solutions in each group.

    Parameters
    ----------
    dict_sols_per_eq : dict
        Dictionary containing the solutions for each line of the matrix C.
    parts_groups : list
        List of sets representing the disjoint groups of partitions.
    verbose : bool, optional
        If True, print verbose output (default is False).

    Returns
    -------
    S_groups : list of lists
        List of numpy.ndarray representing the product of solutions in each group.
    """

    S_groups = []
    for group in parts_groups:
        if verbose:
            print(
                f"Group {group}\n{[parts for parts in sorted(dict_sols_per_eq, key=lambda parts: len(dict_sols_per_eq[parts]), reverse=False) if parts[0] in group]}"
            )
        group_sorted = sort_group_of_partitions(dict_sols_per_eq, group)
        if verbose:
            print(f"Group sorted {group_sorted}")
        group_involved = []
        S_group = [[-1] * len(list(dict_sols_per_eq.values())[0][0])]
        while len(group_sorted) > 0:
            find_one = False
            j = 0
            while find_one == False:
                parts = group_sorted[j]
                if (
                    group_involved == []
                    or len(set.intersection(set(parts), set(group_involved))) > 0
                ):
                    find_one = True
                    group_involved = group_involved + list(parts)
                    group_sorted.remove(parts)
                else:
                    j = j + 1
            if verbose:
                print(
                    f"Parts {parts}.\ndict_sols[parts] x S_tmp: {len(dict_sols_per_eq[parts])} x {len(S_group)}"
                )
            S_group = intersection_of_solutions(
                dict_sols_per_eq[parts], S_group, test_compatibility=True
            )
        S_groups.append(S_group)
    return S_groups


########################################################################################################################
# Solve function
########################################################################################################################


def solve_by_partitions(A, b, max_nbr_partition=int(1e5), verbose=False):
    """
    Solve the diophantine system AX=b on N with the partitions algorithm.

    Parameters
    ----------
    A : np.array
        Matrix of shape (m,n).
    b : np.array
        Vector of shape (m,1).
    max_nbr_partition : int, optional
        A bound integer for the number of partitions to use (default is 100000).
    verbose : bool, optional
        If True, print verbose output (default is False).

    Returns
    -------
    S : np.array
        A list of all the solutions of AX=b.
    bool_timeout : bool
        True if the max_nbr_partition has been reached, False otherwise.
    """

    if 1 == 0:
        print(f"A\n {repr(A)}")
        print(f"b\n {repr(b)}")
    bool_timeout = False
    # We start by extracting the matrices P, C and the N vector
    C, P, N, parity_indices, graph_line, graph_index = extract_matrices_C_P(A, b)
    # We compute the partitions of each element of N wrt P
    if verbose:
        print(f"We compute the partitions of each Ni")
    dict_partitions, tups, bool_timeout = partitions_P_N(
        P, N, max_nbr_partition, bool_timeout
    )
    if verbose:
        for i in range(P.shape[0]):
            print(
                f"Part {i}. Nb parts of {N[i]} in {tups[i][1]-tups[i][0]} parts: {len(dict_partitions[i])}"
            )
        if bool_timeout:
            print("T I M E O U T.")
    # We suppress eqs trivially satisfied since constants on partitions
    indices, graph = equations_trivially_satisfied(
        C, N, tups, parity_indices, graph_index
    )
    if verbose:
        print(
            f"Nb of lines deleted since trivially satisfied: {C.shape[0]-len(indices)}"
        )
        print(f"\nGraphicality satisfied directly? {graph}")
    C = C[indices, :]
    parity_indices = [parity_indices[i] for i in indices]
    # If all equations where trivially satisfied, we can return the solutions
    if C.shape[0] == 0:
        l = [dict_partitions[i] for i in dict_partitions]
        l = [list(itertools.chain(*x)) for x in itertools.product(*l)]
        S = np.array(l)
        if graph == False:
            indices_tmp = np.where(np.dot(S, np.transpose(graph_line)) % 2 == 0)[0]
            S = S[indices_tmp, :]
        # return S, percent_parts_used
        return S, bool_timeout
    # We compute partitions involved in each eq and sort the lines of C following the number of partitions involved
    C, parity_indices, partitions_involved = sort_C_wrt_partitions_involved(
        C, tups, parity_indices
    )
    if verbose:
        print(f"The couple of partitions involved: {partitions_involved} \n")
    # We compute the solutions for each equation in C
    if verbose:
        print(f"We compute the solutions of each equation")
    dict_sols_per_eq, dict_partitions, bool_timeout = solutions_per_line(
        C,
        tups,
        parity_indices,
        dict_partitions,
        partitions_involved,
        max_nbr_partition,
        bool_timeout,
        verbose,
    )
    # If timeout has been reached, it is possible that no solution has been found
    if len(dict_sols_per_eq) == 0:
        return np.array([]), bool_timeout
    # We clean local solutions
    if verbose:
        print(
            f"\nWe clean each local sol with the partitions cleaned during the process"
        )
    dict_sols_per_eq = clean_local_solutions(tups, dict_sols_per_eq, dict_partitions)
    # We add missing partitions
    if verbose:
        print(
            f"All the partitions involved: {set(itertools.chain.from_iterable(partitions_involved))}"
        )
    missing_parts = set(range(len(tups))) - set(
        itertools.chain.from_iterable(partitions_involved)
    )
    for j in missing_parts:
        l = []
        for i in range(len(tups)):
            if i == j:
                l.append(dict_partitions[i])
            else:
                l.append([[-1] * len(dict_partitions[i][0])])
        l = np.array([list(itertools.chain(*x)) for x in itertools.product(*l)])
        dict_sols_per_eq[tuple([j])] = l.tolist()
    if verbose:
        print(f"We add {len(missing_parts)} missing partitions")
    # We compute solutions per group
    partitions_involved = list(dict_sols_per_eq.keys())
    parts_groups = partitions_groups(partitions_involved)
    if verbose:
        print(f"The disjoint groups of partitions: {parts_groups}\n")
    # We compute the product of solutions in each group
    if verbose:
        print(f"We compute solutions per group (compatibility + intersection)")
    S_groups = groups_of_solutions(dict_sols_per_eq, parts_groups, verbose)
    # We compute the product of solutions between groups
    if verbose:
        print(f"\nWe compute sol between the distinct groups (intersection)")
    S = [[-1] * C.shape[1]]
    for S_group in S_groups:
        if verbose:
            print(f"S_tmp x S_group: {len(S)} x {len(S_group)}")
        S = intersection_of_solutions(S, S_group, test_compatibility=False)
        if verbose:
            print(f"after inter: {len(S)}")
    S = np.array(S)
    # We restrict by the graphicality if necessary
    if S.shape[0] != 0 and graph == False:
        indices_tmp = np.where(np.dot(S, np.transpose(graph_line)) % 2 == 0)[0]
        S = S[indices_tmp, :]

    return S, bool_timeout
    # return S, percent_parts_used
