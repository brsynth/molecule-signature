# =================================================================================================
# This library solve a diophantine system associated to the enumeration
# of signatures from a Morgan vector via a partition method
#
# Authors:
#  - Jean-loup Faulon <jfaulon@gmail.com>
#  - Thomas Duigou <thomas.duigou@inrae.fr>
#  - Philippe Meyer <philippe.meyer@inrae.fr>
# =================================================================================================


import collections
import itertools
import math
from collections import Counter

import numpy as np
import scipy.linalg.blas as blas
from sympy.utilities.iterables import multiset_permutations

# =================================================================================================
# Local functions
# =================================================================================================


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

    return [part for i in range(1, min(k, n) + 1) for part in sized_partitions(n, i)][::-1]


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

    S_inter = [list(x) for x in set(tuple(x) for x in S1).intersection(set(tuple(x) for x in S2))]
    return S_inter


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
    S1 : list
        List of solutions.
    S2 : list
        List of solutions.
    test_compatibility : bool
        If True, test compatibility between solutions.

    Returns
    -------
    S_inter : list of lists
        List of solutions.
    """

    if test_compatibility:
        S_inter = [
            list(np.maximum(sol1, sol2)) for sol1 in S1 for sol2 in S2 if compatibility(sol1, sol2)
        ]
    else:
        S_inter = [list(np.maximum(sol1, sol2)) for sol1 in S1 for sol2 in S2]
    return S_inter


def partitions_groups(partitions):
    """
    Compute groups of partitions.

    This function computes the groups of partitions, where a group is a set of partitions that are interconnected through their positions of non-negative numbers.

    Parameters
    ----------
    partitions : list
        List of partitions.

    Returns
    -------
    list
        List of sets representing the groups of partitions.
    """

    parts_groups = []
    while len(partitions) > 0:
        group = set(partitions[0])
        partitions.remove(partitions[0])
        find_one = True
        while find_one:
            find_one = False
            i = 0
            while i < len(partitions):
                if len(set.intersection(set(partitions[i]), group)) > 0:
                    group = group | set(partitions[i])
                    partitions.remove(partitions[i])
                    find_one = True
                else:
                    i = i + 1
        parts_groups.append(group)
    return parts_groups


def partition_to_local_sol(partition, part_line_of_P, nb_lin):
    """
    Map partition identifiers to a local solution list.

    This function takes a list of partition identifiers and a corresponding list
    of indices, and maps them to a local solution list of fixed length. The local
    solution list (`local_sol`) is initialized with -1, and is updated such that
    each position specified in `part_line_of_P` is assigned the corresponding
    partition identifier from `partition`.

    Parameters
    ----------
    partition : list
        List of partition identifiers. Each element represents the partition or
        group assignment of an item.
    part_line_of_P : list
        List of indices mapping each element in `partition` to a specific position
        in the `local_sol` list.
    nb_lin : int
        The total number of lines or elements in the `local_sol` list.

    Returns
    -------
    list
        A list of length `nb_lin` where each element is either the corresponding
        partition identifier from `partition` or -1 if not assigned.
    """

    local_sol = [-1] * nb_lin
    for ind in range(len(part_line_of_P)):
        indice = part_line_of_P[ind]
        part = partition[ind]
        local_sol[indice] = part
    return local_sol


def update_C(C, nb_col, graph=False):
    """
    Update the matrix C by trimming columns and optionally separating the last row for graphicality.

    This function processes the matrix `C` by identifying rows with non-zero elements in
    columns beyond `nb_col`. It then trims `C` to include only the first `nb_col` columns.
    If the `graph` flag is set to True, the last row of `C` is separated out as `graph_line`,
    and its index is returned.

    Parameters
    ----------
    C : numpy.ndarray
        The input matrix.
    nb_col : int
        The number of columns to retain in the matrix.
    graph : bool, optional
        If True, the last row of the matrix is separated and returned along with its index.
        Default is False.

    Returns
    -------
    tuple
        If `graph` is False:
            - C (numpy.ndarray): The trimmed matrix with only the first `nb_col` columns.
            - parity_indices (list): A list of indices of rows in the original matrix that
              contain non-zero elements in the columns beyond `nb_col`.
        If `graph` is True:
            - C (numpy.ndarray): The trimmed matrix without the last row, and with only the first
              `nb_col` columns.
            - parity_indices (list): A list of indices of rows in the original matrix that
              contain non-zero elements in the columns beyond `nb_col`.
            - graph_line (numpy.ndarray): The graphicality row of the original matrix.
            - graph_index (int): The index of the graphicality row in the original matrix.
    """

    parity_indices = [
        i for i in range(C.shape[0]) if all(x == 0 for x in list(C[i, nb_col:])) == False
    ]
    C = C[:, :nb_col]
    if graph:
        graph_line = C[-1:, :]
        graph_index = C.shape[0] - 1
        C = C[:-1, :]
        return C, parity_indices, graph_line, graph_index
    else:
        return C, parity_indices


def restrict_sol_by_one_line_of_C(S, C, i, parity_indices):
    """
    Restrict the solution set S based on constraints defined by a row in matrix C.

    This function filters the rows of the solution set `S` based on a specific row `i`
    of the matrix `C`. The filtering condition depends on whether the index `i` is in
    the `parity_indices` list. If `i` is in `parity_indices`, the function keeps rows
    in `S` that satisfy an even parity condition; otherwise, it keeps rows where the
    matrix product with the row `i` of `C` equals zero.

    Parameters
    ----------
    S : numpy.ndarray or list of numpy.ndarray
        The solution set, represented as a matrix or a list of vectors.
    C : numpy.ndarray
        The matrix containing constraints for filtering `S`.
    i : int
        The index of the row in `C` used to filter the solutions.
    parity_indices : list of int
        A list of indices indicating which rows in `C` correspond to parity constraints.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The filtered solution set, containing only the rows that meet the specified
        conditions.
    """

    if i in parity_indices:
        indices_tmp = np.where(blas.dgemm(alpha=1.0, a=S, b=np.transpose(C[i, :])) % 2 == 0)[0]
    else:
        indices_tmp = np.where(blas.dgemm(alpha=1.0, a=S, b=np.transpose(C[i, :])) == 0)[0]
    S = [S[ind] for ind in indices_tmp]
    return S


def restrict_sol_by_C(
    S,
    part_P,
    C,
    parity_indices,
    partitions_involved_for_C,
    lines_of_C_already_satisfied,
    verbose=False,
):
    """
    Restrict the solution set by applying constraints based on specific partitions.

    This function filters the solution set `S` by applying constraints defined in `C`. It focuses on
    partitions involved in `partitions_involved_for_C` and checks their presence in `part_P`. The
    solution set is updated by restricting it according to these partitions, and the indices of the
    constraints that have been satisfied are tracked.

    Parameters
    ----------
    S : list
        The current solution set that needs to be restricted.
    part_P : tuple
        The partition of the solution set that will be checked against the constraints.
    C : list
        A list of constraints to be applied to the solution set.
    parity_indices : list
        A list of indices indicating the positions of parity checks.
    partitions_involved_for_C : dict
        A dictionary where keys are partition indices and values are tuples representing the partitions
        involved in the constraints.
    lines_of_C_already_satisfied : set
        A set of constraint line indices that have already been satisfied.
    verbose : bool, optional
        If True, the function will print verbose output for debugging purposes. Default is False.

    Returns
    -------
    tuple
        - S (list): The updated solution set after applying the constraints.
        - lines_of_C_already_satisfied (set): The updated set of constraint line indices that have been satisfied.
    """

    all_part_C_to_restrict_on = {}
    for j in partitions_involved_for_C:
        part_C = partitions_involved_for_C[j]
        if len(part_C) == 0:
            continue
        if all(item in part_P for item in part_C):
            all_part_C_to_restrict_on[j] = part_C
    all_part_C_to_restrict_on_sorted = dict(
        sorted(all_part_C_to_restrict_on.items(), key=lambda item: len(item[1]), reverse=True)
    )
    for j in all_part_C_to_restrict_on_sorted:
        part_C = partitions_involved_for_C[j]
        if verbose:
            print("Restriction of the local sol of P with part", part_P, "with the C part", part_C)
            print("Bef rest", len(S))
        S = restrict_sol_by_one_line_of_C(S, C, j, parity_indices)
        if len(S) == 0:
            return [], lines_of_C_already_satisfied
        if verbose:
            print("Aft rest", len(S))
        if part_P == part_C:
            lines_of_C_already_satisfied.add(j)
    return S, lines_of_C_already_satisfied


def partitions_on_non_constant(v, target_sum, max_nbr_partition_non_constant=int(1e3)):
    """
    Generate partitions of a target sum over a vector with constraints on non-constant elements.

    This function generates partitions of a given `target_sum` across the indices of the vector `v`.
    The partitions are constrained such that no subset corresponding to a non-constant section in `v`
    (where elements are not all 1) has uniform values in the partition. It also limits the number of
    unique partitions returned to `max_nbr_partition_non_constant` to avoid excessive computation.

    Parameters
    ----------
    v : list of int
        A vector representing the number of positions available for each index.
        For example, if v = [2, 1, 3], it implies there are 2 positions for the first index, 1 for the second, and 3 for the third.
    target_sum : int
        The target sum that the partitioning should achieve.
    max_nbr_partition_non_constant : int, optional
        The maximum number of partitions to generate for non-constant sections, default is 1000.

    Returns
    -------
    tuple
        - partitions_3 (list of list of int): The list of valid partitions that meet the criteria.
        - bool_threshold_reached_loc (bool): A flag indicating if the partitioning process was truncated due to reaching `max_nbr_partition_non_constant`.
    """

    bool_threshold_reached_loc = False
    if target_sum == 0:
        return [[0] * len(v)]
    v2 = []
    for i in range(len(v)):
        v2 = v2 + [1] * v[i]
    k = len(v2)
    partitions_1 = list(partitions(target_sum, k))
    partitions_1 = [x + [0] * (k - len(x)) for x in partitions_1]

    ind = []
    for i in range(len(partitions_1)):
        x = partitions_1[i]
        x_counter = Counter(x)
        x_max_count = max(x_counter.values())
        if max(v) <= x_max_count:
            ind.append(i)
    partitions_1 = [partitions_1[i] for i in ind]

    partitions_2 = []
    for x in partitions_1:
        nb_perm = nb_permutations(x)
        if nb_perm > max_nbr_partition_non_constant:
            bool_threshold_reached_loc = True
            partitions_2 = partitions_2 + list(
                itertools.islice(
                    multiset_permutations(x),
                    max_nbr_partition_non_constant,
                )
            )
        else:
            partitions_2 = partitions_2 + list(multiset_permutations(x))

    ind = []
    for i in range(len(v)):
        if v[i] != 1:
            for j in range(len(partitions_2)):
                sol = partitions_2[j]
                sol2 = sol[sum(v[:i]) : sum(v[:i]) + v[i]]
                if len(set(sol2)) != 1:
                    ind.append(j)
    ind = list(set(ind))
    partitions_2 = list(filter(lambda x: partitions_2.index(x) not in ind, partitions_2))

    partitions_3 = []
    for sol in partitions_2:
        partitions_3.append([sol[sum(v[:i])] for i in range(len(v))])

    return partitions_3, bool_threshold_reached_loc


def groups_of_solutions(
    dict_sols_per_eq,
    parts_groups,
    C,
    partitions_involved_for_C,
    parity_indices,
    lines_of_C_already_satisfied,
    verbose=False,
):
    """
    Group solutions based on partitions and constraints.

    This function processes solutions for different equations or partitions (`dict_sols_per_eq`)
    by grouping them according to specified partition groups (`parts_groups`). It applies constraints
    from matrix `C`, using the list of involved partitions and parity indices. It also updates the list
    of lines in `C` that have been satisfied. The function returns the grouped solutions and the updated
    list of satisfied lines.

    Parameters
    ----------
    dict_sols_per_eq : dict
        A dictionary where keys are tuples representing partitions and values are lists of solutions
        corresponding to those partitions.
    parts_groups : list of sets
        A list of sets, where each set contains the partitions that form a group.
    C : numpy.ndarray
        The matrix containing constraints that need to be satisfied by the solutions.
    partitions_involved_for_C : list of int
        Indices of partitions involved in matrix `C`.
    parity_indices : list of int
        Indices indicating which rows in `C` correspond to parity constraints.
    lines_of_C_already_satisfied : set of int
        A set of line indices in `C` that have already been satisfied by previous groups of solutions.
    verbose : bool, optional
        If True, print detailed information about the processing of each group. Default is False.

    Returns
    -------
    tuple
        - S_groups (list of list): A list of lists, where each sublist contains the solutions for a specific group.
        - lines_of_C_already_satisfied (set of int): Updated set of line indices in `C` that are satisfied.
    """

    S_groups = []
    for group in parts_groups:
        if verbose:
            print(
                f"Group {group}\nList of parts {[parts for parts in sorted(dict_sols_per_eq, key=lambda parts: len(dict_sols_per_eq[parts]), reverse=False) if parts[0] in group]}"
            )
        dict_sols_per_eq_loc = {}
        for parts in dict_sols_per_eq:
            if parts[0] in group:
                dict_sols_per_eq_loc[parts] = dict_sols_per_eq[parts]
        S_group, lines_of_C_already_satisfied = solution_of_one_group(
            dict_sols_per_eq_loc,
            C,
            partitions_involved_for_C,
            parity_indices,
            lines_of_C_already_satisfied,
            verbose=verbose,
        )
        if len(S_group) == 0:
            return [], []
        S_groups.append(S_group)
    return S_groups, lines_of_C_already_satisfied


def solution_of_one_group(
    dict_sols_per_eq,
    C,
    partitions_involved_for_C,
    parity_indices,
    lines_of_C_already_satisfied,
    verbose=False,
):
    """
    Merge and filter solutions for a group based on compatibility and constraints.

    This function iteratively merges solution sets from `dict_sols_per_eq` that share common elements.
    The merging process continues until a single solution set remains. During each merge, the function
    checks for compatibility between solutions, and applies constraints from the matrix `C` when applicable.
    The lines of `C` that have been satisfied by the merged solutions are tracked.

    Parameters
    ----------
    dict_sols_per_eq : dict
        A dictionary where keys are tuples representing partitions and values are lists of solutions
        corresponding to those partitions.
    C : numpy.ndarray
        The matrix containing constraints that need to be satisfied by the solutions.
    partitions_involved_for_C : list of int
        Indices of partitions involved in matrix `C`.
    parity_indices : list of int
        Indices indicating which rows in `C` correspond to parity constraints.
    lines_of_C_already_satisfied : set of int
        A set of line indices in `C` that have already been satisfied by previous groups of solutions.
    verbose : bool, optional
        If True, print detailed information about the merging and filtering process. Default is False.

    Returns
    -------
    tuple
        - list: The merged and filtered solution set for the group.
        - list: Updated list of line indices in `C` that are satisfied.
    """

    while len(dict_sols_per_eq.keys()) > 1:
        d_prod_len = {}
        for i in range(len(dict_sols_per_eq.keys()) - 1):
            part_i = list(dict_sols_per_eq.keys())[i]
            for j in range(i + 1, len(dict_sols_per_eq.keys())):
                part_j = list(dict_sols_per_eq.keys())[j]
                tup = (part_i, part_j)
                if len(set.intersection(set(part_i), set(part_j))) > 0:
                    d_prod_len[tup] = len(dict_sols_per_eq[part_i]) * len(dict_sols_per_eq[part_j])

        # Step 1: Find the minimum value and keys with this value in a single pass
        min_value = float("inf")
        min_keys = []
        for key, value in d_prod_len.items():
            if value < min_value:
                min_value = value
                min_keys = [key]
            elif value == min_value:
                min_keys.append(key)
        # Step 2: Calculate the length of the union of sets for the keys with the minimum value
        d_tmp = {
            min_key_tmp: len(set(min_key_tmp[0]) | set(min_key_tmp[1])) for min_key_tmp in min_keys
        }
        # Step 3: Find the key with the minimum value in d_tmp
        min_key = min(d_tmp, key=d_tmp.get)

        merged_parts = tuple(sorted(set(dict.fromkeys(min_key[0] + min_key[1]))))
        if verbose:
            print("Intersection", min_key, "merged part", merged_parts)
        merged_sol = intersection_of_solutions(
            dict_sols_per_eq[min_key[0]], dict_sols_per_eq[min_key[1]], test_compatibility=True
        )
        if verbose:
            print("Merged sol", len(merged_sol), "max possible was", d_prod_len[min_key])
        if len(merged_sol) == 0:
            return [], []
        # We restrict the solutions to C when it is possible if merged part is new
        if merged_parts in dict_sols_per_eq:
            if merged_parts not in min_key:
                merged_sol = intersection_of_solutions(
                    dict_sols_per_eq[merged_parts], merged_sol, test_compatibility=True
                )
        else:
            merged_sol, lines_of_C_already_satisfied = restrict_sol_by_C(
                merged_sol,
                merged_parts,
                C,
                parity_indices,
                partitions_involved_for_C,
                lines_of_C_already_satisfied,
                verbose=verbose,
            )
        if len(merged_sol) == 0:
            return [], lines_of_C_already_satisfied
        del dict_sols_per_eq[min_key[0]]
        del dict_sols_per_eq[min_key[1]]
        dict_sols_per_eq[merged_parts] = merged_sol
    return dict_sols_per_eq[list(dict_sols_per_eq.keys())[0]], lines_of_C_already_satisfied


def solutions_of_P(
    P,
    morgan,
    C,
    parity_indices,
    partitions_involved_for_C,
    lines_of_C_already_satisfied,
    max_nbr_partition=int(1e5),
    verbose=False,
):
    """
    Compute all possible solutions for a matrix P given constraints.

    This function computes solutions for each row in the matrix `P`, based on the `morgan` values,
    with constraints applied from matrix `C`. The function generates partitions for each row's
    non-zero elements and checks for compatibility with `C`, considering parity constraints and
    already satisfied lines. It limits the number of partitions to avoid excessive computation.

    Parameters
    ----------
    P : numpy.ndarray
        The matrix representing the system for which solutions are computed. Rows with non-zero
        elements indicate parts that need to be partitioned according to the corresponding `morgan` value.
    morgan : list of int
        A list of integers representing the values that each row in `P` should sum to when partitions
        are considered.
    C : numpy.ndarray
        The matrix containing constraints that solutions must satisfy.
    parity_indices : list of int
        Indices indicating which rows in `C` correspond to parity constraints.
    partitions_involved_for_C : list of list of int
        A list where each element is a list of indices involved in a specific row of `C`.
    lines_of_C_already_satisfied : set of int
        A set of line indices in `C` that have already been satisfied by previous computations.
    max_nbr_partition : int, optional
        The maximum number of partitions to generate, default is 100,000.
    verbose : bool, optional
        If True, print detailed information about the processing steps. Default is False.

    Returns
    -------
    tuple
        - dict_sols (dict): A dictionary where keys are tuples representing partitions, and values
          are lists of possible solutions for each partition.
        - lines_of_C_already_satisfied (set of int): Updated list of line indices in `C` that are satisfied.
        - bool_threshold_reached: A flag indicating if the partitioning process was truncated due to reaching `max_nbr_partition`.
    """

    nb_lin, nb_col = P.shape[0], P.shape[1]
    dict_sols = {}
    bool_threshold_reached = False
    local_bound = max(10, int(max_nbr_partition / 100))
    for i in range(nb_lin):
        bool_threshold_reached_local = False
        # For each line we compute all the partitions of the corresponding morgan bit
        part_line_of_P = tuple([j for j in range(nb_col) if P[i, j] != 0])
        counts = [P[i, j] for j in part_line_of_P]
        if len(set(counts)) != 1 or (morgan[i] / counts[0]).is_integer() == False:
            All_parts_morgan_in_k2, bool_threshold_reached_local = partitions_on_non_constant(
                counts, morgan[i], max_nbr_partition_non_constant=local_bound
            )
            if bool_threshold_reached_local:
                bool_threshold_reached = True
        else:
            k = len(part_line_of_P)
            morgan_normed = int(morgan[i] / counts[0])
            All_parts_morgan_in_k = list(partitions(morgan_normed, k))
            All_parts_morgan_in_k2 = []
            if len(All_parts_morgan_in_k) > local_bound:
                bool_threshold_reached = True
                bool_threshold_reached_local = True
                All_parts_morgan_in_k = All_parts_morgan_in_k[:local_bound]
            for x in All_parts_morgan_in_k:
                nb_perm = nb_permutations(x + [0] * (sum(P[i, :]) - len(x)))
                if nb_perm > max_nbr_partition:
                    bool_threshold_reached = True
                    bool_threshold_reached_local = True
                    All_parts_morgan_in_k2 = All_parts_morgan_in_k2 + list(
                        itertools.islice(
                            multiset_permutations(x + [0] * (k - len(x))),
                            max_nbr_partition,
                        )
                    )
                else:
                    All_parts_morgan_in_k2 = All_parts_morgan_in_k2 + list(
                        multiset_permutations(x + [0] * (k - len(x)))
                    )
        if bool_threshold_reached_local and verbose:
            print(
                "Threshold reached! part of P",
                part_line_of_P,
                "P values",
                counts,
                "morgan",
                morgan[i],
            )
        # We complete the solutions with -1 for all zeros in the line
        local_sols = [
            partition_to_local_sol(part, part_line_of_P, nb_col) for part in All_parts_morgan_in_k2
        ]
        # We restrict the solutions to C when it is possible
        local_sols, lines_of_C_already_satisfied = restrict_sol_by_C(
            local_sols,
            part_line_of_P,
            C,
            parity_indices,
            partitions_involved_for_C,
            lines_of_C_already_satisfied,
            verbose=verbose,
        )
        if len(local_sols) == 0:
            return {}, lines_of_C_already_satisfied, bool_threshold_reached
        # We add the solutions of this line to the dictionary of solutions
        if part_line_of_P in dict_sols:
            dict_sols[part_line_of_P] = intersection_of_solutions(
                dict_sols[part_line_of_P], local_sols, test_compatibility=True
            )
        else:
            dict_sols[part_line_of_P] = local_sols
    return dict_sols, lines_of_C_already_satisfied, bool_threshold_reached


def is_vector_inferior_or_equal(vector1, vector2):
    """
    Check if each component of the first vector is less than or equal to the corresponding component of the second vector.

    This function compares two vectors component-wise and returns `True` if every element in `vector1`
    is less than or equal to the corresponding element in `vector2`. If any element in `vector1`
    is greater than the corresponding element in `vector2`, the function returns `False`.

    Parameters
    ----------
    vector1 : list or array-like
        The first vector to compare. This vector's components are checked against the corresponding components of `vector2`.
    vector2 : list or array-like
        The second vector to compare. This vector serves as the upper bound for the comparison.

    Returns
    -------
    bool
        `True` if all components of `vector1` are less than or equal to the corresponding components of `vector2`.
        `False` otherwise.
    """

    return all(a <= b for a, b in zip(vector1, vector2))


def sol_max(P, morgan):
    """
    Compute the maximum possible solution for each column of matrix `P` based on the `morgan` values.

    This function iterates over each column of the matrix `P` and determines the maximum possible value
    for that column by finding the minimum value in `morgan` corresponding to the non-zero elements of
    `P` in that column. The result is a list where each element corresponds to the maximum possible
    value for the respective column in `P`.

    Parameters
    ----------
    P : numpy.ndarray
        A 2D array where each column represents a set of elements to be considered. Non-zero elements
        in a column indicate the relevant rows in the matrix.
    morgan : list of int
        A list of integers where each element corresponds to a row in matrix `P`. The values in `morgan`
        are used to determine the maximum possible solution for each column in `P`.

    Returns
    -------
    list of int
        A list where each element represents the maximum possible solution for the corresponding
        column in `P`, determined by the minimum value of `morgan` among the non-zero elements in that column.
    """

    sol_max = []
    for i in range(P.shape[1]):
        indices_P_i = [j for j in range(P.shape[0]) if P[j, i] != 0]
        min_morgan_i = min([int(morgan[j] / P[j, i]) for j in indices_P_i])
        sol_max.append(min_morgan_i)
    return sol_max


def clean_solutions_by_sol_max(sol_max, dict_sols):
    """
    Filter solutions in a dictionary by comparing them against a maximum solution vector.

    This function iterates through each key in `dict_sols` and filters the list of solutions associated
    with that key. Only those solutions that are component-wise less than or equal to the corresponding
    components in `sol_max` are retained. The function returns the updated dictionary with filtered solutions.

    Parameters
    ----------
    sol_max : list of int
        A list representing the maximum allowed values for each component of the solutions.
        Solutions in `dict_sols` that exceed these values in any component are removed.
    dict_sols : dict
        A dictionary where each key corresponds to a partition or identifier, and the value is a list
        of possible solutions (each solution is a list or vector of integers).

    Returns
    -------
    dict
        The input dictionary `dict_sols`, with the lists of solutions filtered so that only those
        solutions that are component-wise less than or equal to `sol_max` remain.
    """

    for key in dict_sols:
        dict_sols[key] = [s for s in dict_sols[key] if is_vector_inferior_or_equal(s, sol_max)]
    return dict_sols


# =================================================================================================
# Solve function
# =================================================================================================


def solve_by_partitions(P, morgan, C, max_nbr_partition=int(1e5), verbose=False):
    """
    Solve the system defined by matrix P, morgan, and constraints matrix C using partitions.

    This function attempts to find solutions to a system where each row of `P` corresponds to a
    partitioned set of the `morgan` values, while satisfying constraints from matrix `C`. The
    solution process includes updating `C`, computing partitions, forming groups of partitions,
    and intersecting groups to obtain the final set of solutions.

    Parameters
    ----------
    P : numpy.ndarray
        The matrix defining the system to solve, with non-zero elements indicating parts to partition
        according to `morgan`.
    morgan : list of int
        A list of target sums for each row in `P`, dictating how the parts should sum.
    C : numpy.ndarray
        The constraint matrix, where each row represents a constraint that the solution must satisfy.
    max_nbr_partition : int, optional
        The maximum number of partitions to generate per partitioning, default is 100,000.
    verbose : bool or int, optional
        If True or set to 1, print general progress information. If set to 2, print detailed
        debug information including matrices. Default is False.

    Returns
    -------
    tuple
        - S (list): The list of solutions that satisfy both the partitioning and the constraints in `C`.
        - bool_threshold_reached (bool): A flag indicating if the partitioning process was truncated due to reaching `max_nbr_partition`.
    """

    if max_nbr_partition <= 0:
        return [], True
    if verbose:
        print(f"P {P.shape}, C {C.shape}, morgan {len(morgan)}")
    if verbose == 2:
        print(f"P\n {repr(P)}")
        print(f"morgan\n {repr(morgan)}")
        print(f"C\n {repr(C)}")
    bool_threshold_reached = False
    nb_col = P.shape[1]
    # We update C and compute its partitions
    C, parity_indices = update_C(C, nb_col)
    partitions_involved_for_C = {}
    for i in range(C.shape[0]):
        parts = tuple([j for j in range(C.shape[1]) if C[i, j] != 0])
        partitions_involved_for_C[i] = parts
    if verbose:
        print("Partitions involved for C:", list(partitions_involved_for_C.values()))
    lines_of_C_already_satisfied = set()
    # We compute the solutions for each line of P
    dict_sols, lines_of_C_already_satisfied, bool_threshold_reached = solutions_of_P(
        P,
        morgan,
        C,
        parity_indices,
        partitions_involved_for_C,
        lines_of_C_already_satisfied,
        max_nbr_partition,
        verbose,
    )
    if len(dict_sols) == 0:
        return [], bool_threshold_reached
    # We clean solutions using the maximum possible solution
    s_max = sol_max(P, morgan)
    dict_sols = clean_solutions_by_sol_max(s_max, dict_sols)
    # We compute the partitions of each line of P
    partitions_involved = list(dict_sols.keys())
    if verbose:
        print("Partitions involved for P:", partitions_involved)
    # We compute the groups of partitions
    parts_groups = partitions_groups(partitions_involved)
    if verbose:
        print("Groups of partitions:", parts_groups)
    # We compute the groups of solutions for each partition group
    S_groups, lines_of_C_already_satisfied = groups_of_solutions(
        dict_sols,
        parts_groups,
        C,
        partitions_involved_for_C,
        parity_indices,
        lines_of_C_already_satisfied,
        verbose=verbose,
    )
    if len(S_groups) == 0:
        return [], bool_threshold_reached
    # We compute the intersection between the groups of partitions
    if verbose:
        print("We compute the sol between the distinct groups (intersection)")
    S = S_groups[0]
    for i in range(1, len(S_groups)):
        S_group = S_groups[i]
        if verbose:
            print(f"S_tmp x S_group: {len(S)} x {len(S_group)}")
        S = intersection_of_solutions(S, S_group, test_compatibility=False)
        if verbose:
            print(f"after inter: {len(S)}")
    if len(S) == 0:
        return [], bool_threshold_reached
    # We now restrict of the solutions of P with the remaining lines of C
    if verbose:
        print(
            "before C, S len",
            len(S),
            "C shape",
            C.shape,
            "len(lines_of_C_already_satisfied)",
            len(lines_of_C_already_satisfied),
        )
    for i in range(C.shape[0]):
        if i not in lines_of_C_already_satisfied:
            S = restrict_sol_by_one_line_of_C(S, C, i, parity_indices)
            if len(S) == 0:
                return [], bool_threshold_reached
    if verbose:
        print("Finally, S len", len(S))
    return S, bool_threshold_reached
