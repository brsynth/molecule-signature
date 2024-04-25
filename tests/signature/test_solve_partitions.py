# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:57:23 2024

@author: meyerp
"""

import numpy as np

from signature.solve_partitions import (
    clean_local_solutions,
    compatibility,
    equations_trivially_satisfied,
    extract_matrices_C_P,
    groups_of_solutions,
    intersection_of_solutions,
    intersection_of_lists_of_lists,
    nb_permutations,
    partitions,
    partitions_groups,
    partitions_P_N,
    sized_partitions,
    solutions_per_line,
    solve_by_partitions,
    sort_C_wrt_partitions_involved,
    sort_group_of_partitions,
)


def test_sized_partitions():
    assert list(sized_partitions(5, 0)) == []
    assert list(sized_partitions(4, 5)) == []
    assert list(sized_partitions(0, 8)) == []
    assert list(sized_partitions(3, 3)) == [[1, 1, 1]]
    assert list(sized_partitions(2, 1)) == [[2]]
    assert list(sized_partitions(8, 3)) == [[6, 1, 1], [5, 2, 1], [4, 3, 1], [4, 2, 2], [3, 3, 2]]


def test_partitions():
    assert partitions(0, 5) == []
    assert partitions(5, 0) == []
    assert partitions(5, 1) == [[5]]
    assert partitions(5, 5) == [[5], [4, 1], [3, 2], [3, 1, 1], [2, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert partitions(4, 3) == [[4], [3, 1], [2, 2], [2, 1, 1]]
    assert partitions(3, 6) == [[3], [2, 1], [1, 1, 1]]


def test_nb_permutations():
    assert nb_permutations([]) == 1
    assert nb_permutations([1, 2, 3]) == 6
    assert nb_permutations([1, 1]) == 1
    assert nb_permutations([1, 1, 2]) == 3


def test_extract_matrices_C_P():
    A = np.array([[1, 0, -2, 0, 0], [0, 1, 1, 2, -2], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    b = np.array([0, 0, 1, 2])
    C, P, N, parity_indices, graph_line, graph_index = extract_matrices_C_P(A, b)
    expected_C = np.array([[1, 0, -2], [0, 1, 1]])
    expected_P = np.array([[1, 1, 0], [0, 0, 1]])
    expected_N = np.array([1, 2])
    expected_parity_indices = [False, True]
    expected_graph_line = np.array([[0, 1, 1]])
    expected_graph_index = 1
    assert np.array_equal(C, expected_C)
    assert np.array_equal(P, expected_P)
    assert np.array_equal(N, expected_N)
    assert parity_indices == expected_parity_indices
    assert np.array_equal(graph_line, expected_graph_line)
    assert graph_index == expected_graph_index


def test_partitions_P_N():
    P = np.array([[1, 1, 0], [0, 0, 1]])
    N = np.array([3, 1])
    max_nbr_partition = 1e5
    bool_timeout = False
    dict_partitions, tups, bool_timeout = partitions_P_N(P, N, max_nbr_partition, bool_timeout)
    expected_dict_partitions = dict()
    expected_dict_partitions[0] = [[0, 3], [3, 0], [1, 2], [2, 1]]
    expected_dict_partitions[1] = [[1]]
    expected_tups = [(0, 2), (2, 3)]
    assert dict_partitions == expected_dict_partitions
    assert tups == expected_tups
    assert bool_timeout == False


def test_equations_trivially_satisfied():
    C = np.array([[1, 0, -2], [0, 1, 1]])
    N = np.array([1, 2])
    tups = [(0, 2), (2, 3)]
    parity_indices = [False, True]
    graph_index = 1
    indices, graph = equations_trivially_satisfied(C, N, tups, parity_indices, graph_index)
    expected_indices = [0, 1]
    expected_graph = False
    assert indices == expected_indices
    assert graph == expected_graph


def test_sort_C_wrt_partitions_involved():
    C = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    tups = [(0, 2), (2, 3)]
    parity_indices = [True, False, True]
    sorted_C, sorted_parity_indices, sorted_partitions_involved = sort_C_wrt_partitions_involved(
        C, tups, parity_indices
    )
    expected_C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 1]])
    expected_parity_indices = [False, True, True]
    expected_partitions_involved = [(0,), (1,), (0, 1)]
    assert np.array_equal(sorted_C, expected_C)
    assert sorted_parity_indices == expected_parity_indices
    assert sorted_partitions_involved == expected_partitions_involved


def test_intersection_of_lists_of_lists():
    assert intersection_of_lists_of_lists([[1]], [[2]]) == []
    assert intersection_of_lists_of_lists([[1]], [[1], [2]]) == [[1]]


def test_solutions_per_line():
    C = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        ]
    )
    tups = [
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 16),
    ]
    max_nbr_partition = int(1e5)
    bool_timeout = False
    parity_indices = [False, False, False, False]
    dict_partitions = {
        0: [[0, 2], [2, 0], [1, 1]],
        1: [[1]],
        2: [[1]],
        3: [[1]],
        4: [[1]],
        5: [[1]],
        6: [[1]],
        7: [[1]],
        8: [[1]],
        9: [[1]],
        10: [[1]],
        11: [[1]],
        12: [[1]],
        13: [[0, 2], [2, 0], [1, 1]],
    }
    partitions_involved = [(6, 13), (9, 13), (0, 7), (0, 11)]
    dict_sols_per_eq, dict_partitions, bool_timeout = solutions_per_line(
        C, tups, parity_indices, dict_partitions, partitions_involved, max_nbr_partition, bool_timeout
    )
    expected_dict_sols_per_eq = {
        (6, 13): [[-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
        (9, 13): [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1]],
        (0, 7): [[1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]],
        (0, 11): [[1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1]],
    }
    expected_dict_partitions = {
        0: [[1, 1]],
        1: [[1]],
        2: [[1]],
        3: [[1]],
        4: [[1]],
        5: [[1]],
        6: [[1]],
        7: [[1]],
        8: [[1]],
        9: [[1]],
        10: [[1]],
        11: [[1]],
        12: [[1]],
        13: [[1, 1]],
    }
    assert dict_sols_per_eq == expected_dict_sols_per_eq
    assert dict_partitions == expected_dict_partitions
    assert bool_timeout == False


def test_clean_local_solutions():
    tups = [(0, 2), (2, 3)]
    dict_sols_per_eq = {(0,): [[1, 2, -1], [2, 1, -1], [3, 0, -1], [0, 3, -1]], (1,): [[-1, -1, 2]]}
    dict_partitions = {0: [[1, 2]], 1: [[2]]}
    dict_sols_per_eq = clean_local_solutions(tups, dict_sols_per_eq, dict_partitions)
    expected_dict_sols_per_eq = {(0,): [[1, 2, -1]], (1,): [[-1, -1, 2]]}
    assert dict_sols_per_eq == expected_dict_sols_per_eq


def test_partitions_groups():
    partitions_involved = [(6, 13), (9, 13), (0, 7), (0, 11), (1,), (2,), (3,), (4,), (5,), (8,), (10,), (12,)]
    expected_groups = [{9, 13, 6}, {0, 11, 7}, {1}, {2}, {3}, {4}, {5}, {8}, {10}, {12}]
    groups = partitions_groups(partitions_involved)
    assert groups == expected_groups


def test_compatibility():
    assert compatibility([-1, 2], [1, -1]) == True
    assert compatibility([-1, 2], [1, 1]) == False


def test_intersection_of_solutions():
    assert intersection_of_solutions([[-1, 2]], [[1, -1]], False) == [[1, 2]]
    assert intersection_of_solutions([[-1, 2]], [[1, -1], [-1, 1]], True) == [[1, 2]]


def test_sort_group_of_partitions():
    dict_sols_per_eq = {
        (0,): [[1, 2], [3, 4], [5, 6]],
        (1,): [[7, 8], [9, 10]],
        (2,): [[11, 12], [13, 14], [15, 16], [17, 18]],
    }
    group = {0, 1, 2}
    sorted_group = sort_group_of_partitions(dict_sols_per_eq, group)
    expected_sorted_group = [(1,), (0,), (2,)]
    assert sorted_group == expected_sorted_group


def test_groups_of_solutions():
    dict_sols_per_eq = {
        (6, 13): [[-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
        (9, 13): [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1]],
        (0, 7): [[1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]],
        (0, 11): [[1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1]],
        (1,): [[-1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        (2,): [[-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        (3,): [[-1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        (4,): [[-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        (5,): [[-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        (8,): [[-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]],
        (10,): [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1]],
        (12,): [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]],
    }
    parts_groups = [{9, 13, 6}, {0, 11, 7}, {1}, {2}, {3}, {4}, {5}, {8}, {10}, {12}]
    S_groups = groups_of_solutions(dict_sols_per_eq, parts_groups)
    expected_S_groups = [
        [[-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1]],
        [[1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1]],
        [[-1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        [[-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        [[-1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        [[-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        [[-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]],
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1]],
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]],
    ]
    for S_group, expected_S_group in zip(S_groups, expected_S_groups):
        assert np.array_equal(S_group, expected_S_group)


def test_solve_by_partitions():
    A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
            [-1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 0, 1, 0, -1, -1, 2, -2],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        ]
    )
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    S, bool_timeout = solve_by_partitions(A, b)
    expected_S = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    assert np.array_equal(S, expected_S)
    assert bool_timeout is False
