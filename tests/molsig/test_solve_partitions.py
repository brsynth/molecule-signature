# -*- coding: utf-8 -*-
"""
Tests for functions in the `signature.solve_partitions` module.

Authors:
  - Jean-loup Faulon <jfaulon@gmail.com>
  - Thomas Duigou <thomas.duigou@inrae.fr>
  - Philippe Meyer <philippe.meyer@inrae.fr>
"""

import numpy as np

from molsig.solve_partitions import (clean_solutions_by_sol_max,
                                        compatibility, groups_of_solutions,
                                        intersection_of_lists_of_lists,
                                        intersection_of_solutions,
                                        is_vector_inferior_or_equal,
                                        nb_permutations,
                                        partition_to_local_sol, partitions,
                                        partitions_groups,
                                        partitions_on_non_constant,
                                        restrict_sol_by_C,
                                        restrict_sol_by_one_line_of_C,
                                        sized_partitions, sol_max,
                                        solution_of_one_group, solutions_of_P,
                                        solve_by_partitions, update_C)


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
    assert partitions(5, 5) == [[1, 1, 1, 1, 1], [2, 1, 1, 1], [2, 2, 1], [3, 1, 1], [3, 2], [4, 1], [5]]
    assert partitions(4, 3) == [[2, 1, 1], [2, 2], [3, 1], [4]]
    assert partitions(3, 6) == [[1, 1, 1], [2, 1], [3]]


def test_nb_permutations():
    assert nb_permutations([]) == 1
    assert nb_permutations([1, 2, 3]) == 6
    assert nb_permutations([1, 1]) == 1
    assert nb_permutations([1, 1, 2]) == 3


def test_intersection_of_lists_of_lists():
    assert intersection_of_lists_of_lists([[1]], [[2]]) == []
    assert intersection_of_lists_of_lists([[1]], [[1], [2]]) == [[1]]


def test_compatibility():
    assert compatibility([-1, 2], [1, -1]) == True
    assert compatibility([-1, 2], [1, 1]) == False


def test_intersection_of_solutions():
    assert intersection_of_solutions([[-1, 2]], [[1, -1]], False) == [[1, 2]]
    assert intersection_of_solutions([[-1, 2]], [[1, -1], [-1, 1]], True) == [[1, 2]]


def test_partitions_groups():
    partitions_involved = [(6, 13), (9, 13), (0, 7), (0, 11), (1,), (2,), (3,), (4,), (5,), (8,), (10,), (12,)]
    expected_groups = [{9, 13, 6}, {0, 11, 7}, {1}, {2}, {3}, {4}, {5}, {8}, {10}, {12}]
    groups = partitions_groups(partitions_involved)
    assert groups == expected_groups


def test_partition_to_local_sol():
    # Test case 1: Typical case
    partition = [1, 2, 3]
    part_line_of_P = [0, 2, 4]
    nb_lin = 5
    expected_output = [1, -1, 2, -1, 3]
    assert partition_to_local_sol(partition, part_line_of_P, nb_lin) == expected_output

    # Test case 2: Empty partition and part_line_of_P
    partition = []
    part_line_of_P = []
    nb_lin = 3
    expected_output = [-1, -1, -1]
    assert partition_to_local_sol(partition, part_line_of_P, nb_lin) == expected_output

    # Test case 3: All indices filled
    partition = [0, 1, 2]
    part_line_of_P = [0, 1, 2]
    nb_lin = 3
    expected_output = [0, 1, 2]
    assert partition_to_local_sol(partition, part_line_of_P, nb_lin) == expected_output

    # Test case 4: nb_lin larger than needed
    partition = [1, 2]
    part_line_of_P = [0, 1]
    nb_lin = 5
    expected_output = [1, 2, -1, -1, -1]
    assert partition_to_local_sol(partition, part_line_of_P, nb_lin) == expected_output


def test_update_C():
    # Test case 1: Typical case without graph
    C = np.array([[1, 2, 3], [0, 0, 4], [5, 6, 0]])
    nb_col = 2
    graph = False
    expected_C = np.array([[1, 2], [0, 0], [5, 6]])
    expected_parity_indices = [0, 1]
    result_C, result_parity_indices = update_C(C, nb_col, graph)
    assert np.array_equal(result_C, expected_C)
    assert result_parity_indices == expected_parity_indices

    # Test case 2: Typical case with graph
    C = np.array([[1, 2, 3], [0, 0, 4], [5, 6, 0]])
    nb_col = 2
    graph = True
    expected_C = np.array([[1, 2], [0, 0]])
    expected_parity_indices = [0, 1]
    expected_graph_line = np.array([[5, 6]])
    expected_graph_index = 2
    result_C, result_parity_indices, result_graph_line, result_graph_index = update_C(C, nb_col, graph)
    assert np.array_equal(result_C, expected_C)
    assert result_parity_indices == expected_parity_indices
    assert np.array_equal(result_graph_line, expected_graph_line)
    assert result_graph_index == expected_graph_index

    # Test case 3: No rows beyond nb_col have non-zero elements
    C = np.array([[1, 2, 0], [0, 0, 0], [5, 6, 0]])
    nb_col = 2
    graph = False
    expected_C = np.array([[1, 2], [0, 0], [5, 6]])
    expected_parity_indices = []
    result_C, result_parity_indices = update_C(C, nb_col, graph)
    assert np.array_equal(result_C, expected_C)
    assert result_parity_indices == expected_parity_indices

    # Test case 4: Empty matrix
    C = np.array([]).reshape(0, 3)
    nb_col = 2
    graph = False
    expected_C = np.array([]).reshape(0, 2)
    expected_parity_indices = []
    result_C, result_parity_indices = update_C(C, nb_col, graph)
    assert np.array_equal(result_C, expected_C)
    assert result_parity_indices == expected_parity_indices

    # Test case 5: Single row matrix with graph
    C = np.array([[1, 2, 3]])
    nb_col = 2
    graph = True
    expected_C = np.array([]).reshape(0, 2)
    expected_parity_indices = [0]
    expected_graph_line = np.array([[1, 2]])
    expected_graph_index = 0
    result_C, result_parity_indices, result_graph_line, result_graph_index = update_C(C, nb_col, graph)
    assert np.array_equal(result_C, expected_C)
    assert result_parity_indices == expected_parity_indices
    assert np.array_equal(result_graph_line, expected_graph_line)
    assert result_graph_index == expected_graph_index


def test_restrict_sol_by_one_line_of_C():
    # Test case 1: Typical case with parity constraint
    S = np.array([[1, 0], [0, 1], [1, 1]])
    C = np.array([[1, 1], [0, 1]])
    i = 0
    parity_indices = [0]
    expected_output = [S[2]]  # Only the row with even parity matches
    result = restrict_sol_by_one_line_of_C(S, C, i, parity_indices)
    assert np.array_equal(result, expected_output)

    # Test case 2: No matching rows
    S = np.array([[1, 0], [-1, 1], [1, 1]])
    C = np.array([[1, 1], [1, 0]])
    i = 1
    parity_indices = [0]
    expected_output = []  # No rows satisfy the condition
    result = restrict_sol_by_one_line_of_C(S, C, i, parity_indices)
    assert np.array_equal(result, expected_output)

    # Test case 3: Empty solution set
    S = np.array([]).reshape(0, 2)
    C = np.array([[1, 1], [0, 1]])
    i = 0
    parity_indices = [0]
    expected_output = []  # Empty input should return empty output
    result = restrict_sol_by_one_line_of_C(S, C, i, parity_indices)
    assert np.array_equal(result, expected_output)


def test_restrict_sol_by_C():
    S = [[2, 0], [0, 1], [1, 1]]
    part_P = (0, 1)
    C = np.array([[1, 0], [0, 1], [0, 1]])
    parity_indices = [0]
    partitions_involved_for_C = {0: (0,), 1: (1,), 2: (0, 1)}
    lines_of_C_already_satisfied = set()
    expected_S = [[2, 0]]
    expected_lines_satisfied = {2}
    result_S, result_lines_satisfied = restrict_sol_by_C(
        S, part_P, C, parity_indices, partitions_involved_for_C, lines_of_C_already_satisfied
    )
    assert result_S == expected_S
    assert result_lines_satisfied == expected_lines_satisfied


def test_partitions_on_non_constant():
    v = [2, 1, 3]
    target_sum = 3
    expected_partitions = [[0, 0, 1], [1, 1, 0], [0, 3, 0]]
    result_partitions, threshold_reached = partitions_on_non_constant(v, target_sum)
    assert result_partitions == expected_partitions
    assert not threshold_reached


def test_groups_of_solutions():
    dict_sols_per_eq = {
        (0, 1): [[1, 1, -1, -1]],
        (2,): [[-1, -1, 1, -1]],
        (3,): [[-1, -1, -1, 1]],
        (0,): [[1, -1, -1, -1]],
        (1,): [[-1, 1, -1, -1]],
    }
    parts_groups = {
        (0, 1): [[1, 1, -1, -1]],
        (2,): [[-1, -1, 1, -1]],
        (3,): [[-1, -1, -1, 1]],
        (0,): [[1, -1, -1, -1]],
        (1,): [[-1, 1, -1, -1]],
    }
    C = np.array([[-1, 1, 0, 0], [0, 1, 0, -1], [1, 0, -1, 0], [0, 0, -1, -1]])
    partitions_involved_for_C = {0: (0, 1), 1: (1, 3), 2: (0, 2), 3: (2, 3)}
    parity_indices = [3]
    lines_of_C_already_satisfied = {0}
    expected_S_groups = [
        [[np.int64(1), np.int64(1), np.int64(-1), np.int64(-1)]],
        [[-1, -1, 1, -1]],
        [[-1, -1, -1, 1]],
        [[np.int64(1), np.int64(1), np.int64(-1), np.int64(-1)]],
        [[-1, 1, -1, -1]],
    ]
    expected_lines_of_C_already_satisfied = {0}
    result_S_groups, result_lines_of_C_already_satisfied = groups_of_solutions(
        dict_sols_per_eq, parts_groups, C, partitions_involved_for_C, parity_indices, lines_of_C_already_satisfied
    )
    assert result_S_groups == expected_S_groups
    assert result_lines_of_C_already_satisfied == expected_lines_of_C_already_satisfied


def test_solution_of_one_group():
    dict_sols_per_eq = {(2,): [[-1, -1, 1]]}
    C = np.array([[-1, 0, 1], [-1, 1, 0], [0, -1, -1]])
    partitions_involved_for_C = {0: (0, 2), 1: (0, 1), 2: (1, 2)}
    parity_indices = [2]
    lines_of_C_already_satisfied = set()
    result_S_group, result_lines_of_C_already_satisfied = solution_of_one_group(
        dict_sols_per_eq, C, partitions_involved_for_C, parity_indices, lines_of_C_already_satisfied
    )
    expected_S_group = [[np.int64(-1), np.int64(-1), np.int64(1)]]
    expected_lines_of_C_already_satisfied = set()
    assert result_S_group == expected_S_group
    assert result_lines_of_C_already_satisfied == expected_lines_of_C_already_satisfied


def test_solutions_of_P():
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    morgan = [1, 1, 1, 1, 1, 1]
    C = np.array([[-1, 0, 1], [-1, 1, 0], [0, -1, -1]])
    parity_indices = [2]
    partitions_involved_for_C = {0: (0, 2), 1: (0, 1), 2: (1, 2)}
    lines_of_C_already_satisfied = set()
    result_dict_sols, result_lines_of_C_already_satisfied, result_bool_threshold_reached = solutions_of_P(
        P, morgan, C, parity_indices, partitions_involved_for_C, lines_of_C_already_satisfied
    )
    expected_dict_sols = {
        (0,): [[np.int64(1), np.int64(-1), np.int64(-1)]],
        (1,): [[np.int64(-1), np.int64(1), np.int64(-1)]],
        (2,): [[np.int64(-1), np.int64(-1), np.int64(1)]],
    }
    expected_lines_of_C_already_satisfied = set()
    expected_bool_threshold_reached = False
    assert result_dict_sols == expected_dict_sols
    assert result_lines_of_C_already_satisfied == expected_lines_of_C_already_satisfied
    assert result_bool_threshold_reached == expected_bool_threshold_reached


def test_is_vector_inferior_or_equal():
    # Test case 1: All elements of vector1 are less than or equal to vector2
    vector1 = [1, 2, 3]
    vector2 = [3, 2, 3]
    assert is_vector_inferior_or_equal(vector1, vector2) == True

    # Test case 2: An element in vector1 is greater than the corresponding element in vector2
    vector1 = [1, 4, 3]
    vector2 = [3, 2, 3]
    assert is_vector_inferior_or_equal(vector1, vector2) == False

    # Test case 3: Both vectors are identical
    vector1 = [5, 5, 5]
    vector2 = [5, 5, 5]
    assert is_vector_inferior_or_equal(vector1, vector2) == True

    # Test case 4: vector1 is empty
    vector1 = []
    vector2 = []
    assert is_vector_inferior_or_equal(vector1, vector2) == True


def test_sol_max():
    # Test case 1: Simple matrix with valid morgan values
    P = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    morgan = [4, 9, 3]
    expected_sol_max = [4, 3, 3]
    result_sol_max = sol_max(P, morgan)
    assert result_sol_max == expected_sol_max


def test_clean_solutions_by_sol_max():
    # Test case 1: Basic filtering
    sol_max = [3, 2, 4]
    dict_sols = {
        "group1": [[1, 2, 3], [4, 1, 3], [3, 2, 4]],
        "group2": [[2, 1, 4], [3, 2, 5], [0, 0, 0]],
    }
    expected_dict_sols = {
        "group1": [[1, 2, 3], [3, 2, 4]],
        "group2": [[2, 1, 4], [0, 0, 0]],
    }
    result_dict_sols = clean_solutions_by_sol_max(sol_max, dict_sols)
    assert result_dict_sols == expected_dict_sols


def test_solve_by_partitions():
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    morgan = [1, 1, 1, 1, 1, 1]
    C = np.array([[-1, 0, 1, 0, 0], [-1, 1, 0, 0, 0], [0, -1, -1, 2, -2]])
    expected_S = [[1, 1, 1]]
    expected_bool_threshold_reached = False
    result_S, result_bool_threshold_reached = solve_by_partitions(P, morgan, C)
    assert result_S == expected_S
    assert result_bool_threshold_reached == expected_bool_threshold_reached
