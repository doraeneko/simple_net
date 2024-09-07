#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Matrix class
#################################################################

from .common import *


def test_matrix_1():
    m1 = construct_m1()
    assert m1.shape() == (3, 4)
    assert m1[1, 1] == -0.25
    assert m1[2, 3] == 0.0
    assert m1[2, 3] == 0.0
    m1[2, 3] += 1.5
    assert m1[2, 3] == 1.5


def test_matrix_2():
    m1 = construct_m1()
    m1.apply(lambda n: n + 1.0)
    assert m1[1, 1] == 0.75
    assert m1[2, 3] == 1.0
    assert m1[2, 3] == 1.0


def test_matrix_3():
    m1 = construct_m1()
    v1 = m1.get_column_vector(0)
    assert v1[0] == 0.25
    assert v1[1] == -0.25
    assert v1[2] == 0.0
    assert v1[3] == 1.0
    v1[0] += 0.25
    assert m1[0, 0] == 0.25
    m1.set_column_vector(0, v1)
    assert m1[0, 0] == 0.5
    v2 = m1.get_column_vector(1)
    assert v2[3] == 1.0


def test_matrix_4():
    m1 = construct_m1()
    m1.apply_relu()
    assert m1[0, 1] == 0.0
    assert m1[0, 0] == 0.25
    m1.elementwise_add_with(10.0)
    assert m1[0, 0] == 10.25


def test_matrix_5():
    m2 = construct_m2()
    v1 = construct_v1()
    v2 = m2.multiply_vec_right(v1)
    assert v2[0] == 0.0625
    assert v2[1] == -0.0625 - 0.25 + 1.0


def test_matrix_6():
    m1 = construct_m1()
    m1.add_with(m1)
    assert m1[1, 1] == -0.5


def test_matrix_6():
    m1 = construct_m1()
    m1.add_with(m1)
    assert m1[1, 1] == -0.5
    m1.elementwise_add_with(10.0)
    assert m1[1, 1] == 9.5


def test_matrix_7():
    m1 = construct_m1()
    m1.apply_relu()
    assert m1[1, 1] == 0.0
    assert m1[0, 3] == 1.0
