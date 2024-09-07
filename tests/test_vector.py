#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Vector class
#################################################################

from .common import *


def test_vector_1():
    v1 = construct_v1()
    assert v1.shape() == (4,)
    assert v1.dim() == 4


def test_vector_2():
    v1 = construct_v1()
    vs = v1 + v1
    assert vs[0] == 0.5
    assert vs[1] == -0.5
    assert vs[2] == 0.0
    assert vs[3] == 2.0


def test_vector_3():
    v1 = construct_v1()
    v1[0] = 0.0
    assert v1[0] == 0.0
    v2 = construct_v2()
    v1.add_with(v2)
    assert v1[0] == 0.25
    assert v1[1] == 0.0
    assert v1[2] == 0.5
    assert v1[3] == 1.5


def test_vector_4():
    v1 = construct_v1()
    v1.apply_relu()
    assert v1[0] == 0.25
    assert v1[1] == 0.0
    assert v1[2] == 0.0
    assert v1[3] == 1.0
