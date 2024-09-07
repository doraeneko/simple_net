#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Tensor3D class
#################################################################

from .common import *


def test_tensor3d_1():
    t1 = construct_t1()
    assert t1.shape() == (2, 3, 3)

    assert t1[0, 1, 1] == 1.25
    assert t1[1, 0, 2] == -1.0
    assert t1[1, 2, 2] == 0.75
    t1[1, 2, 2] += 0.25
    assert t1[1, 2, 2] == 1.0


def test_tensor3d_2():
    t1 = construct_t1()
    s1 = t1.get_slice(0)
    assert s1[1, 1] == 1.25
    s1[1, 1] += 0.75
    assert t1[0, 1, 1] == 2.0
    m = Matrix(3, 3)
    m[0, 0] = 42.0
    t1.set_slice(1, m)
    assert t1[1, 0, 0] == 42.0
