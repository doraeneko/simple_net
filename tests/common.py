#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests, helpers
#################################################################

from ..simple_net.matrix import Matrix
from ..simple_net.vector import Vector
from ..simple_net.tensor3d import Tensor3D


def compare_outputs(keras_output, simple_net_output: Tensor3D):
    (c, x, y) = simple_net_output.shape()
    for pos_c in range(c):
        for pos_x in range(x):
            for pos_y in range(y):
                if (
                    abs(
                        simple_net_output[pos_c, pos_x, pos_y]
                        - keras_output[0, pos_x, pos_y, pos_c]
                    )
                    > 0.001
                ):
                    print(
                        "Output differs too much: (%s, %s); %s vs. %s"
                        % (
                            pos_x,
                            pos_y,
                            simple_net_output[pos_c, pos_x, pos_y],
                            keras_output[0, pos_x, pos_y, pos_c],
                        )
                    )
                    assert False


def construct_v1():
    v1 = Vector(4)
    v1[0] = 0.25
    v1[1] = -0.25
    v1[2] = 0.0
    v1[3] = 1.0
    return v1


def construct_v2():
    v2 = Vector(4)
    v2[0] = 0.25
    v2[1] = 0.25
    v2[2] = 0.5
    v2[3] = 0.5
    return v2


def construct_m1():
    m = Matrix(3, 4)
    m[0, 0] = 0.25
    m[0, 1] = -0.25
    m[0, 2] = 0.0
    m[0, 3] = 1.0
    m[1, 0] = 0.25
    m[1, 1] = -0.25
    m[1, 2] = 0.0
    m[1, 3] = 1.0
    m[2, 0] = 0.0
    m[2, 1] = 0.0
    m[2, 2] = 0.0
    m[2, 3] = 0.0
    return m


def construct_m2():
    m = Matrix(4, 2)
    m[0, 0] = 0.25
    m[0, 1] = -0.25
    m[1, 0] = 0.0
    m[1, 1] = 1.0
    m[2, 0] = 0.25
    m[2, 1] = -0.25
    m[3, 0] = 0.0
    m[3, 1] = 1.0
    return m


def construct_t1():
    t = Tensor3D(2, 3, 3)
    t[0, 0, 0] = 1.0
    t[0, 0, 1] = -0.25
    t[0, 0, 2] = 0.75
    t[0, 1, 0] = 1.0
    t[0, 1, 1] = 1.25
    t[0, 1, 2] = 1.75
    t[0, 2, 0] = 0.5
    t[0, 2, 1] = 0.25
    t[0, 2, 2] = 0.75
    t[1, 0, 0] = 0.0
    t[1, 0, 1] = 0.0
    t[1, 0, 2] = -1.0
    t[1, 1, 0] = 1.0
    t[1, 1, 1] = 0.0
    t[1, 1, 2] = 0.25
    t[1, 2, 0] = 0.0
    t[1, 2, 1] = 0.0
    t[1, 2, 2] = 0.75
    return t
