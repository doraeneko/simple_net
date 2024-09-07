#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/25
# (C) Andreas Gaiser (doraeneko@github)
# Flatten (3D->1D) layer implementation
#################################################################

from .tensor3d import Tensor3D
from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self):
        pass

    def output(self, input: Tensor3D):
        c, x, y = input.shape()
        result = Tensor3D(1, 1, c * x * y)
        for pos_channel in range(c):
            for pos_x in range(x):
                for pos_y in range(y):
                    result[0, 0, pos_channel + c * pos_y + c * y * pos_x] = input[
                        pos_channel, pos_x, pos_y
                    ]
        return result

    def __str__(self):
        return "Flatten"
