#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/25
# (C) Andreas Gaiser (doraeneko@github)
# Rescaling layer implementation
#################################################################

from .tensor3d import Tensor3D
from .layer import Layer


class RescalingLayer(Layer):
    def __init__(self, scaling_factor: float):
        self._s = scaling_factor

    def output(self, input: Tensor3D):
        c, x, y = input.shape()
        result = Tensor3D(c, x, y)
        for pos_c in range(c):
            for pos_x in range(x):
                for pos_y in range(y):
                    result[pos_c, pos_x, pos_y] = input[pos_c, pos_x, pos_y] * self._s
        return result

    def __str__(self):
        return "Rescaling (scale: %s)" % self._s
