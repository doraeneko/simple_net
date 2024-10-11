#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/25
# (C) Andreas Gaiser (doraeneko@github)
# Sequential layer abstraction
#################################################################

from .tensor3d import Tensor3D
from .layer import Layer


class Sequential(Layer):
    def __init__(self):
        self._seq = []

    def push_layer(self, layer: Layer):
        self._seq.append(layer)

    def output(self, input: Tensor3D) -> Tensor3D:
        result = input
        for layer in self._seq:
            # print("Executing layer %s" % layer)
            result = layer.output(result)
            # print("Intermediate result shape: %s" % str(result.shape()))
            # for c in range(result.shape()[0]):
            #    print(
            #        "Channel %s: min: %s, max: %s"
            #        % (
            #            c,
            #            result.get_slice(c).min_entry()[1],
            #            result.get_slice(c).max_entry()[1],
            #        )
            #    )
        return result

    def __str__(self):
        result = "[\n"
        for layer in self._seq:
            result += "%s\n" % layer
        result += "]"
        return result

    def __del__(self):
        pass
