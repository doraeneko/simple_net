#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/25
# (C) Andreas Gaiser (doraeneko@github)
# Max pooling (2D) layer implementation
#################################################################

import typing
from .matrix import Matrix
from .tensor3d import Tensor3D
from .layer import Layer
from .filter import SamePaddingFilter


class Max2DFilter(SamePaddingFilter):
    def __init__(self, dim_x: int, dim_y: int, stride_x: int, stride_y: int):
        super(Max2DFilter, self).__init__(dim_x, dim_y)
        self._sx = stride_x
        self._sy = stride_y

    def output(self, input: Matrix):
        in_x, in_y = input.shape()
        output = Matrix(max(1, in_x // self._sx), max(1, in_y // self._sy))
        o_x, o_y = output.shape()
        for x in range(o_x):
            for y in range(o_y):
                max_val = None
                for filter_pos_x in range(self.fx()):
                    for filter_pos_y in range(self.fy()):
                        current_val = self.get_padded_input(
                            in_x,
                            in_y,
                            input,
                            x * self._sx + filter_pos_x,
                            y * self._sy + filter_pos_y,
                        )
                        if max_val is None or max_val < current_val:
                            max_val = current_val
                assert max_val is not None
                output[x, y] = max_val
        return output

    def __str__(self):
        return "[[ MAX_FILTER(%s, %s) ]]\n" % (self.fx(), self.fy())


class MaxPooling2DLayer(Layer):
    def __init__(
        self, pool_size: typing.Tuple[int, int], strides: typing.Tuple[int, int]
    ):
        self._sx, self._sy = strides
        self._pool_size = pool_size
        self._filter = Max2DFilter(pool_size[0], pool_size[1], self._sx, self._sy)

    def output(self, input: Tensor3D):
        c, x, y = input.shape()
        result = Tensor3D(c, max(1, x // self._sx), max(1, y // self._sy))
        for channel_index in range(c):
            filtered = self._filter.output(input.get_slice(channel_index))
            result.set_slice(channel_index, filtered)
        return result

    def __str__(self):
        return "MaxPooling2D (pool size: %s, strides: %s)" % (
            str(self._pool_size),
            str((self._sx, self._sy)),
        )
