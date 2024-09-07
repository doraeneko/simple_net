#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Dense layer implementation
#################################################################


from .vector import Vector
from .matrix import Matrix
from .tensor3d import Tensor3D
from .layer import Layer


class Dense1DLayer(Layer):
    """Simple dense 1D layer."""

    def __init__(self, input_dim, output_dim, apply_relu=True):
        """Create a layer with input_dim*output_dim + output_dim many weights."""
        self._w = Matrix(input_dim, output_dim)
        self._b = Vector(output_dim)
        self._apply_relu = apply_relu

    def input_dim(self) -> int:
        """Return input dimension."""
        return self._w.shape()[0]

    def output_dim(self) -> int:
        """Return output dimension."""
        return self._w.shape()[1]

    def weights(self) -> Matrix:
        """Return weights by reference."""
        return self._w

    def biases(self) -> Vector:
        """Return bias by reference."""
        return self._b

    def set_all_weights(self, m: Matrix):
        """Replace weights by the matrix m. Modifies self."""
        assert m.shape() == self._w.shape()
        self._w = m

    def output(self, t: Tensor3D) -> Tensor3D:
        c, x, y = t.shape()
        assert y == self.input_dim()
        result = Tensor3D(c, x, self.output_dim())
        for pos_c in range(c):
            input_slice = t.get_slice(pos_c)
            result_slice = result.get_slice(pos_c)
            for pos_x in range(x):
                input_vector = input_slice.get_column_vector(pos_x)
                out = self._w.multiply_vec_right(input_vector)
                out.add_with(self._b)
                if self._apply_relu:
                    out.apply_relu()
                result_slice.set_column_vector(pos_x, out)
        return result

    def __str__(self):
        return "Dense (input dim: %s, output dim: %s, apply relu: %s)" % (
            str(self.input_dim()),
            str(self.output_dim()),
            self._apply_relu,
        )
