#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Vector implementation
#################################################################

import typing
import array


class Vector:
    """Implementation of a float-valued vector, using Python arrays."""

    def __init__(self, dim):
        """Create a new dim-dimensional vector, initialized with 0.0-values."""
        self._shape = (dim,)
        self._vals = array.array("f", [0.0] * dim)

    def shape(self) -> typing.Tuple[int]:
        """Return the shape of the vector as 1D-tuple."""
        return self._shape

    def dim(self) -> int:
        """Return the dimension of the vector."""
        return self._shape[0]

    def copy(self) -> "Vector":
        """Create a deep copy of the vector."""
        result = Vector(self._shape[0])
        result._vals = self._vals[:]
        return result

    def __getitem__(self, item: int):
        """Get a component of the vector."""
        return self._vals[item]

    def __setitem__(self, key: int, value: float):
        """Set a component of the vector."""
        self._vals[key] = value

    def apply_relu(self):
        """Apply standard RELU to the components of the vector (modifies self)."""
        for i in range(len(self._vals)):
            self._vals[i] = max(0.0, self._vals[i])

    def add_with(self, other):
        """Perform self _= self+other (modifies self)."""
        assert other.shape() == self.shape()
        for i in range(self.dim()):
            self[i] += other[i]

    def __add__(self, other) -> "Vector":
        """Return self+other."""
        assert other.shape() == self.shape()
        result = Vector(self.dim())
        for i in range(self.dim()):
            result[i] += self[i] + other[i]
        return result

    def min_index_and_value(self) -> typing.Tuple[int, float]:
        """Return (x, v) with self[x] minimal and self[x] == v."""
        min_index = -1
        min_val = None
        for i in range(len(self._vals)):
            if min_index == -1 or self._vals[i] < self._vals[min_index]:
                min_index = i
                min_val = self._vals[i]
        return min_index, min_val

    def max_index_and_value(self) -> typing.Tuple[int, float]:
        """Return (x, v) with self[x] maximal and self[x] == v."""
        max_index = -1
        max_val = None
        for i in range(len(self._vals)):
            if max_index == -1 or self._vals[i] > self._vals[max_index]:
                max_index = i
                max_val = self._vals[i]
        return max_index, max_val

    def __str__(self) -> str:
        """Output vector in textual format."""
        return "[" + ",".join(str(v) for v in self._vals) + "]"
