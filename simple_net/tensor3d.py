#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tensor3D implementation
#################################################################

import typing
from .matrix import Matrix


class Tensor3D:
    """Simple implementation of a 3D-tensor, based on Matrix implementation."""

    def __init__(self, c, x, y):
        """Construct a 3D-tensor having dimensions (c,x,y) (c normally referred to as channels)."""
        self._shape = (c, x, y)
        self._vals = list(Matrix(x, y) for _ in range(c))

    def shape(self) -> typing.Tuple[int, int, int]:
        """Return the shape / dimensions of the tensor."""
        return self._shape

    def __getitem__(self, cell: typing.Tuple[int, int, int]) -> float:
        """Get tensor cell."""
        c, x, y = cell
        return self._vals[c][x, y]

    def __setitem__(self, cell: typing.Tuple[int, int, int], value: float):
        """Set tensor cell."""
        c, x, y = cell
        self._vals[c][x, y] = value

    def set_slice(self, c, m: Matrix):
        """Set the entries of an entire channel (i.e., self[c,_,_] := m). Modifies self."""
        assert m.shape() == (self._shape[1], self._shape[2])
        assert 0 <= c < self._shape[0]
        self._vals[c] = m

    def get_slice(self, c) -> Matrix:
        """Get the entries of an entire channel (i.e., self[c,_,_]) as reference."""
        assert 0 <= c < self._shape[0]
        return self._vals[c]

    def min_value(self):
        min_val = None
        for c in range(self._shape[0]):
            if min_val is None:
                min_val = self.get_slice(c).min_entry()[1]
                continue
            current = self.get_slice(c).min_entry()[1]
            if min_val > current:
                min_val = current
        return min_val

    def max_value(self):
        max_val = None
        for c in range(self._shape[0]):
            if max_val is None:
                max_val = self.get_slice(c).max_entry()[1]
                continue
            current = self.get_slice(c).max_entry()[1]
            if max_val < current:
                max_val = current
        return max_val

    def __str__(self) -> str:
        """Output the tensor as string."""
        result = "[[\n"
        for c in self._vals:
            result += "%s\n" % str(c)
        result += "]]"
        return result
