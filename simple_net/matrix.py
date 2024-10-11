#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Matrix implementation
#################################################################

import typing
import array
from .vector import Vector


class Matrix:
    """Implementation of a float-valued matrix (2D tensor), using standard Python arrays internally."""

    def __init__(self, dim_x, dim_y):
        """Construct a dim_x X dim_y matrix."""
        self._shape = (dim_x, dim_y)
        self._x = dim_x
        self._y = dim_y
        self._vals = array.array("f", [0.0] * dim_x * dim_y)

    def shape(self) -> typing.Tuple[int, int]:
        """Return shape of the matrix."""
        return self._shape

    def __getitem__(self, cell: typing.Tuple[int, int]) -> float:
        """Get matrix cell."""
        x, y = cell
        return self._vals[x * self._y + y]

    def __setitem__(self, cell: typing.Tuple[int, int], value: float):
        """Set matrix cell."""
        x, y = cell
        self._vals[x * self._y + y] = value

    def min_entry(self) -> typing.Tuple[typing.Tuple[int, int], float]:
        """Return (x, y), v with self[x,y] == v minimal."""
        min_index = None
        min_val = None
        for x in range(self._x):
            for y in range(self._y):
                if min_index is None or self[x, y] < min_val:
                    min_index = (x, y)
                    min_val = self[x, y]
        return min_index, min_val

    def max_entry(self) -> typing.Tuple[typing.Tuple[int, int], float]:
        """Return (x, y), v with self[x,y] == v maximal."""
        max_index = None
        max_val = None
        for x in range(self._x):
            for y in range(self._y):
                if max_index is None or self[x, y] > max_val:
                    max_index = (x, y)
                    max_val = self[x, y]
        return max_index, max_val

    def apply(self, cell_transformer: typing.Callable[[float], float]):
        """Replace original value v of each matrix cell by celly_transformer(v). Modifies self."""
        for x in range(self._x):
            for y in range(self._y):
                self[x, y] = cell_transformer(self[x, y])

    def get_column_vector(self, x) -> Vector:
        """Return a copy of the column vector self[x, _]."""
        result = Vector(self._y)
        for y in range(self._y):
            result[y] = self[x, y]
        return result

    def set_column_vector(self, x, v: Vector) -> Vector:
        """Set column vector self[x, _] to v. Modifies self. Uses v as reference."""
        for y in range(self._y):
            self[x, y] = v[y]

    def apply_relu(self):
        """Apply RELU to each matrix cell. Modifies self."""
        self.apply(lambda v: max(0.0, v))

    def elementwise_add_with(self, scalar):
        """Add a constant to each matrix cell. Modifies self."""
        self.apply(lambda v: v + scalar)

    def multiply_vec_right(self, v: Vector) -> Vector:
        """Compute and return self * v."""
        (dim_x, dim_y) = self._shape
        assert v.dim() == dim_x
        result = Vector(dim_y)
        y = 0
        while y < dim_y:
            x = 0
            val = 0.0
            while x < dim_x:
                val += self[x, y] * v[x]
                x = x + 1
            result[y] = val
            y = y + 1
        return result

    def add_with(self, m: "Matrix"):
        """Perform component-wise addition (self + m). Modifies self."""
        (dim_x, dim_y) = self.shape()
        assert (dim_x, dim_y) == m.shape()
        for x in range(dim_x):
            for y in range(dim_y):
                self[x, y] = self[x, y] + m[x, y]

    def __str__(self) -> str:
        """Output matrix contents as string."""
        result = "[\n"
        i = 0
        for col in self._vals:
            j = 0
            for row in col:
                result += "w[%s, %s] = %s\n" % (i, j, row)
                j += 1
            i += 1
        result += "]"
        return result
