#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Conv2D layer implementation
#################################################################

import typing
from .matrix import Matrix
from .tensor3d import Tensor3D
from .layer import Layer
from .filter import SamePaddingFilter

MULTIPROCESSING = False


class Conv2DFilter(SamePaddingFilter):
    def __init__(self, channels, dim_x, dim_y):
        """Create a single Conv2D filter for the given number of channels and kernel size (dim_x, dim_y).
        Only "same" padding is supported (and therefore implicitly assumed), and stride is always 1."""
        super(Conv2DFilter, self).__init__(dim_x, dim_y)
        self._w = [Matrix(dim_x, dim_y) for _ in range(channels)]
        self._number_entries = dim_x * dim_y
        self._bias = 0.0

    def set_bias(self, bias):
        """Set bias value."""
        self._bias = bias

    def get_bias(self):
        """Get bias value."""
        return self._bias

    def weights(self, channel) -> Matrix:
        """Return weight matrix of the given channel. Returns a reference."""
        return self._w[channel]

    def output(self, channel, input: Matrix):
        in_x, in_y = input.shape()
        output = Matrix(in_x, in_y)
        o_x, o_y = output.shape()
        f_x, f_y = self._w[0].shape()

        for x in range(o_x):
            for y in range(o_y):
                val = 0.0
                for filter_pos_x in range(f_x):
                    for filter_pos_y in range(f_y):
                        new_factor = self._w[channel][
                            filter_pos_x, filter_pos_y
                        ] * self.get_padded_input(
                            in_x, in_y, input, x + filter_pos_x, y + filter_pos_y
                        )
                        val += new_factor
                output[x, y] = val
        return output

    def __str__(self):
        def same_sign(x, y):
            return x >= 0 and y >= 0 or x <= 0 and y <= 0

        result = "[[\n"
        c = 0
        for f in self._w:
            result += (
                "Channel %s: max value: %s, min value: %s, same sign: %s bias: %s"
                % (
                    c,
                    f.max_entry()[1],
                    f.min_entry()[1],
                    same_sign(f.max_entry()[1], f.min_entry()[1]),
                    self._bias,
                )
            )
            c += 1
        return result


class Conv2DLayer(Layer):
    """Conv2D layer."""

    def __init__(
        self, channels: int, number_filters: int, filter_shape: typing.Tuple[int, int]
    ):
        self._filter_shape = filter_shape
        f_x, f_y = filter_shape
        self._channels = channels
        self._filters = list(
            Conv2DFilter(channels, f_x, f_y) for _ in range(number_filters)
        )

    def filter(self, index) -> Conv2DFilter:
        return self._filters[index]

    class FilterResultComputation:
        """Helper class to perform multiprocessing step for filters."""

        def __init__(self):
            pass

        def __call__(self, index, filter: Conv2DFilter, input: Tensor3D):
            c_dim, x_dim, y_dim = input.shape()
            in_min = input.min_value()
            in_max = input.max_value()
            channel_bounds = []
            # compute approximation here
            for channel in range(c_dim):
                f_min = filter.weights(channel).min_entry()[1]
                f_max = filter.weights(channel).max_entry()[1]
                channel_bound = (
                    x_dim
                    * y_dim
                    * max(
                        f_min * in_min, f_min * in_max, f_max * in_min, f_max * in_max
                    )
                )
                channel_bounds.append(channel_bound)

            filter_result = None
            for channel in range(c_dim):
                # compute a feature map for each channel
                feature_map = filter.output(channel, input.get_slice(channel))

                # sum them up component-wise
                if filter_result is None:
                    filter_result = feature_map
                else:
                    filter_result.add_with(feature_map)
            # add bias
            filter_result.elementwise_add_with(filter.get_bias())
            filter_result.apply_relu()
            return index, filter_result

    def output(self, input: Tensor3D) -> Tensor3D:

        c, x, y = input.shape()
        result = Tensor3D(len(self._filters), x, y)
        if not MULTIPROCESSING:
            for (index, filter) in enumerate(self._filters):
                _, slice = Conv2DLayer.FilterResultComputation()(index, filter, input)
                result.set_slice(index, slice)
        else:
            from multiprocessing import Pool

            # thanks to https://superfastpython.com/multiprocessing-pool-starmap/
            with Pool() as pool:
                for (index, slice) in pool.starmap(
                    Conv2DLayer.FilterResultComputation(),
                    [
                        (index, filter, input)
                        for (index, filter) in enumerate(self._filters)
                    ],
                ):
                    result.set_slice(index, slice)
        return result

    def __str__(self):
        result = "Conv2D (channels: %s, filters: %s, filter shape: %s)" % (
            self._channels,
            len(self._filters),
            str(self._filter_shape),
        )
        for filter in self._filters:
            result += str(filter) + "\n"
        return result
