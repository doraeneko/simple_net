#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Batch normalization layer implementation
#################################################################
import typing


from .vector import Vector
from .tensor3d import Tensor3D
from .layer import Layer
import math


class BatchNormalizationLayer(Layer):
    """Batch normalization layer."""

    def __init__(self, channels: int, epsilon: float = 1e-3):
        self._channels = channels
        self._gammas = Vector(channels)
        self._betas = Vector(channels)
        self._running_means = Vector(channels)
        self._running_variances = Vector(channels)
        self._epsilon = epsilon

    def gammas(self):
        """Return reference to gamma vector."""
        return self._gammas

    def betas(self):
        """Return reference to beta vector."""
        return self._betas

    def running_means(self):
        """Return reference to running means vector."""
        return self._running_means

    def running_variances(self):
        """Return reference to running variances vector."""
        return self._running_variances

    def output(self, input: Tensor3D) -> Tensor3D:
        c, x, y = input.shape()
        result = Tensor3D(c, x, y)
        for i in range(c):
            gamma = self.gammas()[i]
            beta = self.betas()[i]
            mean = self.running_means()[i]
            variance = self.running_variances()[i]
            scale_denominator = math.sqrt(variance + self._epsilon)
            for j in range(x):
                for k in range(y):
                    norm_1 = (input[i, j, k] - mean) / scale_denominator
                    norm_2 = gamma * norm_1 + beta
                    result[i, j, k] = norm_2
        return result

    def __str__(self):
        return "BatchNormalization (channels: %s, eps: %s)" % (
            self._channels,
            self._epsilon,
        )
