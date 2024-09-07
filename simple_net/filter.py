#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Abstract classes for filters
#################################################################
import abc
import typing
import functools

from .matrix import Matrix


class SamePaddingFilter:
    def __init__(self, fx, fy):
        """Init the filter with kernel dimensions x and y."""
        self._fx = fx
        self._fy = fy
        self._x_pad = (fx - 1) // 2
        self._y_pad = (fy - 1) // 2

    def fx(self):
        return self._fx

    def fy(self):
        return self._fx

    def get_padded_input(self, in_x, in_y, input: Matrix, x, y):
        """in_x, in_y refer to input's shape dimensions (= input.shape())"""
        # https://stackoverflow.com/questions/51131821/even-sized-kernels-with-same-padding-in-tensorflow
        if x < self._x_pad or x - self._x_pad >= in_x:
            return 0.0
        if y < self._y_pad or y - self._y_pad >= in_y:
            return 0.0
        return input[x - self._x_pad, y - self._y_pad]

    @abc.abstractmethod
    def output(self, channel, input: Matrix):
        pass
