#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Base class for network layers
#################################################################

import abc
from .tensor3d import Tensor3D


class Layer(abc.ABC):
    """Base class for network layers."""

    @abc.abstractmethod
    def output(self, t: Tensor3D) -> Tensor3D:
        """Each layer has to provide this output method, transforming a tensor into another tensor. Even
        dense layers get their input in tensor form."""
        pass
