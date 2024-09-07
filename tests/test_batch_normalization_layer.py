#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for BatchNormalizationLayer class
#################################################################

import pytest
from .common import *
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from ..keras2simple_net import (
    transfer_batch_size_1_np_tensor_to_tensor3d,
    transfer_batch_normalization_layer,
)


@pytest.mark.parametrize(
    "x, y, channels",
    [
        (10, 10, 3),
        (200, 200, 10),
        (10, 10, 100),
    ],
)
def test_batch_normalization_layer(x, y, channels):
    np.random.seed(42)
    tf.random.set_seed(42)
    keras_model = models.Sequential()
    keras_model.add(layers.BatchNormalization(input_shape=(x, y, channels)))
    keras_model.summary()
    simple_net_instance = transfer_batch_normalization_layer(keras_model.layers[0])
    keras_input = np.random.rand(1, x, y, channels)
    keras_output = keras_model.predict(keras_input)
    simple_net_input = transfer_batch_size_1_np_tensor_to_tensor3d(keras_input)
    simple_net_output = simple_net_instance.output(simple_net_input)
    compare_outputs(keras_output, simple_net_output)
