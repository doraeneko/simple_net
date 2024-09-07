#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Rescaling layer class
#################################################################

import pytest
from .common import *
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling
from ..simple_net.rescaling_layer import RescalingLayer
from ..keras2simple_net import transfer_batch_size_1_np_tensor_to_tensor3d


@pytest.mark.parametrize(
    "scale, x, y, channels",
    [
        (2.0, 200, 30, 5),
        (0.5, 10, 10, 3),
        (1/128.0, 10, 10, 3),
    ],
)
def test_flatten_layer(
    scale, x, y, channels
):
    np.random.seed(42)
    tf.random.set_seed(42)
    # neural network with one conv2d layer
    keras_model = Sequential(
        [
            Rescaling(scale)
        ]
    )
    keras_model.compile()
    simple_net_instance = RescalingLayer(scaling_factor=scale)
    # Test random example value
    keras_input = np.random.rand(1, x, y, channels)
    keras_output = keras_model.predict(keras_input)
    simple_net_input = transfer_batch_size_1_np_tensor_to_tensor3d(keras_input)
    simple_net_output = simple_net_instance.output(simple_net_input)
    compare_outputs(keras_output, simple_net_output)
