#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Dense1DLayer class
#################################################################

import pytest
from .common import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D
from ..keras2simple_net import transfer_dense_layer

@pytest.mark.parametrize(
    "size, units, apply_relu",
    [ (10, 20, True),
      (10, 10, True),
      (100, 5, False),
      (100, 15, False)
    ]
)
def test_dense1d_layer(size, units, apply_relu):
    np.random.seed(42)
    # neural network with one dense layer
    keras_model = Sequential()
    if apply_relu:
        keras_model.add(
            Dense(units=units, input_dim=size, activation="relu")
        )
    else:
        keras_model.add(
            Dense(units=units, input_dim=size)
        )
    keras_model.compile(optimizer="adam", loss="mean_squared_error")
    simple_net_instance = transfer_dense_layer(keras_model.layers[0])
    # Example value

    keras_input = np.random.rand(1, size)
    keras_output = keras_model.predict(keras_input)

    simple_net_input = Tensor3D(1, 1, size)
    i = 0
    for v in keras_input[0]:
        simple_net_input[0, 0, i] = v
        i += 1
    simple_net_out = simple_net_instance.output(simple_net_input)
    i = 0
    for v in keras_output[0]:
        assert abs(v - simple_net_out[0, 0, i]) < 0.001
        i += 1
    keras_weights, keras_biases = keras_model.layers[0].get_weights()
    assert keras_weights[0, 0] == simple_net_instance.weights()[0, 0]
    assert keras_weights[0, 1] == simple_net_instance.weights()[0, 1]
    assert keras_weights[0, 2] == simple_net_instance.weights()[0, 2]
    assert keras_weights[1, 1] == simple_net_instance.weights()[1, 1]
    assert keras_biases[0] == simple_net_instance.biases()[0]
    assert keras_biases[1] == simple_net_instance.biases()[1]
    assert keras_biases[2] == simple_net_instance.biases()[2]
    assert keras_biases[3] == simple_net_instance.biases()[3]
