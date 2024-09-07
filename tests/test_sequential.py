#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Sequential abstraction
#################################################################

import pytest
from .common import *
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Rescaling
from ..simple_net.rescaling_layer import RescalingLayer
from ..keras2simple_net import (
    transfer_batch_size_1_np_tensor_to_tensor3d,
    transfer_sequential,
)
from keras.saving import load_model
from tensorflow.keras.preprocessing import image


def test_sequential_1():
    from pathlib import Path

    test_dir = Path(__file__).resolve().parent
    model_path = test_dir / "model2.keras"
    model = load_model(model_path)
    simple_net_seq = transfer_sequential(model)
    print(simple_net_seq)
    img_path = test_dir / "images/test_000001.png"
    img = image.load_img(img_path, color_mode="grayscale", target_size=(30, 30))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    keras_output = model.predict([img_array])[0]
    keras_output = np.reshape(keras_output, (1, 1, -1, 1))

    (_, x, y, c) = img_array.shape
    simplenet_img_array = np.reshape(img_array, (1, x, y, c))
    simplenet_img_array = transfer_batch_size_1_np_tensor_to_tensor3d(
        simplenet_img_array
    )

    import time

    start = time.process_time()
    simple_net_output = simple_net_seq.output(simplenet_img_array)
    print(time.process_time() - start)

    compare_outputs(keras_output, simple_net_output)

    # import pickle
    # with open(test_dir / "kanji_recognizer.pickle", "wb") as save_file:
    #    pickle.dump(simple_net_seq, save_file)
