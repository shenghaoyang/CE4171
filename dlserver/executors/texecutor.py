"""
iexecutor.py

Implementation details of the training executor.
"""

import pathlib
import numpy as np
import tensorflow as tf
import logging
from tensorflow import keras
from train.preprocess import get_spectrogram, transform_spectrogram_for_inference


model: keras.Model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init(model_path: pathlib.Path, ):
    """
    Initialization function for the training executors.

    Should only be called from an inference executor.

    :param model_path: path to the model that the executors should load.
    """
    global model
    model = keras.models.load_model(model_path)


def train(data_label: list[np.ndarray, int]):
    """
    Run training on the audio samples.

    Should only be called from the training executor.
    """
    global model



    tensor = tf.constant(audio_samples, dtype=tf.float32)
    spect = transform_spectrogram_for_inference(get_spectrogram(tensor))
    label = tf.argmax(model.predict(spect), axis=1)[0]

    return label
