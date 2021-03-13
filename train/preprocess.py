"""
preprocess.py

Preprocessing routines for audio sample normalization and spectrogram
generation.
"""


import pathlib
import tensorflow as tf
import math
import os
from functools import partial
from typing import Sequence


AUTOTUNE = tf.data.AUTOTUNE


def download_speech_dataset(data_dir: pathlib.Path) -> pathlib.Path:
    """
    Download the speech dataset, if it was not already downloaded.

    :return: path to the extracted speech data.
    """
    if (speech_data_dir := data_dir.joinpath("mini_speech_commands")).exists():
        return speech_data_dir

    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=".",
        cache_subdir=str(data_dir),
    )

    return speech_data_dir


def split_train_test_validate(
    filenames: Sequence, train: int = 80, test: int = 10, validate: int = 10
) -> tuple[Sequence, Sequence, Sequence]:
    """
    Split the audio filenames into train, test, and validate segments.

    :param filenames: filenames corresponding to audio training data files.
    :param train: percentage of files to use for the train dataset.
    :param test: percentage of files to use for the test dataset.
    :param validate: percentage of files to use for the validation dataset.
    """
    if train + test + validate != 100:
        raise ValueError("train + test + validate percentages != 100%")

    total = len(filenames)
    traind = math.floor((train / 100.0) * total)
    vald = math.floor((validate / 100.0) * total)

    return (
        filenames[:traind],
        filenames[traind + vald :],
        filenames[traind : traind + vald],
    )


def load_wav(path: tf.Tensor) -> tf.Tensor:
    """
    Load audio data encoded in a mono 16-bit PCM WAV file into
    a floating point tensor, normalizing the sample values into the
    range ``[-1, 1]``.

    The audio data must have a sample rate of 16000 Hz.

    :param path: path to the WAV file.
    """
    audio, _ = tf.audio.decode_wav(path, desired_channels=1)
    return tf.squeeze(audio, axis=1)


def get_label(file_path: str) -> tf.RaggedTensor:
    """
    Obtain the label of a sample identified by its path.

    :param file_path: sample path relative to the directory containing
        the data directory.
    """
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_data_point(file_path: str) -> tuple[tf.Tensor, tf.RaggedTensor]:
    """
    Obtain a data point, which is a tuple containing of the sample's label
    plus the sample's data, encoded in a float tensor according to
    ``load_wav()``.

    :param file_path: sample's path relative to the directory containing
        the data directory.
    """
    label = get_label(file_path)
    audio = tf.io.read_file(file_path)
    waveform = load_wav(audio)

    return waveform, label


def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    """
    Obtain the spectrogram for an audio waveform.

    The audio must be represented as a Tensor of rank 1, and contain
    at most 16000 samples at a sample rate of 16000Hz.

    Each sample must be in the ``float32`` data format.

    :return: generated spectrogram.
    """
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def get_spectrogram_and_label_id(
    audio: tf.Tensor, label: tf.RaggedTensor, commands: tf.Tensor
) -> tuple[tf.Tensor, tf.RaggedTensor]:
    """
    Obtain a tuple containing the spectrogram of the audio
    data coupled with the numeric label.

    :param audio: audio data corresponding to a particular label.
    :param label: (string) data label.
    :param commands: tensor mapping data label to numeric label.
    """
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)

    return spectrogram, label_id


def preprocess_dataset(files: Sequence[str], commands: Sequence[str]):
    """
    Preprocess a given dataset.

    :param files: filenames contained within the dataset.
    :param commands: speech commands associated with dataset files.
    """
    commands = tf.constant(commands, dtype=tf.string)
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_data_point, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        partial(get_spectrogram_and_label_id, commands=commands),
        num_parallel_calls=AUTOTUNE,
    )
    return output_ds


def transform_spectrogram_for_inference(spectrogram: tf.Tensor) -> tf.Tensor:
    """
    Reshape a spectogram as returned by ``get_spectrogram()`` for inference
    as compared to training.

    :param spectrogram: spectrogram to transform.
    :return: reshaped tensor that can be used as an input to ``predict()``.
    """
    shape = spectrogram.shape.as_list()
    shape = (1, *shape, 1)
    return tf.reshape(spectrogram, shape)
