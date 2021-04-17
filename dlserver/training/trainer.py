"""
trainer.py

Module containing routines needed to train the model used for voice recognition.
"""


import numpy as np

from dlserver.preprocess import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models

from functools import cache
from itertools import chain
from pathlib import Path
from collections.abc import Sequence


class Trainer:
    """
    Trainer class that sets up the training environment and allows for training to proceed.
    """
    def __init__(self, data_paths: Sequence[Path], batch_size: int = 16, epochs: int = 20):
        """
        Initialize a trainer.

        :param data_paths: paths to data directories.
            Each path in this Sequence must be a path to a directory containing subfolders named
            with label names, with each subfolder containing 16-bit 16000Hz mono WAV files of
            less than or equal to a second each corresponding to the audio that is to be matched
            with the subfolder's name.

            Other entries in the top-level directories that are not folders are ignored.
        :param batch_size: training batch size in number of audio samples per batch.
        :param epochs: number of training epochs.
            Note that training may stop early because of the early-stopping mechanism.
        """
        self._batch_size = batch_size
        self._epochs = epochs

        command_names = set(
            map(lambda p: p.name,
                filter(lambda p: p.is_dir(),
                       chain.from_iterable(data_path.iterdir() for data_path in data_paths))))
        self._commands = sorted(command_names)

        filenames = []
        for data_path in data_paths:
            pattern = str(data_path.joinpath('*/*'))
            filenames.extend(tf.io.gfile.glob(pattern))

        np.random.shuffle(filenames)
        self._training_filenames = filenames

        self._model = self._build_model()
        self._history = None

    def _build_model(self) -> tf.keras.Model:
        """
        Build the model for training.
        """
        # Split the dataset into train, test, validate components
        splits = split_train_test_validate(self.samples)

        # Preprocess dataset to generate FFTs
        train_ds, val_ds, test_ds = tuple(
            map(lambda files: preprocess_dataset(files, self.commands), splits))

        # Pre-calculate the input shape for entry into the model.
        self._input_shape = next(iter(map(lambda t: t[0].shape, train_ds.take(1))))

        # Batch and configure prefetching and caching for data reads.
        train_ds = train_ds.batch(self.batch_size)
        val_ds = val_ds.batch(self.batch_size)
        self._test_ds = test_ds
        self._train_ds = train_ds.cache().prefetch(AUTOTUNE)
        self._val_ds = val_ds.cache().prefetch(AUTOTUNE)

        num_labels = len(self.commands)
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(train_ds.map(lambda x, _: x))

        model = models.Sequential([
            layers.InputLayer(input_shape=self._input_shape),
            preprocessing.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        return model

    @property
    def batch_size(self) -> int:
        """
        Obtain the batch size used for training.

        :return: batch size.
        """
        return self._batch_size

    @property
    def epochs(self) -> int:
        """
        Obtain the number of training epochs.

        :return: number training epochs.
        """
        return self._epochs

    @property
    def commands(self) -> Sequence[str]:
        """
        Obtain the commands (labels) read from the data paths.

        :return: sequence of commands read.
        """
        return self._commands

    @property
    def samples(self) -> Sequence[str]:
        """
        Obtain paths to the audio samples that will be used for training.

        :return: paths to audio samples.
        """
        return self._training_filenames

    @property
    def model(self) -> tf.keras.Model:
        """
        Obtain the model created.

        :return: created model.
        """
        return self._model

    def train(self):
        """
        Train the model on the provided samples.

        No-op if already trained.
        """
        if self._history is not None:
            return

        self._history = self.model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=self.epochs,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

    def save(self, path: Path):
        """
        Save the model to the provided path.

        The model must have been trained.

        :param path: path to save the model to.
            Parent directories of the path must exist.
        """
        if self._history is None:
            raise RuntimeError("model not trained")

        self.model.save(path)

    @cache
    def test(self) -> float:
        """
        Test the trained model on the test dataset and obtain the test set accuracy
        in the range ``[0, 1]``.

        :return: test set accuracy.
        """
        test_size = len(self._test_ds)
        test_audio = np.zeros([test_size] + self._input_shape.as_list(),
                              dtype=np.float32)
        test_labels = np.zeros([test_size], dtype=np.int)

        for i, (audio, label) in enumerate(self._test_ds.as_numpy_iterator()):
            test_audio[i] = audio
            test_labels[i] = label

        y_pred = np.argmax(self.model.predict(test_audio), axis=1)
        y_true = test_labels

        return np.sum(y_pred == y_true) / len(y_true)
