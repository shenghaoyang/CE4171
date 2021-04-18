"""
iexecutor.py

Implementation details of the inference executor.
"""


import asyncio
import pathlib
import numpy as np
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dlserver.labels import Labels


# Global storing loaded model
model = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init(model_path: pathlib.Path):
    """
    Initialization function for the inference executors.

    Should only be called from an inference executor.

    :param model_path: path to the model that the executors should load.
    """
    global model
    from tensorflow import keras

    model = keras.models.load_model(model_path)


def infer(audio_samples: np.ndarray) -> Labels:
    """
    Run inference on the audio samples.

    Should only be called from an inference executor.

    :param audio_samples: samples to use. 1-D NumPy vector of type ``float32``.
    :return: inferred label.
    """
    global model
    import tensorflow as tf
    from dlserver.preprocess import get_spectrogram, transform_spectrogram_for_inference

    tensor = tf.constant(audio_samples, dtype=tf.float32)
    spect = transform_spectrogram_for_inference(get_spectrogram(tensor))
    # noinspection PyUnresolvedReferences
    # suppressed because init() will set global ``model`` to a loaded model.
    label = tf.argmax(model.predict(spect), axis=1)[0]

    return Labels(label)


def warmup(*args, **kwargs):
    """
    Warmup function that just sleeps to force initializer execution.

    todo: maybe this isn't the best way to do things.
    """
    time.sleep(0.1)


class Inferer:
    def __init__(self, model_path: Path, workers: int = 1, mp_context=None):
        """
        Create a new inferrer, that performs inference using a model
        by delegating inference work to created subprocesses.

        :param model_path: path to the model to use.
        :param workers: workers to start.
        :param mp_context: multiprocessing context to use.
        """
        self._model_path = model_path
        self._workers = workers
        self._mp_context = mp_context
        self._loop = asyncio.get_running_loop()

        # Executor. Not warmed up initially.
        self._inference_executor = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=self._mp_context,
            initializer=init,
            initargs=(model_path,),
        )

        self._swapping = False
        self._swap_done = asyncio.Event()
        self._swap_done.set()

        self._tasks: set[asyncio.Task] = set()

    async def _startup(self, model_path: Path) -> ProcessPoolExecutor:
        """
        Create and warm up a new executor.

        :return: warmed-up executor.
        """
        e = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=self._mp_context,
            initializer=init,
            initargs=(model_path,),
        )
        # Try and warm up executors.
        # Not guaranteed to always work.
        futs = [e.submit(warmup) for _ in range(self._workers)]
        async_futs: list[asyncio.Future] = list(map(asyncio.wrap_future, futs))
        try:
            await asyncio.wait(async_futs)
        except asyncio.CancelledError:
            for f in async_futs:
                f.cancel()
            e.shutdown(wait=False, cancel_futures=True)
            raise

        return e

    async def swap(self, new_path: Path) -> bool:
        """
        Attempt to swap to a new model.

        :param new_path: path to the saved model.
        :return: `False` if a swap could not be started, `True` if the swap started and
            was completed.
        """
        if self._swapping:
            return False

        try:
            self._swapping = True
            new_e = await self._startup(new_path)
        except asyncio.CancelledError:
            self._swapping = False
            raise

        # Block all tasks after swap has completed.
        self._swap_done.clear()
        # Wait for all pending tasks to complete and swap the executor over.
        try:
            tasks = list(self._tasks)
            if tasks:
                await asyncio.wait(tasks)
            self._inference_executor.shutdown(wait=False, cancel_futures=True)
            self._inference_executor = new_e
            self._model_path = new_path
        except asyncio.CancelledError:
            new_e.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            self._swap_done.set()
            self._swapping = False

        return True

    @property
    def model_path(self) -> Path:
        """
        Obtain the path to the model that is currently used for inference.

        Can have two elements if there are two paths in use (swap is ongoing).

        :return:
        """
        return self._model_path

    async def infer(self, samples: np.ndarray) -> Labels:
        """
        Perform inference on a given sample.

        Will block if a model swap is in progress.

        :param samples: samples to use. 1-D NumPy vector of type ``float32``.
        :return: inferred label.
        """
        if not self._swap_done.is_set():
            await self._swap_done.wait()

        async def coro() -> Labels:
            return await self._loop.run_in_executor(
                self._inference_executor, infer, samples
            )

        task = asyncio.create_task(coro())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.add(task)

        await task
        if (e := task.exception()) is not None:
            raise e

        return task.result()

    async def shutdown(self):
        """
        Shuts down this inferer.

        Safe to call if already shut down.
        """
        if self._swapping:
            await self._swap_done.wait()

        self._inference_executor.shutdown(wait=False, cancel_futures=True)
