"""
iexecutor.py

Implementation of a training executor that performs model training in
a subprocess.
"""

import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections.abc import Sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(data_paths: Sequence[Path], batch_size: int, epochs: int, save_to: Path) -> float:
    """
    Run actual training.

    Should only be called in the training subprocess.

    :param data_paths: See the documentation for ``dlserver.training.trainer.Trainer``.
    :param batch_size: See the documentation for ``dlserver.training.trainer.Trainer``.
    :param epochs: See the documentation for ``dlserver.training.trainer.Trainer``.
    :param save_to: Path to save the trained model to.

    :return: accuracy of the model evaluated over the test dataset in the range ``[0, 1]``.
    """
    from dlserver.training.trainer import Trainer as TTrainer
    trainer = TTrainer(data_paths, batch_size, epochs)
    trainer.train()
    trainer.save(save_to)

    return trainer.test()


async def subprocess_train(save_to: Path,
                           data_paths: Sequence[Path], batch_size: int = 16, epochs: int = 20, mp_context=None) \
        -> float:
    """
    Train the voice recognition model using a subprocess.

    :param save_to: Path to save the trained model to.
    :param data_paths: See the documentation for ``dlserver.training.trainer.Trainer``.
    :param batch_size: See the documentation for ``dlserver.training.trainer.Trainer``.
    :param epochs: See the documentation for ``dlserver.training.trainer.Trainer``.
    :param mp_context: multiprocessing context to use.
    :return: accuracy of the model evaluated over the test dataset in the range ``[0, 1]`.
    """
    loop = asyncio.get_running_loop()
    training_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)

    try:
        return await loop.run_in_executor(training_executor, train,
                                          data_paths, batch_size, epochs, save_to)
    finally:
        training_executor.shutdown(wait=False, cancel_futures=True)
