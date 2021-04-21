"""
server.py

Implementation of the deep learning server.
"""
import time
import dataclasses
import logging
import grpc
import uuid
import asyncio
import multiprocessing
import numpy as np
import configparser
from scipy.io import wavfile
from pathlib import Path
from collections.abc import Sequence
from dlserver.labels import Labels
from dlserver.executors import iexecutor
from dlserver.executors import texecutor
from dlserver.server.state import PersistentState
from dlserver import dlserver_pb2_grpc
from dlserver.dlserver_pb2 import InferenceResponse, InferenceRequest, TrainingRequest
from dlserver.dlserver_pb2 import google_dot_protobuf_dot_empty__pb2


Empty = google_dot_protobuf_dot_empty__pb2.Empty

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for the deep learning server.
    """

    # Number of worker processes to create for inference.
    iworkers: int
    # Server state file. If the state file is absent it is
    # recreated.
    state_file: Path
    # Path where samples uploaded for inference will be stored.
    infer_upload_path: Path
    # Path where samples uploaded for training will be stored.
    training_upload_path: Path
    # Path to the samples used for initial model training.
    base_samples_path: Path
    # Path where newly trained models will be stored.
    new_models_path: Path
    # Path to the initial model to use (if no newly trained models are available).
    initial_model_path: Path
    # Number of training samples to collect before starting a training session.
    samples_before_train: int

    @classmethod
    def from_config(cls, config: configparser.ConfigParser) -> "Config":
        """
        Load configuration from a parsed configuration.

        Only the ``[DLServer]`` section will be read.

        :param config: parsed configuration.
        :return: loaded configuration.
        """

        sc = config["DLServer"]
        out = {
            "iworkers": sc.getint("IWorkers"),
            "state_file": Path(sc["StateFile"]),
            "infer_upload_path": Path(sc["InferUploadPath"]),
            "training_upload_path": Path(sc["TrainingUploadPath"]),
            "base_samples_path": Path(sc["BaseSamplesPath"]),
            "new_models_path": Path(sc["NewModelsPath"]),
            "initial_model_path": Path(sc["ModelPath"]),
            "samples_before_train": sc.getint("SamplesBeforeTrain"),
        }

        return cls(**out)


class DLServer(dlserver_pb2_grpc.DLServerServicer):
    """
    Servicer for the DLServer gRPC service.
    """

    def __init__(self, config: Config):
        """
        Create a new servicer.

        :param config: servicer configuration.
        """
        super().__init__()

        self._config = config
        self._persistent_state = PersistentState(
            path=self._config.state_file,
            initial_model_path=self._config.initial_model_path,
        )

        self._loop = asyncio.get_running_loop()
        self._context = multiprocessing.get_context("forkserver")
        self._inferer = iexecutor.Inferer(
            workers=self._config.iworkers,
            model_path=self._persistent_state.model_path,
            mp_context=self._context,
        )
        self._train_swap_task = None

    @staticmethod
    def _check_audio(samples: Sequence[float], context: grpc.ServicerContext):
        """
        Check provided audio samples for validity.

        Returns gRPC errors (using ``context``) if not.

        :param samples: audio samples.
        :param context: gRPC context.
        """
        if not len(samples):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no audio samples provided")

        if len(samples) > 16000:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "too many audio samples provided"
            )

    @property
    def _training_and_swapping(self) -> bool:
        """
        Checks if the server is in the process of training a new model and swapping to it.

        :return: ``True`` if it is, ``False`` otherwise.
        """
        return self._train_swap_task is not None

    def _train_and_swap(self):
        """
        Train the model on newly-uploaded training data and swap over to using the new
        model asynchronously.

        :return: ``True`` if the train and swap was successfully started, ``False`` if another
            train and swap operation is already ongoing.
        """
        if self._training_and_swapping:
            return False

        save_to = self._config.new_models_path.joinpath(str(uuid.uuid4()))

        async def train_and_swap_impl():
            logger.info(f"starting training of new model, saving at {save_to}")
            accuracy = await texecutor.subprocess_train(
                save_to,
                (self._config.base_samples_path, self._config.training_upload_path),
            )
            logger.info(
                f"completed training of new model at {save_to}, accuracy: {accuracy}"
            )

            logger.info(f"swapping to new model at {save_to}")
            await self._inferer.swap(save_to)
            logger.info(f"completed swapping to new model")

        # This calculation may be off because there could be samples added partway during the
        # training process.
        # But that's okay because it just leads to a tiny bit of inefficiency.
        untrained = self._persistent_state.untrained_samples

        def done(task: asyncio.Task):
            self._train_swap_task = None
            if (e := task.exception()) is not None:
                logger.warning(f"failed to train and swap to a new model: {e}")
                return

            logger.info(
                f"updating persistent state, trained on {untrained} additional samples,"
                f" new model path is at {save_to}"
            )
            self._persistent_state.update(
                self._persistent_state.untrained_samples - untrained, save_to
            )

        self._train_swap_task = asyncio.create_task(train_and_swap_impl())
        self._train_swap_task.add_done_callback(done)

        return True

    async def Infer(
        self, request: InferenceRequest, context: grpc.ServicerContext
    ) -> InferenceResponse:
        """
        Perform inference on the submitted audio data.

        :param request: inference request.
        :param context: RPC context.
        :return: inference response.
        """
        self._check_audio(request.audio_samples, context)
        peer = context.peer()
        uid = uuid.uuid4()

        logger.info(f"inferring sample from {peer}: UUID {uid}")

        samples = np.array(request.audio_samples, dtype=np.float32)

        start = time.monotonic_ns()
        label = await self._inferer.infer(
            np.array(request.audio_samples, dtype=np.float32)
        )
        elapsed_ns = time.monotonic_ns() - start

        logger.info(
            f"inferred {peer}'s sample in {elapsed_ns}ns as having label {label}"
        )
        wavfile.write(
            self._config.infer_upload_path.joinpath(f"{uid}-{label.to_text()}.wav"),
            16000,
            samples,
        )

        return InferenceResponse(label=label.value)

    async def Train(
        self, request: TrainingRequest, context: grpc.ServicerContext
    ) -> Empty:
        """
        Re-train the model on the submitted audio data.

        :param request: training request.
        :param context: RPC context.
        :return: empty response.
        """
        self._check_audio(request.audio_samples, context)
        peer = context.peer()
        uid = uuid.uuid4()
        label = Labels(request.label)

        logger.info(f"saving training sample from {peer}: UUID {uid}, label {label}")
        samples = np.array(request.audio_samples, dtype=np.float32)

        destination_directory = self._config.training_upload_path.joinpath(
            label.to_text()
        )
        destination_directory.mkdir(mode=0o770, parents=True, exist_ok=True)

        wavfile.write(
            destination_directory.joinpath(f"{uid}.wav"),
            16000,
            (samples * 32767.0).astype(np.int16),
        )

        untrained_samples = self._persistent_state.increment_untrained_samples(1)
        logger.info(f"server has collected {untrained_samples} untrained samples")

        if untrained_samples >= self._config.samples_before_train:
            logger.info(
                f"collected a sufficient amount of uploaded training samples: starting training"
            )
            if self._train_and_swap():
                logger.info(f"started training and swapping task")
            else:
                logger.info(f"training and swapping ongoing, skipping")

        return Empty()
