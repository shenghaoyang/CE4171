"""
server.py

Implementation of the deep learning server.
"""
import dataclasses
import logging
import pathlib
import grpc
import uuid
import asyncio
import multiprocessing
import tensorflow as tf
import numpy as np
import configparser
from scipy.io import wavfile
from collections.abc import Sequence
from dlserver import iexecutor
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
    # Server state file. If the state file is absent it is
    # recreated.
    state_file: pathlib.Path
    # Path where samples uploaded for inference will be stored.
    infer_upload_path: pathlib.Path
    # Path where samples uploaded for training will be stored.
    training_upload_path: pathlib.Path
    # Path where newly trained models will be stored.
    new_models_path: pathlib.Path
    # Path to the initial model to use (if no newly trained models are available).
    initial_model_path: pathlib.Path

    @classmethod
    def form_file(cls, path: pathlib.Path) -> 'Config':
        """
        Load configuration from a configuration ``.ini`` file.

        Only the ``[Server]`` section will be read.

        :param path: path to the configuration file.
        :return: loaded configuration.
        """
        config = configparser.ConfigParser()
        config.read(path)

        sc = config['Server']
        out = {
            'state_file': sc['StateFile'],
            'infer_upload_path': sc['InferUploadPath'],
            'training_upload_path': sc['TrainingUploadPath'],
            'new_models_path': sc['NewModelsPath'],
            'initial_model_path': sc['ModelPath']
        }
        out_paths = dict((k, pathlib.Path(v)) for k, v in out.items())

        return cls(**out_paths)


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

        self._loop = asyncio.get_running_loop()
        self._context = multiprocessing.get_context("forkserver")
        self._inferer = iexecutor.Inferer(model_path=self._config.initial_model_path, mp_context=self._context)

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
        label = await self._inferer.infer(np.array(request.audio_samples, dtype=np.float32))

        logger.info(f"inferred {peer}'s sample as having label {label}")

        wavfile.write(self._config.infer_upload_path.joinpath(f"{uid}-{label.to_text()}.wav"), 16000, samples)

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

        logger.info(f"training on sample from {peer}")

        tensor = tf.constant(request.audio_samples)

        return Empty()
