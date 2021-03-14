"""
server.py

Implementation of the deep learning server.
"""


import logging
import pathlib
import grpc
import wavio
import tensorflow as tf
from tensorflow import keras
from train.preprocess import get_spectrogram, transform_spectrogram_for_inference
from dlserver import dlserver_pb2_grpc
from dlserver.dlserver_pb2 import InferenceResponse, InferenceRequest


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DLServer(dlserver_pb2_grpc.DLServerServicer):
    """
    Servicer for the DLServer gRPC service.
    """

    def __init__(self, model_path: pathlib.Path):
        """
        Create a new servicer.

        :param model_path: path to the saved audio recognition model.
        """
        super().__init__()
        self._model = keras.models.load_model(model_path)

    async def Infer(
        self, request: InferenceRequest, context: grpc.ServicerContext
    ) -> InferenceResponse:
        """
        Perform inference on the submitted audio data.

        :param request: inference request.
        :param context: RPC context.
        :return: inference response.
        """
        if not len(request.audio_samples):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no audio samples provided")

        if len(request.audio_samples) > 16000:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "too many audio samples provided"
            )

        tensor = tf.constant(request.audio_samples, dtype=tf.float32)
        spect = transform_spectrogram_for_inference(get_spectrogram(tensor))
        label = tf.argmax(self._model.predict(spect), axis=1)[0]

        return InferenceResponse(label=label)
