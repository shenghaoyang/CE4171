"""
testclient.py

Test client for the deep learning server.
"""


import numpy as np
import grpc
import argparse
import sys
import io
from scipy.io import wavfile
from dlserver.labels import Labels
from dlserver.dlserver_pb2_grpc import DLServerStub
from dlserver.dlserver_pb2 import InferenceRequest, TrainingRequest
from collections.abc import Sequence


def main(args: Sequence[str]) -> int:
    """
    Main function for the test client.
    """
    args = argparse.ArgumentParser(
        description="Test client for the deep learning server created to"
        " complete the course project\nfor the CE4171 IoT course"
        " in NTU.\nExpects WAV audio input (max 1s, at 16000Hz)"
        " on STDIN for inference.\n"
        "Optionally, a label can be provided as the second"
        " argument to skip inferring and\nsend a retraining"
        " suggestion instead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        exit_on_error=False,
    )
    args.add_argument(
        "SERVER",
        help="name/address and port of the server to connect to,"
        " e.g. 127.0.0.1:55221",
    )
    args.add_argument(
        "LABEL",
        help="audio sample label - if provided, inference will not be done",
        nargs="?",
    )

    try:
        parsed = args.parse_args()
    except argparse.ArgumentError as e:
        print(e.message, file=sys.stderr)
        return 1

    try:
        all_data = sys.stdin.buffer.read()
        rate, data = wavfile.read(io.BytesIO(all_data))
    except Exception as e:
        print(f"error: could not read input WAV: {e}", file=sys.stderr)
        return 1

    if rate != 16000:
        print(f"error: sample rate of {rate}Hz is not 16000Hz", file=sys.stderr)
        return 1

    if len(data.shape) > 1:
        print(
            f"warning: input WAV is not mono: retaining only first channel",
            file=sys.stderr,
        )
        data = data[0]

    if data.shape[0] > 16000:
        print(f"error: input WAV is longer than 1s")
        return 1

    data = data.astype(np.float32)

    infer = parsed.LABEL is None
    if not infer:
        try:
            label = Labels[parsed.LABEL.upper()]
        except KeyError as e:
            print(f"error: label {args.LABEL} is not a valid label")
            return 1

    with grpc.insecure_channel(parsed.SERVER) as channel:
        stub = DLServerStub(channel)
        if not infer:
            req = TrainingRequest(audio_samples=data, label=label.value)
            res = stub.Train(req)
            print(
                f"Suggestion to label sample with label {label}"
                " was successfully uploaded."
            )
        else:
            req = InferenceRequest(audio_samples=data)
            res = stub.Infer(req)
            print(f"Sample was assigned label '{Labels(res.label).to_text()}'.")

    return 0
