#!/usr/bin/env python3

"""
app.py

Main entry point for the deep learning server application.
"""


import logging
import argparse
import pathlib
import grpc
from typing import Sequence
from concurrent.futures import ThreadPoolExecutor
from dlserver.server import DLServer
from dlserver import dlserver_pb2_grpc


logger = logging.getLogger(__name__)


def dlserver(args: Sequence[str]):
    """
    Entry point for the DLServer application.

    :param args: program command line arguments (including conventional argv[0]).
    """
    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="""Audio recognition inference / training server written for the NTU SCSE IoT course"""
    )
    parser.add_argument("SAVED_MODEL", help="path to the saved audio recognition model")
    parsed = parser.parse_args(args[1:])

    model_path = pathlib.Path(parsed.SAVED_MODEL)
    logger.info(f"using saved model at {model_path}")

    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    dlserver_pb2_grpc.add_DLServerServicer_to_server(DLServer(model_path), server)
    server.add_insecure_port("[::]:55221")
    server.start()
    server.wait_for_termination()

    exit(0)


if __name__ == "__main__":
    import sys

    dlserver(sys.argv)
