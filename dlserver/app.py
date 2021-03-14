#!/usr/bin/env python3

"""
app.py

Main entry point for the deep learning server application.
"""


import logging
import argparse
import pathlib
import grpc
import asyncio
from typing import Sequence
from dlserver.server import DLServer
from dlserver import dlserver_pb2_grpc


logger = logging.getLogger(__name__)


async def dlserver(args: Sequence[str]):
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
    parser.add_argument("ADDRESS_PORT", help="address and port to run the server at")
    parsed = parser.parse_args(args[1:])

    model_path = pathlib.Path(parsed.SAVED_MODEL)
    logger.info(f"using saved model at {model_path}")
    logger.info(f"listening on {parsed.ADDRESS_PORT}")

    server = grpc.aio.server(maximum_concurrent_rpcs=10)
    server.add_insecure_port(parsed.ADDRESS_PORT)
    dlserver_pb2_grpc.add_DLServerServicer_to_server(DLServer(model_path), server)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    import sys

    asyncio.run(dlserver(sys.argv))
    exit(0)
