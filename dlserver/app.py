#!/usr/bin/env python3

"""
app.py

Main entry point for the deep learning server application.
"""


import logging
import argparse
import grpc
import enum
import signal
import asyncio
import dataclasses
import configparser
from functools import partial
from typing import Sequence
from dlserver.server.server import DLServer, Config as DLServerConfig
from dlserver import dlserver_pb2_grpc


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AppConfig:
    """
    Global application configuration.
    """

    # RPC server endpoint.
    endpoint: str
    # Maximum number of concurrent RPCs.
    max_concurrent_rpcs: int
    # DLServer configuration.
    dlserver: DLServerConfig

    @classmethod
    def from_config(cls, config: configparser.ConfigParser) -> "AppConfig":
        """
        Load configuration from a parsed configuration.

        :param config: parsed configuration.
        :return: loaded configuration.
        """
        out = {
            "endpoint": config["RPCServer"]["Endpoint"],
            "max_concurrent_rpcs": config["RPCServer"].getint("MaxConcurrentRPCs"),
            "dlserver": DLServerConfig.from_config(config),
        }

        return cls(**out)


async def dlserver(args: Sequence[str]) -> int:
    """
    Entry point for the DLServer application.

    :param args: program command line arguments (including conventional argv[0]).
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="""Audio recognition inference / training server written for the NTU SCSE IoT course"""
    )

    def read_config(path: str) -> AppConfig:
        p = configparser.ConfigParser()
        p.read(path)

        return AppConfig.from_config(p)

    parser.add_argument(
        "CONFIG", help="path to the configuration .ini file", type=read_config
    )
    parsed = parser.parse_args(args[1:])
    appconfig = parsed.CONFIG

    logger.info(f"running with configuration:\n{appconfig}")

    stopping = False
    server = grpc.aio.server()
    server.add_insecure_port(appconfig.endpoint)
    dlserver_pb2_grpc.add_DLServerServicer_to_server(
        DLServer(appconfig.dlserver), server
    )

    def sig_handler(sig: enum.Enum, loop: asyncio.BaseEventLoop):
        nonlocal stopping
        if stopping:
            return

        logger.info(f"got {sig}, terminating")
        loop.create_task(server.stop(0))
        stopping = True

    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame)
        loop.add_signal_handler(sig, partial(sig_handler, signame, loop))

    await server.start()
    await server.wait_for_termination()

    return 0


def main() -> int:
    import sys

    exit(asyncio.run(dlserver(sys.argv)))


if __name__ == "__main__":
    main()
