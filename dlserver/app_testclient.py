#!/usr/bin/env python3

"""
app.py

Main entry point for the deep learning server test client.
"""


from dlserver.client.testclient import main


def main_wrapper() -> int:
    import sys

    exit(main(sys.argv))


if __name__ == "__main__":
    main_wrapper()
