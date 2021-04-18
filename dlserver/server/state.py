"""
state.py

Routines for loading / storing server state.
"""


import sqlite3
from pathlib import Path


class PersistentState:
    """
    Class storing persistent server state.

    State storage is backed by an external SQLite database.
    """

    def __init__(
        self, path: Path, initial_model_path: Path, force_initialize: bool = False
    ):
        """
        Initialize server state from a database path.

        :param path: path to the database file containing server state.
        :param initial_model_path: path to the model to use if no saved
            state can be found.
        :param force_initialize: whether to force initialization of the
            state storage database.
        """
        self._db_path = path
        self._initial_model_path = initial_model_path

        if force_initialize:
            self._db_path.unlink(missing_ok=True)

        initialize = not self._db_path.exists()
        self._db_conn = sqlite3.connect(path)
        if initialize:
            self._initialize()

    def _initialize(self):
        """
        Initialize the database.
        """
        with self._db_conn:
            self._db_conn.executescript(
                r"""PRAGMA encoding = utf8;
                    CREATE TABLE state(
                        id INTEGER PRIMARY KEY,
                        untrained_samples INTEGER,
                        model_path TEXT);"""
            )
            self._db_conn.execute(
                r"INSERT INTO state VALUES(0, 0, ?)", (str(self._initial_model_path),)
            )

    @property
    def model_path(self) -> Path:
        """
        Obtains the saved model path.

        This may be the path to the most recently retrained model, or the
        initial model in case no models were trained by the server.

        :return: path to model to use.
        """
        with self._db_conn:
            cur = self._db_conn.execute(
                r"""SELECT model_path
                    FROM state
                    WHERE id = 0"""
            )
            return Path(cur.fetchone()[0])

    @property
    def untrained_samples(self) -> int:
        """
        Obtain the number of uploaded samples that have yet to be trained
        on.

        :return: number of untrained samples.
        """
        with self._db_conn:
            cur = self._db_conn.execute(
                r"""SELECT untrained_samples
                    FROM state
                    WHERE id = 0"""
            )
            return cur.fetchone()[0]

    def update(self, untrained_samples: int, model_path: Path):
        """
        Update the persistent state.

        :param untrained_samples: number of samples yet to be trained on.
        :param model_path: path to model to be used for inference.
        """
        with self._db_conn:
            self._db_conn.execute(
                r"""UPDATE state
                    SET untrained_samples = ?, model_path = ?
                    WHERE id = 0""",
                (untrained_samples, str(model_path)),
            )

    def increment_untrained_samples(self, by: int) -> int:
        """
        Increment the number of untrained samples.

        :param by: amount to increment by (can be negative).
        :return: new number of untrained samples.
        """
        with self._db_conn:
            cur = self._db_conn.execute(
                r"""UPDATE state
                    SET untrained_samples = untrained_samples + ?
                    WHERE id = 0
                    RETURNING untrained_samples""",
                (by,),
            )

            return cur.fetchone()[0]
