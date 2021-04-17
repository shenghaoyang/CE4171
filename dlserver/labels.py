"""
labels.py

File containing mappings from inference labels to text.
"""


import enum


class Labels(enum.Enum):
    """
    Enum listing all possible speech labels.
    """

    DOWN = 0
    GO = 1
    LEFT = 2
    NO = 3
    RIGHT = 4
    STOP = 5
    UP = 6
    YES = 7

    def to_text(self) -> str:
        """
        Obtain the textual representation of this label.
        """
        return self.name.lower()
