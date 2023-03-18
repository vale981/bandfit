import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy import signal


def load_data(data_file: str) -> np.ndarray:
    """
    Loads and parses the band measurement data data under ``data``,
    normalizing it so that that each row lies within the range ``0 .. 1``.
    """

    raw = np.loadtxt(data_file)

    raw -= raw.min()
    raw /= raw.max()

    return raw
