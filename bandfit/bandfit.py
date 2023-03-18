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


    # for i in range(1, raw.shape[0]):
    #     last_row = raw[i - 1]
    #     current_row = raw[i]


    #     print(np.argmax(signal.correlate(current_row, last_row))-  current_row.size )

    #     raw[i] = np.roll(current_row, current_row.size - np.argmax(np.abs(signal.correlate(current_row, last_row))))

    return raw


