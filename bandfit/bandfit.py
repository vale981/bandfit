import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy import signal
from skimage.transform import resize
from skimage.filters import hessian
from skimage.morphology import skeletonize


def load_data(data_file: str) -> np.ndarray:
    """
    Loads and parses the band measurement data data under ``data``,
    normalizing it so that that each row lies within the range ``0 .. 1``.
    """

    raw = np.loadtxt(data_file)

    raw -= raw.min()
    raw /= raw.max()

    return raw


def g(k: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """The off-diagonal element of the momentom space hamiltonian."""
    return a + b * np.exp(-1j * k) + c * np.exp(1j * k) + d * np.exp(-2j * k)


def hamiltonian(k: float, *args) -> np.ndarray:
    """
    The long range SSH Hamiltonian at momentum ``k`` for the parameter
    values ``a,b,c,d``. See :any:`g`.
    """

    g_value = g(np.array(k), *args)[0]
    return np.array([[0, g_value], [g_value.conjugate(), 0]])


def energy(*args) -> np.ndarray:
    """The upper band energy for momentum ``k``. See :any:`g` for the parameters."""

    return np.abs(g(*args))


def smear_band(
    a: float,
    b: float,
    c: float,
    d: float,
    resolution: tuple[float, float] = (84, 249),
    y_scaling: float = 1.1,
    x_scaling: float = 1,
    σ_x: float = 20,
    σ_y: float = 1,
    x_shift: float = 0,
    y_shift: float = 0,
) -> np.ndarray:
    """
    Get the
    See :any:`g` for the parameters.
    """

    x_res = 1000
    y_res = 100
    ks = np.linspace(-np.pi, np.pi, x_res) * x_scaling + 2 * np.pi * x_shift
    band_1d = energy(ks, a, b, c, d)

    max_energy = np.max(band_1d) * y_scaling

    band_2d = np.zeros((y_res, x_res))

    for k, energy_k in enumerate(band_1d):
        energy_bin = y_res - round((energy_k / max_energy) * (y_res - 1 / 2))
        if energy_bin >= y_res or energy_bin <= 0:
            continue

        band_2d[energy_bin, k] = 1

    band_2d = sc.ndimage.gaussian_filter(band_2d, sigma=(σ_y, σ_x))

    for i in range(band_2d.shape[0]):
        if band_2d[i].max() > 0:
            band_2d[i] /= band_2d[i].max()

    band_2d = np.concatenate([band_2d, band_2d[::-1, :]])

    return resize(band_2d, resolution)


def optimize():
    d = load_data("../data/t_t_0.5_c_2.0_d_0.0_pc_1.0_pd_1.0.txt")
    d = load_data("../data/t_t_1.0_c_0.75_d_0.2_pc_1.0_pd_1.0.txt")

    def target(x):
        a, c, dd = x
        test = smear_band(
            a, a, c, dd, x_scaling=1, y_scaling=1.04, σ_y=0.3, resolution=d.shape
        )

        return np.linalg.norm(test - d) ** 8

    return sc.optimize.minimize(
        target,
        (1, 0.75, 0.2),
        method="nelder-mead",
    )


def mean_index(data):
    size = data.size
    return round(np.sum(data * (np.arange(size) + 1)) / np.sum(data) - 1)


def detect_band(data: np.ndarray):
    data = skeletonize(hessian(data), method="lee")
    data = (data > 0).astype(float)
    k_0 = data.shape[1] // 2

    radius = 10
    e = np.argmax(data[:radius, k_0].T)
    points = [e]

    print(e)
    res = np.zeros_like(data)
    for k in range(k_0 + 1, data.shape[1]):
        min_index = max(0, e - radius)
        max_index = min(min_index + radius, data.shape[1] - 1)

        weights = []
        for next_e in range(min_index, max_index + 1):
            weights.append(get_weight(data, radius, k, next_e, 0, 3))

        print(weights)
        idx = np.argmax(weights)

        res[e + idx, k] = 2

    return res


def get_weight(data, radius, k, e, weight, depth):
    if depth <= 0 or k < 0 or k >= data.shape[1]:
        return 0

    weight += data[e, k]
    min_index = max(0, e - radius)
    max_index = min(min_index + radius, data.shape[1] - 1)

    for next_e in range(min_index, max_index + 1):
        weight += get_weight(data, radius, k + 1, next_e, weight, depth - 1)

    return weight
