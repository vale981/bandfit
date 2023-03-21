import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy import signal
from skimage.transform import resize
from skimage.filters import hessian
from skimage.morphology import skeletonize
from itertools import chain
from scipy.interpolate import splrep, BSpline


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

    plt.plot()
    return

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


def lorentzian(x, x0, γ):
    return γ**2 / ((x - x0) ** 2 + γ**2)


def double_lorentzian(x, x0_1, x0_2, γ, r=1):
    return lorentzian(x, x0_1, γ) + r * lorentzian(x, x0_2, γ)


def refine_band_fit(k, e0, data):
    pass


def detect_bands_fixed_k(
    k, data, γ, last_separation=0, min_height=0.5, separation_continuity=1 / 2
):
    col = data[:, k].copy()
    col -= col.min()
    col /= col.max()

    e_axis = np.arange(col.size)

    guesses, props = sc.signal.find_peaks(col, distance=2, height=min_height)
    means = guesses[:, None] + guesses[None, :]

    guess_i_1, guess_i_2 = np.unravel_index(
        np.argmin(abs(means - col.size)), means.shape
    )

    guess_1, guess_2 = np.sort((guesses[guess_i_1], guesses[guess_i_2]))

    if (
        last_separation > 0
        and abs((abs(guess_2 - guess_1) - last_separation)) / last_separation
        > separation_continuity
    ):
        guess_idx = (guesses[:, None] == [guess_1, guess_2]).argmax(axis=0)
        heights = props["peak_heights"][guess_idx]

        if heights[0] > heights[1]:
            guess_2 = col.size - guess_1

        else:
            guess_1 = col.size - guess_2

    col /= col[guess_1]

    (e_1, e_2, γ, _), cov = sc.optimize.curve_fit(
        double_lorentzian,
        e_axis,
        col,
        (guess_1, guess_2, γ, 1),
        bounds=(
            (max(guess_1 - γ, 0), max(guess_2 - γ, 0), 0.5, 0.3),
            (min(guess_1 + γ, col.size), min(guess_2 + γ, col.size), col.size, 1 / 0.3),
        ),
    )

    e_1, e_2 = np.sort((e_1, e_2))
    # es = np.linspace(0, col.size, 1000)
    # plt.plot(col)
    # plt.plot(es, double_lorentzian(es, e_1, e_2, γ, _))

    σ_1, σ_2, _, _ = np.sqrt(np.diag(cov))

    return e_1, e_2, σ_1, σ_2


def detect_bands(data, γ=20, min_height=0.5):
    bands = []

    e_1, e_2 = 0, 0
    for k in range(data.shape[1]):
        e_1, e_2, *σ = detect_bands_fixed_k(
            k, data, γ, last_separation=abs(e_2 - e_1), min_height=min_height
        )

        bands.append((e_1, e_2, *σ))

    return np.array(bands)


def plot_data_with_bands(data, bands):
    plt.matshow(data)
    ks = np.arange(data.shape[1])

    plt.errorbar(ks, bands[:, 0], yerr=bands[:, 2])
    plt.errorbar(ks, bands[:, 1], yerr=bands[:, 3])


#    return sc.optimize.curve_fit(double_lorentzian, e_axis, col, (0, 10, 0, 3))


def candidate(k, b, c, d, k_scale, k_shift):
    k = np.asarray(k[: k.size // 2]) * k_scale + k_shift
    energies = energy(k, 1 - b - c - d, b, c, d)
    energies /= energies.max()

    return np.hstack([energies, energies])


def fit_to_bands(bands, b=1, c=1, d=1):
    bands_normalized = bands.copy()

    bands_normalized[:, :2] -= np.sum(bands_normalized[:, :2], axis=1).mean() / 2
    bands_normalized[:, :2] /= np.max(np.abs(bands_normalized[:, :2]), axis=0)
    bands_normalized[:, 0] *= -1

    ks = np.linspace(-np.pi, np.pi, bands_normalized.shape[0])

    plt.plot(ks, bands_normalized[:, 0])
    plt.plot(ks, bands_normalized[:, 1])
    p, cov = sc.optimize.curve_fit(
        candidate,
        np.hstack([ks, ks]),
        np.hstack([bands_normalized[:, 0], bands_normalized[:, 1]]),
        (b, c, d, 1, 0),
        sigma=np.hstack([bands_normalized[:, 2], bands_normalized[:, 3]]),
        bounds=[(0, -10, -10, 0.9, -0.5), (10, 10, 10, 1.1, 0.5)],
    )

    plt.plot(ks, candidate(np.hstack([ks, ks]), *p)[: bands.shape[0]])
    (b, c, d, k_scale, k_shift) = p
    a = 1 - b - c - d

    scale = 1 / a

    a *= scale
    b *= scale
    c *= scale
    d *= scale

    σ = np.sqrt(np.diag(cov))
    σ = np.array([np.sqrt(sum(σ[:3] ** 2)), *σ])

    σ[:4] *= scale

    return ((a, b, c, d, k_scale, k_shift), σ)
