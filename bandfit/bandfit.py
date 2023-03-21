import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy import signal
from skimage.transform import resize
from skimage.filters import hessian
from skimage.morphology import skeletonize
import itertools
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


def energy(*args) -> np.ndarray:
    """The upper band energy for momentum ``k``. See :any:`g` for the parameters."""

    return np.abs(g(*args))


def lorentzian(x, x0, γ):
    return γ**2 / ((x - x0) ** 2 + γ**2)


def double_lorentzian(x, x0_1, x0_2, γ, r=1):
    return lorentzian(x, x0_1, γ) + r * lorentzian(x, x0_2, γ)


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

    σ_1, σ_2, _, _ = np.sqrt(np.diag(cov))

    (e_1, σ_1), (e_2, σ_2) = np.sort(((e_1, σ_1), (e_2, σ_2)), axis=0)

    # es = np.linspace(0, col.size, 1000)
    # plt.plot(col)
    # plt.plot(es, double_lorentzian(es, e_1, e_2, γ, _))

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


def candidate(k, c, d, a, δb, k_scale, k_shift):
    k = np.asarray(k[: k.size // 2]) * k_scale + k_shift
    energies = energy(k, a, a + δb * a, c, d)

    return np.hstack([energies, energies])


def fit_to_bands(
    bands,
    bounds=[(-10, -10, 0.5, -0.5, 0.8, -0.5), (10, 10, 10, 0.5, 1.2, 0.5)],
    ic_scan_steps=5,
    c_d_order=0,
    debug_plots=False,
):
    bands_normalized = bands.copy()

    bands_normalized[:, :2] -= np.sum(bands_normalized[:, :2], axis=1).mean() / 2
    bands_normalized[:, :2] /= np.max(np.abs(bands_normalized[:, :2]), axis=0)
    bands_normalized[:, 0] *= -1

    ks = np.linspace(-np.pi, np.pi, bands_normalized.shape[0])

    if debug_plots:
        plt.plot(ks, bands_normalized[:, 0])
        plt.plot(ks, bands_normalized[:, 1])

    bounds = np.array(bounds)
    Δ_bounds = bounds[1, :2] - bounds[0, :2]

    ics = np.tile(np.linspace(0, 1, ic_scan_steps), (2, 1))
    ics *= Δ_bounds[:, None]
    ics += bounds[0, :2][:, None]

    min_δb = np.inf
    (c, d, a, δb, k_scale, k_shift) = np.zeros(6)
    cov = np.zeros(6)

    σs = []
    for ic in itertools.product(*ics):
        p, current_cov, _, _, success = sc.optimize.curve_fit(
            candidate,
            np.hstack([ks, ks]),
            np.hstack([bands_normalized[:, 0], bands_normalized[:, 1]]),
            (*ic, 1, 0, 1, 0),
            sigma=np.hstack([bands_normalized[:, 2], bands_normalized[:, 3]]),
            bounds=bounds,
            full_output=True,
        )

        if success < 1 or success > 4:
            continue

        if c_d_order == 1 and p[0] > p[1]:
            continue

        if c_d_order == -1 and p[0] < p[1]:
            continue

        σ_δb_current = np.sqrt(current_cov[3, 3])
        σ_rel = np.sqrt(np.sum(np.diag(current_cov))) / np.linalg.norm(p)
        if abs(p[3]) + σ_δb_current < min_δb and (
            len(σs) == 0 or σ_rel <= np.min(σs) * 2
        ):
            (c, d, a, δb, k_scale, k_shift) = p
            min_δb = abs(δb) + σ_δb_current
            cov = current_cov

        σs.append(σ_rel)

    if debug_plots:
        plt.plot(ks, candidate(np.hstack([ks, ks]), *p)[: bands.shape[0]])

    b = a + δb * a

    σ_c, σ_d, σ_a, σ_δb, σ_k_scale, σ_k_shift = np.sqrt(np.diag(cov))

    σ_b = np.sqrt((σ_a * (1 + δb)) ** 2 + (a * σ_δb) ** 2)
    σ = np.array((σ_a, σ_b, σ_c, σ_d, σ_k_scale, σ_k_shift))

    scale = 1 / a

    a *= scale
    b *= scale
    c *= scale
    d *= scale

    σ[:4] *= scale

    return ((a, b, c, d, k_scale, k_shift), σ)


def plot_data_with_bands_and_fit(data, bands, band_fit):
    plt.matshow(data)
    ks = np.arange(data.shape[1])

    (a, b, c, d, k_scale, k_shift), σ, scales, shifts = band_fit

    smooth_ks_unscaled = np.linspace(0, data.shape[1], 1000)
    smooth_ks = smooth_ks_unscaled / smooth_ks_unscaled[-1]
    smooth_ks -= 1 / 2
    smooth_ks *= 2 * np.pi
    smooth_ks *= k_scale
    smooth_ks += k_shift

    upper_band = -energy(smooth_ks, a, b, c, d)
    print(bands[:, 0][data.shape[1] // 2])
    upper_band += energy(k_shift, a, b, c, d) + shifts
    upper_band *= scales[0]

    plt.errorbar(ks, bands[:, 0], yerr=bands[:, 2])
    plt.errorbar(ks, bands[:, 1], yerr=bands[:, 3])
    plt.plot(smooth_ks_unscaled, upper_band)


def plot_error_funnel(p, σ):
    ks = np.linspace(-np.pi, np.pi, 1000)

    params = np.random.multivariate_normal(p, np.diag(σ), 1000)
    for param in params:
        plt.plot(ks, energy(ks, *param), color="gray", alpha=0.1)

    plt.plot(ks, energy(ks, *p), linewidth=2)
