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
    """A Lorentzian centered around ``x0`` with width ``γ``."""
    return γ**2 / ((x - x0) ** 2 + γ**2)


def double_lorentzian(x, x0_1, x0_2, γ, a, b):
    """
    A sum of two lorentzians of equal width cerntered around
    ``x0_1,x0_2`` with width ``γ`` peak heights ``a, b``
    """
    return a * lorentzian(x, x0_1, γ) + b * lorentzian(x, x0_2, γ)


def detect_bands_fixed_k(
    k,
    data,
    γ=10,
    last_separation=0,
    separation_continuity=1 / 2,
    min_height=0.5,
    **kwargs
):
    """
    Detect the location of the peaks corresponding to the bands in a
    vertical slice throught the measured bands structure (i.e. fixed
    ``k``).

    The coarse detection of the peaks is handled by
    ``scipy.signal.find_peaks`` to which the ``**kwargs`` are passed
    through.  Of those peaks, the ones being most symmetric around the
    center of the slice are being selected.  Should the separation of
    the peaks not fullfil the continuity condition (see
    ``separation_continuity``), the peak with the highest amplitude is
    being selected and mirrored across the center.  Subsequently a sum
    of two Lorentzians with peaks within ``2 γ`` of those peaks is
    fitted to the slice.  This fit also quantifies the uncertainty of
    the peak positions.

    :param k: The index of the column in the ``data`` where the bands
              are to be detected.
    :param data: An two dimensional array of shape ``(# energy slices,
        # k states)`` containing the measured band structure.
    :param γ: The bounds (in pixels) of the fine-tuning of the peak
              detection around the coarse-detection value.  Usually a
              value around ``10`` is a good idea.  If the fit gets too
              jittery try increasing this value.  Too large a value
              will make it possible for the fit to miss the bands
              entirely however.
    :param last_separation: The separation of the detected bands at an
        adjacent ``k`` slice.  This parameter is used to enforce some
        continuity on the detected band structure.
    :param separation_continuity: The separation between the detected
        peaks and ``last_separation`` is enforced to be within
        ``separation_continuity * last_separation`` and to be greater
        than ``separation_continuity * last_separation``.
    :param min_height: The ``height`` parameter for the
        ``scipy.signal.find_peaks``.

    :returns: A tuple ``(e_1, e_2, σ_1, σ_2)`` with the peak positions
              ``e_1 < e_2`` and their uncertainties ``σ_1, σ_2``.
    """

    col = data[:, k].copy()
    col -= col.min()
    col /= col.max()

    e_axis = np.arange(col.size)

    guesses, props = sc.signal.find_peaks(
        col,
        distance=max(2, last_separation * separation_continuity),
        height=min_height,
        **kwargs
    )
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

    (e_1, e_2, γ, *_), cov = sc.optimize.curve_fit(
        double_lorentzian,
        e_axis,
        col,
        (guess_1, guess_2, γ, 1, 1),
        bounds=(
            (max(guess_1 - γ, 0), max(guess_2 - γ, 0), 0.5, 0.1, 0.1),
            (
                min(guess_1 + γ, col.size),
                min(guess_2 + γ, col.size),
                col.size,
                2,
                2,
            ),
        ),
    )

    σ_1, σ_2, *_ = np.sqrt(np.diag(cov))
    (e_1, σ_1), (e_2, σ_2) = np.sort(((e_1, σ_1), (e_2, σ_2)), axis=0)

    return e_1, e_2, σ_1, σ_2


def detect_bands(data, *args, **kwargs):
    """
    Returns the bands detected in the measured band structure
    ``data``.

    The rest of the arguments are passed on to
    :any:`detect_bands_fixed_k`.

    :returns: An array of shape ``(#k, 4)``.  Each row contains
              ``(e_1, e_2, σ_1, σ_2)`` with the peak positions ``e_1 <
              e_2`` and their uncertainties ``σ_1, σ_2``.
    """
    bands = []

    if "last_separation" in kwargs:
        del kwargs["last_separation"]

    e_1, e_2 = 0, 0
    for k in range(data.shape[1]):
        e_1, e_2, *σ = detect_bands_fixed_k(
            k, data, *args, **kwargs, last_separation=abs(e_2 - e_1)
        )

        bands.append((e_1, e_2, *σ))

    return np.array(bands)


def plot_data_with_bands(data, bands):
    """
    Plot the measured band structure ``data`` together with the output
    of :any:`detect_bands`.
    """

    plt.matshow(data)
    ks = np.arange(data.shape[1])

    plt.errorbar(ks, bands[:, 0], yerr=bands[:, 2], color="white")
    plt.errorbar(ks, bands[:, 1], yerr=bands[:, 3], color="white")


def candidate(k, c, d, a, δb, k_scale, k_shift):
    """
    Returns the theoretical band structure as a function of ``k``
    doubled so that upper and lower band can be fitted simultaneously.

    The ``a,c,d`` parameters correspond to the parameters of the
    hamiltonian and ``b = a * (1 + δb)`` so that the difference
    between ``a`` and ``b`` can be constrained.  Additionally, the
    ``k`` argument is scaled and shifted with ``k_scale, k_shift``.
    """
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
    """
    Fits the bands obtained from :any:`detect_bands` to the
    theoretical band structure to obtain the parameters ``a, b, c,
    d``.

    The fit is weighted by the uncertainties on the detected band
    structure and preformed on the upper and lower band
    simultaneously.  Multiple initial conditions (controlled by
    ``ic_scan_steps``) are tested and the result is selected for
    having non-outlying variance and yields values of ``a`` and ``b``
    that are close to each other.

    Because at ``a==b`` swapping ``c, d`` results in the same band
    structure the parameter ``c_d_order`` controlls whether ``c < d``
    or ``d > c`` is preferred when fitting.

    :param bands: The output of :any:`detect_bands`.
    :param bounds: A list of tuples specifying the lower and upper
        bounds on the fit parameters (see the documentation of
        ``scipy.optimize.curve_fit``).  Note that the fit parameters
        are the parameters of :any:`candidate` which dictates their
        order.
    :param ic_scan_steps: The number of configurations per variable
        that are scanned when fitting.  To find all possible
        parametere configuration the initial points for fitting are
        evenly distributed through the ``c, d`` space set by the
        ``bounds``.
    :param debug_plots: Plot the two bands and the fitted band for
        debugging.

    :returns: A tuple containing the parameters ``a, b, c, d, k_scale,
              k_shift`` and a tuple containing the uncertainty of
              these parameters (as estimated by the fit routine).  The
              paramteters are normalized so that ``a==1``.
    """

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

    final_params = (a, b, c, d, k_scale, k_shift)
    σ[:4] = np.sqrt(
        (σ[:4] * scale) ** 2 + (np.array(final_params[:4])[:4] / a**2 * σ[0]) ** 2
    )

    return (final_params, σ)


def plot_error_funnel(p, σ, ks=None):
    """
    Plot the band structure given the paramters ``p = (a, b, c, d)``
    and their uncertainties ``σ`` for monomentum vales ``ks``.
    """
    ks = ks or np.linspace(-np.pi, np.pi, 1000)

    params = np.random.multivariate_normal(p, np.diag(σ), 2000)
    energies = []
    for param in params:
        energies.append(energy(ks, *param))
        # plt.plot(ks, energy(ks, *param), color="gray", alpha=0.1)

    energies = np.array(energies)
    σ_e = np.std(energies, axis=0)
    mean = energy(ks, *p)
    plt.plot(ks, mean, linewidth=2)
    plt.fill_between(ks, mean - σ_e, mean + σ_e, alpha=0.2)

    return mean, mean - σ_e, mean + σ_e
