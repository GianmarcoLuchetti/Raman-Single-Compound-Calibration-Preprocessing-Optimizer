from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import itertools
from scipy.signal import find_peaks
from preprocessing import Baseline_removal, Noise_filter, Scattering_correction
from scipy.stats import gmean


class MultipleMatchesError(RuntimeError):
    """Raised when a filename search returns more than one match.

    The error message states the number of matches and the searched name,
    then lists each matching path on its own line.

    Parameters
    ----------
    filename : str
        Name or pattern that was searched for.
    matches : Iterable[Path]
        Paths that matched `filename`. Must support ``len()``.

    Attributes
    ----------
    matches : list of Path
        The matching paths, in the order given.
    """

    def __init__(self, filename: str, matches: Iterable[Path]) -> None:
        msg = f"{len(matches)} matches found for {filename!r}\n" + "\n".join(map(str, matches))
        super().__init__(msg)
        self.matches = list(matches)


def get_data_path(filename: str, allow_multiple: bool = False) -> Path | list[Path]:
    """Locate a file by searching the working directory and its ancestors.

    Starting from the current working directory, recursively searches each
    directory and then each of its parents in turn, up to the filesystem
    root, for paths matching `filename`.

    Parameters
    ----------
    filename : str
        File name or glob pattern passed to `Path.rglob`.
    allow_multiple : bool, default=False
        If False, stop at the first directory level with any match and
        require that match to be unique. If True, search every level up
        to the root and collect all matches.

    Returns
    -------
    Path or list of Path
        If `allow_multiple` is False, the single resolved matching path.
        If True, a list of resolved paths gathered from all levels, in
        search order.

    Raises
    ------
    MultipleMatchesError
        If `allow_multiple` is False and the first level with any match
        contains more than one match.
    FileNotFoundError
        If no match is found at any level.
    """
    start = Path.cwd().resolve()  # Start from current working directory
    all_matches: List[Path] = []

    # Traverse upward through parent directories
    for ancestor in (start, *start.parents):
        matches = list(ancestor.rglob(filename))
        if matches:
            if not allow_multiple:
                if len(matches) == 1:
                    return matches[0].resolve()
                raise MultipleMatchesError(filename, matches)
            all_matches.extend(m.resolve() for m in matches)

    if allow_multiple and all_matches:
        return all_matches

    raise FileNotFoundError(
        f"Could not find '{filename}' under {start} or any of its parents"
    )


def expand_param_grid(algos_cfg: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand per-algorithm hyperparameter grids into individual configs.

    For each algorithm, forms the Cartesian product of its parameter value
    lists and returns one configuration dictionary per combination, in the
    ``{'name': ..., 'params': ...}`` format used by the preprocessing
    classes.

    Parameters
    ----------
    algos_cfg : dict of str to dict
        Mapping from algorithm name to its grid specification. Each
        specification maps a parameter name to a list of candidate values.
        An ``'alg'`` key, if present, is ignored.

    Returns
    -------
    list of dict
        One dictionary per parameter combination, each with keys:

        ``'name'`` : str
            The algorithm's key in `algos_cfg`.
        ``'params'`` : dict
            Parameter name to value for this combination.

        Algorithms appear in the order of `algos_cfg`; combinations
        follow `itertools.product` order, with the last parameter varying
        fastest.
    """
    expanded = []
    for name, spec in algos_cfg.items():
        keys = [k for k in spec if k != "alg"]
        grids = [spec[k] for k in keys]

        for values in itertools.product(*grids):
            converted_values = []
            for val in values:
                if isinstance(val, str):
                    try:
                        converted_values.append(float(val))
                    except ValueError:
                        converted_values.append(val)
                else:
                    converted_values.append(val)

            params = dict(zip(keys, converted_values))
            expanded.append({"name": name, "params": params})

    return expanded


def peak_finder(spectra: np.ndarray, wavenumbers: np.ndarray,
                prominence: float = 100) -> tuple[list, list, list]:
    """Detect peaks in each spectrum by minimum prominence.

    Applies `scipy.signal.find_peaks` row by row and collects the
    positions, prominences and indices of the detected peaks.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
        Spectra in arbitrary intensity units, one per row.
    wavenumbers : np.ndarray, shape (n_wavenumbers,)
        Wavenumber axis, in cm⁻¹.
    prominence : float, default=100
        Minimum peak prominence, in the intensity units of `spectra`.

    Returns
    -------
    all_positions : list of np.ndarray
        One array per spectrum, with the wavenumbers of the detected
        peaks in cm⁻¹, shape (n_peaks_i,).
    all_heights : list of np.ndarray
        One array per spectrum, with the prominence of each detected
        peak, shape (n_peaks_i,), in intensity units.
    all_idx : list of np.ndarray
        One array per spectrum, with the indices of the detected peaks
        on `wavenumbers`, shape (n_peaks_i,).
    """
    all_idx = []
    all_positions = []
    all_heights = []

    for spectrum in spectra:
        peaks, properties = find_peaks(spectrum, prominence=prominence)
        all_idx.append(peaks)
        all_positions.append(wavenumbers[peaks])
        all_heights.append(properties['prominences'])

    return all_positions, all_heights, all_idx


def group_similar_peaks(all_positions: list, all_heights: list, all_idx: list,
                        reference_peaks: np.ndarray = None,
                        tolerance: float = 2.5,
                        min_occurrence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Group peaks across spectra and score them against reference peaks.

    Pools the peaks detected in all spectra and sorts them by position. A
    new group starts wherever the gap between neighbouring positions is
    larger than `tolerance`. Groups present in fewer than `min_occurrence`
    of the spectra are discarded. Groups whose median position lies within
    `tolerance` of a reference peak are counted as true positives, and the
    F1 score of reference-peak recovery is computed for each spectrum.

    Parameters
    ----------
    all_positions : list of np.ndarray
        One array per spectrum, with the peak positions in cm⁻¹, shape
        (n_peaks_i,).
    all_heights : list of np.ndarray
        One array per spectrum, with the peak magnitudes (prominences from
        `peak_finder`), shape (n_peaks_i,).
    all_idx : list of np.ndarray
        One array per spectrum, with the peak indices on the wavenumber
        axis, shape (n_peaks_i,).
    reference_peaks : np.ndarray, shape (n_reference,), optional
        Reference peak positions, in cm⁻¹. If None or empty, no
        true-positive analysis is done.
    tolerance : float, default=2.5
        Maximum gap, in cm⁻¹, between consecutive sorted positions within
        a group. Also the maximum distance between a group centre and a
        reference peak for a true-positive match.
    min_occurrence : float, default=0.95
        Minimum fraction of spectra in which a group must have a peak to
        be kept.

    Returns
    -------
    peak_matrix_filtered : np.ndarray, shape (n_spectra, n_groups)
        Peak magnitude of each spectrum in each retained group; NaN where
        the spectrum has no peak in that group.
    peak_idx_matrix_filtered : np.ndarray, shape (n_spectra, n_groups)
        Wavenumber indices matching `peak_matrix_filtered`, as floats;
        NaN where absent.
    tp_idx_matrix : np.ndarray, shape (n_spectra, n_tp_groups)
        Columns of `peak_idx_matrix_filtered` for true-positive groups;
        shape (n_spectra, 0) without reference peaks.
    tp_peak_matrix : np.ndarray, shape (n_spectra, n_tp_groups)
        Columns of `peak_matrix_filtered` for true-positive groups;
        shape (n_spectra, 0) without reference peaks.
    F1_per_spectrum : np.ndarray, shape (n_spectra,)
        F1 score of reference-peak recovery for each spectrum; all NaN
        without reference peaks.

    Notes
    -----
    Group centres are the median position of their member peaks. For each
    spectrum, TP is the number of retained true-positive groups in which
    it has a peak, FP the number of other retained groups in which it has
    a peak, and ``FN = n_reference - TP``. Precision and recall are set to
    0 when their denominators are zero, and F1 is 0 when both are 0.

    If a spectrum has several peaks in one group, only the last one in
    position order, i.e. the highest wavenumber, is stored.
    """
    n_spectra = len(all_positions)

    # Collect all peaks
    all_peaks = []
    for spectrum_idx in range(n_spectra):
        for pos, height, idx in zip(all_positions[spectrum_idx],
                                    all_heights[spectrum_idx],
                                    all_idx[spectrum_idx]):
            all_peaks.append({
                'position': pos,
                'height': height,
                'spectrum': spectrum_idx,
                'original_idx': idx
            })

    # Sort by position
    all_peaks.sort(key=lambda p: p['position'])

    # Group nearby peaks using diff-based split
    positions = np.array([p['position'] for p in all_peaks])
    boundaries = (np.where(np.diff(positions) > tolerance)[0] + 1).astype(int)
    groups = np.split(np.arange(len(all_peaks)), boundaries)

    # Compute group centers (median position of each group)
    group_centers = np.array([np.median([all_peaks[i]['position'] for i in group]) for group in groups])

    # Create output arrays
    n_peaks = len(groups)
    peak_matrix = np.full((n_spectra, n_peaks), np.nan)
    peak_idx_matrix = np.full((n_spectra, n_peaks), np.nan)

    for group_idx, group in enumerate(groups):
        for peak_idx in group:
            peak = all_peaks[peak_idx]
            peak_matrix[peak['spectrum'], group_idx] = peak['height']
            peak_idx_matrix[peak['spectrum'], group_idx] = peak['original_idx']

    # Filter: keep only groups present in at least min_occurrence fraction of spectra
    occurrence = np.mean(~np.isnan(peak_matrix), axis=0)
    valid = occurrence >= min_occurrence

    peak_matrix_filtered = peak_matrix[:, valid]
    peak_idx_matrix_filtered = peak_idx_matrix[:, valid]
    group_centers_filtered = group_centers[valid]

    # Calculate median TP, median F1, and build TP index matrix
    if reference_peaks is not None and len(reference_peaks) > 0:
        n_reference = len(reference_peaks)

        is_tp_group = np.array([
            np.any(np.abs(reference_peaks - center) <= tolerance)
            for center in group_centers_filtered
        ])

        tp_idx_matrix = peak_idx_matrix_filtered[:, is_tp_group]
        tp_peak_matrix = peak_matrix_filtered[:, is_tp_group]

        F1_per_spectrum = np.zeros(n_spectra)

        for spectrum_idx in range(n_spectra):
            present = ~np.isnan(peak_matrix_filtered[spectrum_idx, :])

            TP = np.sum(present & is_tp_group)
            FP = np.sum(present & ~is_tp_group)
            FN = n_reference - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            if precision + recall > 0:
                F1_per_spectrum[spectrum_idx] = 2 * precision * recall / (precision + recall)
            else:
                F1_per_spectrum[spectrum_idx] = 0.0
    else:
        tp_idx_matrix = np.full((n_spectra, 0), np.nan)
        tp_peak_matrix = np.full((n_spectra, 0), np.nan)
        F1_per_spectrum = np.full(n_spectra, np.nan)

    return peak_matrix_filtered, peak_idx_matrix_filtered, tp_idx_matrix, tp_peak_matrix, F1_per_spectrum


def calculate_noise_at_peaks(spectra: np.ndarray, wavenumbers: np.ndarray,
                             peak_idx_matrix: np.ndarray,
                             window_cm: float = 50.0) -> np.ndarray:
    """Estimate local noise at each peak from consecutive-spectrum differences.

    For each pair of consecutive spectra, takes their difference and
    computes the standard deviation of that difference in a window centred
    on each peak, divided by √2.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
        Spectra in acquisition order, in arbitrary intensity units.
    wavenumbers : np.ndarray, shape (n_wavenumbers,)
        Wavenumber axis, in cm⁻¹. The median absolute spacing converts
        `window_cm` to points, so ascending and descending axes both work.
    peak_idx_matrix : np.ndarray, shape (n_spectra, n_peaks)
        Wavenumber indices of peaks for each spectrum, as floats, with NaN
        where a peak is absent.
    window_cm : float, default=50.0
        Full width, in cm⁻¹, of the window centred on each peak. Converted
        to a half-width in points of at least 3.

    Returns
    -------
    np.ndarray, shape (n_spectra - 1, n_peaks)
        Noise estimate for each consecutive pair (row ``i`` compares
        spectra ``i`` and ``i + 1``) and each peak, in the intensity units
        of `spectra`. NaN if the peak is absent from both spectra of the
        pair, its index is out of range, or the clipped window has fewer
        than 3 points.
    """
    n_spectra = spectra.shape[0]
    n_peaks = peak_idx_matrix.shape[1]

    # Calculate differences between consecutive spectra
    diff_spectra = np.diff(spectra, axis=0)
    n_diffs = n_spectra - 1

    # Calculate fixed half-window in points (once, outside the loop)
    wn_spacing = np.abs(np.median(np.diff(wavenumbers)))
    half_window = int(np.round((window_cm / 2) / wn_spacing))
    half_window = max(half_window, 3)  # Ensure minimum window size

    # Initialize noise matrix with NaN
    noise_matrix = np.full((n_diffs, n_peaks), np.nan)

    # For each difference spectrum (comparing consecutive pairs)
    for diff_idx in range(n_diffs):
        diff_spectrum = diff_spectra[diff_idx]

        # For each peak group
        for peak_idx in range(n_peaks):
            # Get peak indices from both spectra in the pair
            peak_pos_1 = peak_idx_matrix[diff_idx, peak_idx]
            peak_pos_2 = peak_idx_matrix[diff_idx + 1, peak_idx]

            # Skip if peak doesn't exist in either spectrum
            if np.isnan(peak_pos_1) and np.isnan(peak_pos_2):
                continue

            # Use the available peak position (prefer first spectrum if both exist)
            if not np.isnan(peak_pos_1):
                wn_idx = int(peak_pos_1)
            else:
                wn_idx = int(peak_pos_2)

            # Validate index is within bounds
            if wn_idx < 0 or wn_idx >= len(wavenumbers):
                continue

            # Get window around peak position in the DIFFERENCE spectrum
            start = max(0, wn_idx - half_window)
            end = min(len(wavenumbers), wn_idx + half_window + 1)

            # Extract the region from difference spectrum
            local_diff = diff_spectrum[start:end]

            # Need at least a few points for meaningful std calculation
            if len(local_diff) < 3:
                continue

            # Calculate noise: std / √2
            noise = np.std(local_diff) / np.sqrt(2)

            noise_matrix[diff_idx, peak_idx] = noise

    return noise_matrix


def calculate_snr(peak_matrix: np.ndarray, noise_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a per-peak signal-to-noise ratio from median signal and noise.

    For each peak column, takes the median of the non-NaN signal values and
    the median of the non-NaN noise values, and returns their ratio.

    Parameters
    ----------
    peak_matrix : np.ndarray, shape (n_rows_signal, n_peaks)
        Peak signal for each spectrum and peak, e.g. prominences from
        `group_similar_peaks`, in intensity units; NaN where absent.
    noise_matrix : np.ndarray, shape (n_rows_noise, n_peaks)
        Noise estimate for each row and peak, e.g. from
        `calculate_noise_at_peaks`, in the same intensity units; NaN where
        unavailable. The number of rows may differ from `peak_matrix`.

    Returns
    -------
    snr : np.ndarray, shape (n_peaks,)
        ``median_heights / median_noise`` for each peak; NaN if either
        column has no valid values or the median noise is not positive.
    median_heights : np.ndarray, shape (n_peaks,)
        Median of the non-NaN values in each column of `peak_matrix`;
        NaN if the column has none.
    median_noise : np.ndarray, shape (n_peaks,)
        Median of the non-NaN values in each column of `noise_matrix`;
        NaN if the column has none.

    Notes
    -----
    Signal and noise medians are taken independently over their own rows,
    so the two matrices only need the same number of columns.
    """
    n_peaks = peak_matrix.shape[1]

    snr = np.zeros(n_peaks)
    median_heights = np.zeros(n_peaks)
    median_noise = np.zeros(n_peaks)

    for i in range(n_peaks):
        valid_heights = peak_matrix[:, i][~np.isnan(peak_matrix[:, i])]
        valid_noise = noise_matrix[:, i][~np.isnan(noise_matrix[:, i])]

        median_heights[i] = np.median(valid_heights) if len(valid_heights) > 0 else np.nan
        median_noise[i] = np.median(valid_noise) if len(valid_noise) > 0 else np.nan

        if len(valid_heights) > 0 and len(valid_noise) > 0 and median_noise[i] > 0:
            snr[i] = median_heights[i] / median_noise[i]
        else:
            snr[i] = np.nan

    return snr, median_heights, median_noise


def large_is_best(metric: pd.Series, lower: float | None = None, upper: float | None = None,
                  physical_bound: bool = False) -> pd.Series:
    """Map a larger-is-better metric to a linear desirability in [0, 1].

    Scales the metric linearly between a lower and an upper anchor, with
    the lower anchor mapping to 0 and the upper to 1, and clips the result
    to [0, 1]. By default the anchors are the 5th and 95th percentiles of
    `metric`.

    Parameters
    ----------
    metric : pd.Series
        Metric values, one per candidate (e.g. per preprocessing pipeline).
    lower : float, optional
        Value mapped to desirability 0. Defaults to the 5th percentile of
        `metric`.
    upper : float, optional
        Value mapped to desirability 1. Defaults to the 95th percentile of
        `metric`.
    physical_bound : bool, default=False
        If True, set the desirability to 0 wherever `metric` is below the
        lower anchor.

    Returns
    -------
    pd.Series
        Desirability for each value, in [0, 1], with the index of
        `metric`.
    """
    p5 = lower if lower is not None else metric.quantile(0.05)
    p95 = upper if upper is not None else metric.quantile(0.95)

    if np.isclose(p5, p95):
        p5 = metric.min()
        p95 = metric.max()

    des = (metric - p5) / (p95 - p5)
    des = np.clip(des, 0, 1)

    if physical_bound:
        des[metric < p5] = 0

    return des


def small_is_best(metric: pd.Series, lower: float | None = None, upper: float | None = None,
                  physical_bound: bool = False) -> pd.Series:
    """Map a smaller-is-better metric to a linear desirability in [0, 1].

    Scales the metric linearly between a lower and an upper anchor, with
    the lower anchor mapping to 1 and the upper to 0, and clips the result
    to [0, 1]. By default the anchors are the 5th and 95th percentiles of
    `metric`.

    Parameters
    ----------
    metric : pd.Series
        Metric values, one per candidate (e.g. per preprocessing pipeline).
    lower : float, optional
        Value mapped to desirability 1. Defaults to the 5th percentile of
        `metric`.
    upper : float, optional
        Value mapped to desirability 0. Defaults to the 95th percentile of
        `metric`.
    physical_bound : bool, default=False
        If True, set the desirability to 0 wherever `metric` is below the
        lower anchor, treating such values as physically implausible
        rather than ideal.

    Returns
    -------
    pd.Series
        Desirability for each value, in [0, 1], with the index of
        `metric`.
    """
    p5 = lower if lower is not None else metric.quantile(0.05)
    p95 = upper if upper is not None else metric.quantile(0.95)

    if np.isclose(p5, p95):
        p5 = metric.min()
        p95 = metric.max()

    des = (p95 - metric) / (p95 - p5)
    des = np.clip(des, 0, 1)

    if physical_bound:
        des[metric < p5] = 0

    return des


def find_peak_window(y: np.ndarray, p: int, zero_threshold_frac: float = 0.05) -> tuple[int, int]:
    """Find a symmetric index window around a peak for profile fitting.

    Walks outwards from the peak on each side until the signal falls to a
    fraction of the peak value or starts rising again, i.e. past a local
    minimum. It then shrinks the longer side so the window has the same
    number of points on both sides of the peak.

    Parameters
    ----------
    y : np.ndarray, shape (n_wavenumbers,)
        Single spectrum, in arbitrary intensity units, expected to be
        baseline-corrected.
    p : int
        Index of the peak maximum in `y`.
    zero_threshold_frac : float, default=0.05
        Fraction of ``y[p]`` at or below which the walk stops. With 0, the
        walk stops only where the signal reaches zero or starts rising.

    Returns
    -------
    left : int
        Inclusive start index of the window, ``p - half_width``.
    right : int
        Inclusive end index of the window, ``p + half_width``.

    Notes
    -----
    On each side, the walk stops at the first index that either satisfies
    ``y <= zero_threshold_frac * y[p]`` or is higher than its neighbour
    nearer the peak. That index is included in the window. Equal
    neighbouring values do not stop the walk. The walk also stops at the
    first or last index of `y`. ``half_width`` is the shorter of the two
    walked distances. A peak at either end of `y` therefore gives a
    single-point window.
    """
    n         = len(y)
    threshold = zero_threshold_frac * y[p]

    left_idx = p
    while left_idx > 0:
        if y[left_idx] <= threshold:
            break
        if left_idx < p and y[left_idx] > y[left_idx + 1]:
            break
        left_idx -= 1

    right_idx = p
    while right_idx < n - 1:
        if y[right_idx] <= threshold:
            break
        if right_idx > p and y[right_idx] > y[right_idx - 1]:
            break
        right_idx += 1

    half_width = min(p - left_idx, right_idx - p)
    return max(p - half_width, 0), min(p + half_width, n - 1)


def pseudo_voigt(x: np.ndarray, x0: float, fG: float, fL: float, A: float) -> np.ndarray:
    """Evaluate a pseudo-Voigt profile with Thompson–Cox–Hastings mixing.

    Returns a linear combination of a Lorentzian of FWHM `fL` and a
    Gaussian of FWHM `fG`, both centred on `x0` and normalised to unit
    height. The mixing fraction is computed from the two widths with the
    Thompson–Cox–Hastings approximation.

    Parameters
    ----------
    x : np.ndarray, shape (n_points,)
        Wavenumbers at which to evaluate the profile, in cm⁻¹.
    x0 : float
        Peak centre, in cm⁻¹.
    fG : float
        FWHM of the Gaussian component, in cm⁻¹. Must be positive.
    fL : float
        FWHM of the Lorentzian component, in cm⁻¹. Must be positive.
    A : float
        Peak height at `x0`, in intensity units.

    Returns
    -------
    np.ndarray, shape (n_points,)
        Profile intensity at `x`, in the units of `A`.
    """
    f5 = (fG**5 + 2.69269*fG**4*fL + 2.42843*fG**3*fL**2
          + 4.47163*fG**2*fL**3 + 0.07842*fG*fL**4 + fL**5)
    f   = f5 ** 0.2
    eta = 1.36603*(fL/f) - 0.47719*(fL/f)**2 + 0.11116*(fL/f)**3
    eta = np.clip(eta, 0.0, 1.0)

    sigma_G = fG / (2 * np.sqrt(2 * np.log(2)))
    gamma_L = fL / 2.0

    G = np.exp(-0.5 * ((x - x0) / sigma_G) ** 2)
    L = 1.0 / (1.0 + ((x - x0) / gamma_L) ** 2)

    return A * (eta * L + (1.0 - eta) * G)


def thompson_fwhm(fG, fL):
    """Total FWHM and eta from Thompson et al. (1987)."""
    f5 = (fG**5 + 2.69269*fG**4*fL + 2.42843*fG**3*fL**2
          + 4.47163*fG**2*fL**3 + 0.07842*fG*fL**4 + fL**5)
    f   = f5 ** 0.2
    eta = 1.36603*(fL/f) - 0.47719*(fL/f)**2 + 0.11116*(fL/f)**3
    return f, np.clip(eta, 0.0, 1.0)


def apply_preprocessing(spectra_raw: np.ndarray, wn: np.ndarray, scattering_config: dict,
                        baseline_config: dict, filtering_config: dict) -> np.ndarray:
    """Apply scatter correction, baseline removal and noise filtering.

    Applies scatter correction to the whole matrix, then baseline removal
    and noise filtering to each spectrum in turn. Finally, if any value is
    negative, adds one constant to the whole matrix so that its minimum is
    zero. Any step whose config name is ``'none'`` is skipped.

    Parameters
    ----------
    spectra_raw : np.ndarray, shape (n_spectra, n_wavenumbers)
        Raw spectra, in arbitrary intensity units, one per row. Not
        modified.
    wn : np.ndarray, shape (n_wavenumbers,)
        Wavenumber axis, in cm⁻¹. Not used.
    scattering_config : dict
        ``{'name': ..., 'params': ...}`` for
        `Scattering_correction.correct_scattering`, or ``{'name': 'none'}``.
    baseline_config : dict
        ``{'name': ..., 'params': ...}`` for
        `Baseline_removal.remove_baseline`, or ``{'name': 'none'}``.
    filtering_config : dict
        ``{'name': ..., 'params': ...}`` for `Noise_filter.filter_noise`,
        or ``{'name': 'none'}``.

    Returns
    -------
    np.ndarray, shape (n_spectra, n_wavenumbers)
        Preprocessed spectra as float64, with a minimum of zero or more.

    Raises
    ------
    KeyError
        If any config lacks the ``'name'`` key, or a non-``'none'``
        config lacks ``'params'``.
    ValueError
        If a config names an unknown method.

    Notes
    -----
    Scatter correction sees all spectra at once, so EMSC uses a reference
    estimated from this batch. Baseline removal and filtering receive one
    1D spectrum at a time. The final shift adds :math:`|\\min X|` to
    every value of the matrix when :math:`\\min X < 0`. This shift depends
    on the whole batch.
    """

    scattering_obj = Scattering_correction()
    baseline_obj = Baseline_removal()
    filtering_obj = Noise_filter()

    # Scattering correction: apply to FULL matrix first
    if scattering_config['name'] != 'none':
        X_scatter = scattering_obj.correct_scattering(spectra_raw.copy(), scattering_config)
    else:
        X_scatter = spectra_raw.copy()

    # Baseline and filtering: per-spectrum
    X_corrected = np.zeros_like(X_scatter, dtype=np.float64)
    for idx in range(X_scatter.shape[0]):
        x_baseline = (baseline_obj.remove_baseline(X_scatter[idx], baseline_config)
                      if baseline_config['name'] != 'none' else X_scatter[idx])
        x_filtered = (filtering_obj.filter_noise(x_baseline, filtering_config)
                      if filtering_config['name'] != 'none' else x_baseline)
        X_corrected[idx] = x_filtered

    # Global shift
    global_min = X_corrected.min()
    if global_min < 0:
        X_corrected += abs(global_min)

    return X_corrected


def selection(results, hard_filters, bounds, n_best=1, file_out=None):

    results = results.copy()
    desirability = {'large': large_is_best, 'small': small_is_best}

    # desirabilities
    for des_col, spec in bounds.items():
        metric = spec['metric']
        lower, upper = (
            results.iloc[0][metric].item() if b == 'reference' else b
            for b in (spec['lower'], spec['upper'])
        )
        kwargs = {'physical_bound': spec['physical_bound']} if 'physical_bound' in spec else {}
        results[des_col] = desirability[spec['direction']](
            results[metric], lower=lower, upper=upper, **kwargs
        )

    # geometric mean excludes hard-filtered metrics
    results['Geom_mean'] = results[list(bounds)].apply(gmean, axis=1)

    if file_out is not None:
        results.to_csv(file_out)

    # hard filter
    mask = pd.Series(True, index=results.index)
    for metric, (direction, threshold) in hard_filters.items():
        mask &= results[metric] >= threshold if direction == 'large' else results[metric] <= threshold
    filtered = results[mask].copy()

    return filtered, filtered.nlargest(n_best, 'Geom_mean')