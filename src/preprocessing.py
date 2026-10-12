import numpy as np
from pybaselines import Baseline
from typing import Dict
import chemotools.scatter as sc
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter, median_filter


class Baseline_removal:
    """Subtract a fitted baseline from spectra using `pybaselines`.

    Wraps a `pybaselines.Baseline` instance and dispatches to any of its
    baseline-fitting methods by name, so the algorithm and its
    hyperparameters can be chosen from a configuration dictionary.

    Attributes
    ----------
    br : pybaselines.Baseline
        Baseline fitter created without `x_data`, so methods assume
        evenly spaced points indexed from 0 to "n_points - 1".
    """

    def __init__(self) -> None:
        self.br = Baseline()

    def remove_baseline(self, spectra: np.ndarray, config: Dict) -> np.ndarray:
        """Fit a baseline with the configured method and subtract it.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_points,)
            Spectrum intensities, in arbitrary units, along the
            wavenumber axis.
        config : dict
            Method specification with two keys:

            "'name'" : str
                Name of a `pybaselines.Baseline` method, e.g. "'asls'".
            "'params'" : dict
                Keyword arguments passed unchanged to that method.

        Returns
        -------
        np.ndarray, shape (n_points,)
            Baseline-corrected spectrum, "spectra - baseline", in the
            same units as `spectra`.

        Raises
        ------
        KeyError
            If `config` lacks the "'name'" or "'params'" key.
        ValueError
            If `config['name']` is not an attribute of
            `pybaselines.Baseline`.

        Notes
        -----
        The parameter dictionary returned by the `pybaselines` method is
        discarded.
        """
        name = config['name']
        params = config['params']

        baseline_method = getattr(self.br, name, None)

        if baseline_method is None:
            raise ValueError(f"Unknown baseline method: {name}")

        baseline, _ = baseline_method(spectra, **params)
        spectra_clean = spectra - baseline
        return spectra_clean


class Scattering_correction:
    """Correct scattering effects in spectra using `chemotools` scalers.

    Maps short method names to scatter-correction transformers in the
    `sc` module and applies the selected one with "fit_transform".

    Attributes
    ----------
    scattering_methods : dict of str to str
        Lookup from short method name to transformer class name in `sc`:
        "'EMSC'" (extended multiplicative scatter correction),
        "'SNV'" (standard normal variate) and "'RNV'" (robust normal
        variate).
    """

    def __init__(self) -> None:
        self.scattering_methods = {
            'EMSC': 'ExtendedMultiplicativeScatterCorrection',
            'SNV': 'StandardNormalVariate',
            'RNV': 'RobustNormalVariate'
        }

    def correct_scattering(self, spectra: np.ndarray, config: Dict) -> np.ndarray:
        """Apply the configured scatter correction to one or more spectra.

        A 1D spectrum is reshaped to a single-row matrix before fitting and
        flattened again afterwards, so the output shape matches the input.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_points,) or (n_spectra, n_points)
            Spectrum intensities, in arbitrary units, along the wavenumber
            axis. Rows are individual spectra.
        config : dict
            Method specification with two keys:

            "'name'" : str
                Short method name, one of "'EMSC'", "'SNV'" or
                "'RNV'".
            "'params'" : dict
                Keyword arguments passed to the transformer constructor.

        Returns
        -------
        np.ndarray, shape (n_points,) or (n_spectra, n_points)
            Scatter-corrected spectra, with the same shape as `spectra`.

        Raises
        ------
        KeyError
            If `config` lacks the "'name'" or "'params'" key.
        ValueError
            If `config['name']` is not a key of `scattering_methods`.

        Notes
        -----
        The transformer is refitted on `spectra` at every call via
        "fit_transform"; no fitted state is kept between calls.
        """
        name = config['name']
        params = config['params']

        # Get the class name
        class_name = self.scattering_methods.get(name)
        if class_name is None:
            raise ValueError(f"Unknown scattering correction method: '{name}'")

        scattering_class = getattr(sc, class_name, None)
        scattering_method = scattering_class(**params)


        spectra_2d = spectra.reshape(1, -1) if spectra.ndim == 1 else spectra
        result = scattering_method.fit_transform(spectra_2d)

        return result.ravel() if spectra.ndim == 1 else result


class Noise_filter:
    """Smooth spectra with a selectable noise filter.

    Maps filter names to SciPy smoothing functions and calls the selected
    one with keyword arguments taken from a configuration dictionary.

    Attributes
    ----------
    filters : dict of str to callable
        Lookup from filter name to function: "'savitzky-golay'"
        (`savgol_filter`), "'gaussian'" (`gaussian_filter`) and
        "'median_filter'" (`median_filter`).
    """

    def __init__(self) -> None:
        self.filters = {
            'savitzky-golay': savgol_filter,
            'gaussian': gaussian_filter,
            'median_filter': median_filter
        }

    def filter_noise(self, spectra: np.ndarray, config: Dict) -> np.ndarray:
        """Apply the configured noise filter to spectra.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_points,) or (n_spectra, n_points)
            Spectrum intensities, in arbitrary units, along the wavenumber
            axis. Rows are individual spectra.
        config : dict
            Filter specification with two keys:

            "'name'" : str
                Filter name, one of "'savitzky-golay'", "'gaussian'"
                or "'median_filter'".
            "'params'" : dict
                Keyword arguments passed unchanged to the filter function,
                e.g. "window_length" and "polyorder" for
                Savitzky–Golay, "sigma" for Gaussian, "size" for
                median.

        Returns
        -------
        np.ndarray, shape (n_points,) or (n_spectra, n_points)
            Filtered spectra, with the same shape as `spectra`.

        Raises
        ------
        KeyError
            If `config` lacks the "'name'" or "'params'" key.
        ValueError
            If `config['name']` is not a key of `filters`.
        """
        name = config['name']
        params = config['params']

        filter_func = self.filters.get(name)

        if filter_func is None:
            raise ValueError(f"Unknown noise filter: {name}")

        return filter_func(spectra, **params)