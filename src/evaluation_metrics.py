import numpy as np
import utils as u
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from line_profiler import profile
from scipy.signal import find_peaks


class Policy:
    def __init__(self, config: dict) -> None:
        self.components = config["n_components"]
        self.intensity_tolerance = config["intensity_tolerance"]
        self.reference_peaks = config["reference_peaks"]
        self.wn_tolerance_spectra = config["wn_tolerance_spectra"]
        self.n_splits = config["n_splits"]
        self.wn_tolerance_coef = config["wn_tolerance_coef"]

    # Feature Preservation Metrics
    @profile
    def peak_counter(self, spectra: np.ndarray, wn: np.ndarray, intensity_tolerance: float, reference_peaks: list,
                     wn_tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Detect peaks in spectra and match them to reference positions.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_points)
            Processed spectra, in arbitrary intensity units.
        wn : np.ndarray, shape (n_points,)
            Wavenumber axis, in cm⁻¹.
        intensity_tolerance : float
            Intensity threshold that `u.peak_finder` uses for peak detection.
        reference_peaks : list of float
            Reference peak positions, in cm⁻¹.
        wn_tolerance : float
            Maximum wavenumber distance, in cm⁻¹, for a detected peak to be
            grouped with a reference peak.

        Returns
        -------
        peak_matrix : np.ndarray
            Detected peaks, grouped across spectra by similar position.
        peak_idx_matrix : np.ndarray
            Indices on `wn` of the peaks in `peak_matrix`.
        tp_idx_matrix : np.ndarray
            Indices on `wn` of the true-positive peaks, i.e. detected peaks
            within `wn_tolerance` of a reference peak.
        tp_peak_matrix : np.ndarray
            True-positive peaks matched to `reference_peaks`.
        median_F1 : float
            Median F1 score of peak recovery against `reference_peaks`,
            taken across spectra.

        Notes
        -----
        The method runs in two steps:

        1. `u.peak_finder` detects the peak positions, heights and indices in
           each spectrum.
        2. `u.group_similar_peaks` groups the detected peaks and scores them
           against `reference_peaks` within `wn_tolerance`.

        The arguments are used instead of any instance attributes with the
        same names.
        """

        all_positions, all_heights, all_idx = u.peak_finder(spectra, wn,
                                                            intensity_tolerance)

        peak_matrix, peak_idx_matrix, tp_idx_matrix, tp_peak_matrix, median_F1 = u.group_similar_peaks(all_positions,
                                                                                                       all_heights,
                                                                                                       all_idx,
                                                                                                       reference_peaks,
                                                                                                       wn_tolerance)

        return peak_matrix, peak_idx_matrix, tp_idx_matrix, tp_peak_matrix, median_F1

    @profile
    def peak_selectivity(self, spectra: np.ndarray, y: np.ndarray,
                         tp_idx_matrix: np.ndarray, half_window: int = 5) -> float:
        """Score how much more peak regions correlate with `y` than baseline.

        Computes the Pearson correlation between each wavenumber and the
        response across spectra. It then returns the median correlation inside
        windows around true-positive peaks minus the median correlation in the
        remaining wavenumbers.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
            Processed spectra, in arbitrary intensity units.
        y : np.ndarray, shape (n_spectra,) or (n_spectra, 1)
            Response values, e.g. analyte concentrations, one per spectrum.
            Arrays with more than one dimension are flattened.
        tp_idx_matrix : np.ndarray, shape (n_spectra, n_peaks)
            Wavenumber indices of true-positive peaks for each spectrum, as
            floats, with NaN where no peak was matched.
        half_window : int, default=5
            Half-width, in wavenumber points, of the window marked as peak
            region on each side of a true-positive index.

        Returns
        -------
        float
            Selectivity, "median(r_peaks) - median(r_baseline)", where `r`
            is the signed Pearson correlation coefficient with `y`. Positive
            values mean the peak regions correlate more strongly with `y`
            than the baseline does.

        Notes
        -----
        True-positive indices are pooled across all spectra, so the peak mask
        is the union of the windows around every index matched in any
        spectrum. Windows are clipped at the edges of the spectral axis.

        The correlation is signed, so peaks that are anti-correlated with `y`
        lower `r_peaks`.
        """

        y_flat = y.flatten() if y.ndim > 1 else y
        n_spectra, n_wavenumbers = spectra.shape

        # Compute Pearson r for each wavenumber across all spectra
        correlations = np.array([
            np.corrcoef(spectra[:, i], y_flat)[0, 1]
            for i in range(n_wavenumbers)
        ])

        # Collect all unique TP indices across all spectra
        all_tp_indices = set()
        for spectrum_idx in range(n_spectra):
            tp_indices = tp_idx_matrix[spectrum_idx, :]
            valid_indices = tp_indices[~np.isnan(tp_indices)].astype(int)
            all_tp_indices.update(valid_indices)

        # Peak mask
        peak_mask = np.zeros(n_wavenumbers, dtype=bool)
        for idx in all_tp_indices:
            start = max(0, idx - half_window)
            end = min(n_wavenumbers, idx + half_window + 1)
            peak_mask[start:end] = True

        # Baseline mask (everything except peaks)
        baseline_mask = ~peak_mask

        r_peaks = np.median(correlations[peak_mask])
        r_baseline = np.median(correlations[baseline_mask])

        return r_peaks - r_baseline

    # Peaks Shape Metrics
    @profile
    def voigt_FWHM(self, spectra: np.ndarray, wavenumbers: np.ndarray,
                   peak_idx_matrix: np.ndarray,
                   zero_threshold_frac: float = 0.00) -> tuple[np.ndarray, np.ndarray]:
        """Fit pseudo-Voigt profiles to detected peaks and summarise FWHM.

        For every valid peak, fits a pseudo-Voigt profile within an adaptive
        window, converts the fitted Gaussian and Lorentzian widths to a total
        FWHM, and records the coefficient of determination of the fit. Returns
        the per-spectrum medians of both quantities.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
            Processed spectra, in arbitrary intensity units. Fitting anchors
            the profile to zero at both window edges, so spectra are expected
            to be baseline-corrected.
        wavenumbers : np.ndarray, shape (n_wavenumbers,)
            Wavenumber axis, in cm⁻¹, assumed ascending and evenly spaced.
        peak_idx_matrix : np.ndarray, shape (n_spectra, n_peaks)
            Wavenumber indices of peak maxima for each spectrum, as floats,
            with NaN where no peak is present.
        zero_threshold_frac : float, default=0.00
            Fraction of the peak height passed to `u.find_peak_window` to
            set where the fitting window ends on each side.

        Returns
        -------
        FWHM_median : np.ndarray, shape (n_spectra, 1)
            Median total FWHM, in cm⁻¹, of the successfully fitted peaks in
            each spectrum; NaN if no peak in that spectrum was fitted.
        R2_median : np.ndarray, shape (n_spectra, 1)
            Median coefficient of determination of those fits; NaN if no peak
            in that spectrum was fitted.

        Notes
        -----
        A peak is skipped, and left as NaN before the median, if:

        - its index is NaN or outside "[0, n_wavenumbers)";
        - its intensity is not positive;
        - its window contains fewer than 5 points;
        - the admissible centre interval is narrower than 1e-3 cm⁻¹;
        - `curve_fit` raises `RuntimeError` or `ValueError`.

        One extra point at zero intensity is added on each side of the window,
        one wavenumber step beyond its edges. The fit parameters are the
        centre `x0`, the Gaussian width `fG`, the Lorentzian width `fL` and
        the amplitude `A`. They are bounded to "x0 ± 5" cm⁻¹ (clipped to
        the window), "fG, fL ∈ [1e-3, 30]" cm⁻¹ and "A ≥ 0". The initial
        width estimate is the span of window points at or above half the peak
        height, clipped to [0.5, 30] cm⁻¹ and split equally between `fG`
        and `fL`. R² is computed over the window including the two zero
        anchors, and is set to 0 when the fitted data have zero variance.

        References
        ----------
        [1] P. Thompson, D. E. Cox, J. B. Hastings, J. Appl. Crystallogr. 1987, 20, 79.
        """

        n_spectra, n_peaks = peak_idx_matrix.shape
        n_wavenumbers = len(wavenumbers)

        FWHM_matrix = np.full((n_spectra, n_peaks), np.nan)
        R2_matrix = np.full((n_spectra, n_peaks), np.nan)

        # Flatten valid work items upfront
        work_items = []
        for si in range(n_spectra):
            y = spectra[si]
            for pi in range(n_peaks):
                peak_idx = peak_idx_matrix[si, pi]
                if np.isnan(peak_idx):
                    continue
                p = int(peak_idx)
                if p < 0 or p >= n_wavenumbers:
                    continue
                if y[p] <= 0:
                    continue
                work_items.append((si, pi, p))

        for si, pi, p in work_items:
            y = spectra[si]
            x0 = wavenumbers[p]
            A = y[p]

            # Adaptive symmetric window
            left_idx, right_idx = u.find_peak_window(y, p, zero_threshold_frac)
            x_win = wavenumbers[left_idx:right_idx + 1]
            y_win = y[left_idx:right_idx + 1]

            if len(x_win) < 5:
                continue

            # Zero anchors at both edges
            dx = x_win[1] - x_win[0]
            x_fit = np.empty(len(x_win) + 2)
            y_fit = np.zeros(len(x_win) + 2)
            x_fit[0] = x_win[0] - dx
            x_fit[1:-1] = x_win
            x_fit[-1] = x_win[-1] + dx
            y_fit[1:-1] = y_win

            x_min, x_max = x_fit[0], x_fit[-1]
            x0_lower = max(x0 - 5, x_min)
            x0_upper = min(x0 + 5, x_max)
            if x0_upper <= x0_lower + 1e-3:
                continue

            # Moment-based initial guess for fG and fL
            half_max = A / 2.0
            above_half = x_win[y_win >= half_max]
            if len(above_half) >= 2:
                fwhm_est = np.clip(above_half[-1] - above_half[0], 0.5, 30.0)
            else:
                fwhm_est = np.clip((x_max - x_min) / 4, 0.5, 30.0)

            fG_init = fwhm_est * 0.5
            fL_init = fwhm_est * 0.5

            # Bounds: fG and fL both positive, max 30 cm⁻¹
            p0 = [np.clip(x0, x0_lower + 1e-3, x0_upper - 1e-3),
                  fG_init, fL_init, max(A, 1e-6)]
            lower_bounds = [x0_lower, 1e-3, 1e-3, 0.0]
            upper_bounds = [x0_upper, 30.0, 30.0, np.inf]

            try:
                popt, _ = curve_fit(
                    u.pseudo_voigt,
                    x_fit, y_fit,
                    p0=p0, bounds=(lower_bounds, upper_bounds),
                    maxfev=1000, method='trf',
                    ftol=1e-4, xtol=1e-4, gtol=1e-4
                )
            except (RuntimeError, ValueError):
                continue

            _, fit_fG, fit_fL, _ = popt

            # Thompson et al. (1987) total FWHM
            FWHM, _ = u.thompson_fwhm(fit_fG, fit_fL)

            fitted_curve = u.pseudo_voigt(x_fit, *popt)
            ss_res = np.sum((y_fit - fitted_curve) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            FWHM_matrix[si, pi] = FWHM
            R2_matrix[si, pi] = r_squared

        # Median metrics per spectrum
        R2_median = np.nanmedian(R2_matrix, axis=1, keepdims=True)
        FWHM_median = np.nanmedian(FWHM_matrix, axis=1, keepdims=True)

        return FWHM_median, R2_median

    # Noise Estimation Metrics
    @profile
    def baseline_drift(self, spectra: np.ndarray) -> np.ndarray:
        """Quantify spectrum-to-spectrum baseline across consecutive spectra.

        Takes the difference between each pair of consecutive spectra and
        summarises it as the median absolute difference over wavenumbers.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
            Processed spectra in acquisition order, in arbitrary intensity
            units.

        Returns
        -------
        np.ndarray, shape (n_spectra - 1, 1)
            Median absolute difference between spectrum "i + 1" and
            spectrum "i", in the intensity units of `spectra`.
        """
        diff = np.diff(spectra, axis=0)
        mad = np.median(np.abs(diff), axis=1, keepdims=True)
        return mad

    @profile
    def snr(self, spectra: np.ndarray, wavenumber: np.ndarray, peak_idx_matrix: np.ndarray,
            peak_matrix: np.ndarray, window_cm: float = 50) -> float:
        """Compute the median signal-to-noise ratio of detected peaks.

        Estimates the local noise around each detected peak and divides each
        peak signal by it. The result is the median ratio over all spectra and
        peaks.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
            Processed spectra, in arbitrary intensity units.
        wavenumber : np.ndarray, shape (n_wavenumbers,)
            Wavenumber axis, in cm⁻¹.
        peak_idx_matrix : np.ndarray, shape (n_spectra, n_peaks)
            Wavenumber indices of the detected peaks, as floats, with NaN
            where no peak is present.
        peak_matrix : np.ndarray, shape (n_spectra, n_peaks)
            Peak signals matching `peak_idx_matrix`, in the intensity units
            of `spectra`, with NaN where no peak is present.
        window_cm : float, default=50
            Width, in cm⁻¹, of the region around each peak used by
            `u.calculate_noise_at_peaks` to estimate noise.

        Returns
        -------
        float
            Median SNR, pooled over all spectra and peaks, ignoring NaN
            entries. NaN if no finite SNR value exists.

        Notes
        -----
        Noise estimation and the SNR definition are delegated to
        `u.calculate_noise_at_peaks` and `u.calculate_snr`. Only the first
        of the three outputs of `u.calculate_snr` is used.
        """

        noise_matrix = u.calculate_noise_at_peaks(spectra, wavenumber, peak_idx_matrix, window_cm=window_cm)
        snr_arr, _, _ = u.calculate_snr(peak_matrix, noise_matrix)
        return float(np.nanmedian(snr_arr))

    # PLS Performance Metrics
    @profile
    def calibration_fit(self, spectra: np.ndarray, y: np.ndarray, n_splits: int = 5,
                        n_components: int = 1) -> tuple[float, np.ndarray]:
        """Cross-validate a PLS model and report its worst stratified error.

        Runs venetian-blind cross-validation of a PLS regression and pools the
        out-of-fold predictions. It splits them into quartile bins of `y` and
        returns the largest per-bin RMSE, normalised by the interquartile range
        of `y`, together with the coefficients of a PLS model refitted on all
        data.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
            Processed spectra, in arbitrary intensity units.
        y : np.ndarray, shape (n_spectra,) or (n_spectra, 1)
            Reference response values, e.g. analyte concentrations. Flattened
            to 1D.
        n_splits : int, default=5
            Number of venetian-blind folds.
        n_components : int, default=1
            Number of PLS latent variables.

        Returns
        -------
        worst_rrmse : float
            Largest per-bin cross-validated RMSE, as a percentage of the
            interquartile range of `y`. 0.0 if no bin could be evaluated.
        coef : np.ndarray, shape (n_wavenumbers,)
            Regression coefficients of the PLS model fitted on all spectra,
            in units of `y` per unit intensity.

        Notes
        -----
        Folds are skipped if their training set has fewer than `n_components`
        samples or their test set is empty.
        """

        X = spectra
        y = np.asarray(y).ravel()
        n = X.shape[0]
        pls = PLSRegression(n_components=n_components, scale=False)

        y_true_all, y_pred_all = [], []

        # venetian blind
        for fold in range(n_splits):
            test_idx = np.arange(fold, n, n_splits)
            train_idx = np.setdiff1d(np.arange(n), test_idx)

            if len(train_idx) < n_components or len(test_idx) == 0:
                continue

            pls.fit(X[train_idx], y[train_idx])
            pred = pls.predict(X[test_idx]).ravel()
            true = y[test_idx]
            y_pred_all.extend(pred)
            y_true_all.extend(true)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        # stratified rRMSE: worst quartile
        y_iqr = np.percentile(y, 75) - np.percentile(y, 25)
        quartile_edges = np.unique(np.percentile(y, [0, 25, 50, 75, 100]))
        n_bins = len(quartile_edges) - 1
        worst_rrmse = 0.0

        if n_bins > 0 and y_iqr > 0:
            bin_indices = np.digitize(y_true_all, quartile_edges[1:-1], right=False)
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() < 2:
                    continue

                bin_rmse = np.sqrt(mean_squared_error(y_true_all[mask], y_pred_all[mask]))
                worst_rrmse = max(worst_rrmse, (bin_rmse / y_iqr) * 100)

        # full-data fit
        pls.fit(X, y)
        coef = pls.coef_.ravel()

        return worst_rrmse, coef

    @profile
    def coefficient_peak(self, reg_coef: np.ndarray, wn: np.ndarray, reference_peaks: np.ndarray,
                         wn_tolerance: float = 5.0) -> float:
        """Score how well regression coefficients highlight reference peaks.

        Detects local maxima in the positive part of the regression
        coefficients and matches them to the reference peak positions. It then
        computes the F1 score of that matching and weights it by the fraction
        of squared coefficient energy that is non-negative.

        Parameters
        ----------
        reg_coef : np.ndarray, shape (n_wavenumbers,)
            Regression coefficients, e.g. from `calibration_fit`, in units of
            the response per unit intensity.
        wn : np.ndarray, shape (n_wavenumbers,)
            Wavenumber axis, in cm⁻¹.
        reference_peaks : np.ndarray, shape (n_reference_peaks,) or None
            Reference peak positions, in cm⁻¹.
        wn_tolerance : float, default=5.0
            Maximum distance, in cm⁻¹, between a coefficient peak and a
            reference peak for the reference peak to count as matched.

        Returns
        -------
        float
            Score in [0, 1] for typical inputs, "F1 * (1 - neg_ratio)",
            where `neg_ratio` is the share of "sum(reg_coef ** 2)"
            contributed by negative coefficients. 0 if `reference_peaks` is
            None or empty, or if no coefficient peak is found.
        """
        # Use absolute value to capture both positive and negative important features
        reg_coef_clip = np.clip(reg_coef, 0, None)

        peak_idx, _ = find_peaks(reg_coef_clip)
        coef_peak_positions = wn[peak_idx]

        # Count how many reference peaks have a matching coefficient peak
        if reference_peaks is None or len(reference_peaks) == 0:
            return 0

        if len(coef_peak_positions) == 0:
            return 0

        tp_count = 0
        for ref_pos in reference_peaks:
            if np.any(np.abs(coef_peak_positions - ref_pos) <= wn_tolerance):
                tp_count += 1

        precision = tp_count / len(coef_peak_positions)
        recall = tp_count / len(reference_peaks)

        f1 = 2 * ((recall * precision) / (recall + precision)) if precision > 0 and recall > 0 else 0

        neg_energy = np.sum(reg_coef[reg_coef < 0] ** 2)
        total_energy = np.sum(reg_coef ** 2)
        neg_ratio = neg_energy / total_energy

        return f1 * (1 - neg_ratio)


    # ---------- Orchestration ----------
    @profile
    def evaluate(self, spectra: np.ndarray, wn: np.ndarray, y: np.ndarray, snr_raw: float = None,
                 sel_raw_median: float = None) -> np.ndarray:
        """Compute the policy metric vector for a set of processed spectra.

        Runs the feature-preservation, peak-shape, noise and PLS metrics on
        `spectra`, reduces each one to a scalar, and returns them as a vector
        rounded to four decimals. Thresholds, tolerances, reference peaks and
        PLS settings are taken from the instance attributes.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_spectra, n_wavenumbers)
            Processed spectra in acquisition order, in arbitrary intensity
            units.
        wn : np.ndarray, shape (n_wavenumbers,)
            Wavenumber axis, in cm⁻¹.
        y : np.ndarray, shape (n_spectra,) or (n_spectra, 1)
            Reference analyte concentrations.
        snr_raw : float, optional
            Median SNR of the unprocessed spectra. If given, the SNR metric
            is reported relative to it; otherwise the absolute SNR is used.
        sel_raw_median : float, optional
            Peak selectivity of the unprocessed spectra. If given, the
            selectivity metric is reported as its closeness to this value;
            otherwise the absolute selectivity is used.

        Returns
        -------
        np.ndarray, shape (8,)
            Metric vector, each entry rounded to four decimals:

            0. Median F1 score of reference-peak recovery in the spectra
               (`peak_counter`).
            1. Selectivity, "exp(-|ln(sel_proc / sel_raw_median)|)" if
               `sel_raw_median` is given, otherwise "sel_proc"
               (`peak_selectivity`).
            2. Median total FWHM of fitted pseudo-Voigt peaks, in cm⁻¹
               (`voigt_FWHM`).
            3. Median R² of those fits (`voigt_FWHM`).
            4. Median baseline MAD, in intensity units
               (`baseline_drift`).
            5. SNR, "snr_proc / snr_raw" if `snr_raw` is given,
               otherwise "snr_proc" (`snr`).
            6. Worst quartile-bin cross-validated rRMSECV, as a percentage of
               the interquartile range of `y` (`calibration_fit`).
            7. Coefficient-peak score of the full-data PLS model
               (`coefficient_peak`).

        Notes
        -----
        `peak_selectivity` receives "self.wn_tolerance_spectra" as its
        `half_window` argument. `snr` is computed on the true-positive peaks
        only, using "tp_idx_matrix" and "tp_peak_matrix" with a fixed
        50 cm⁻¹ noise window. With a reference value, the selectivity
        is 1 when processing leaves selectivity unchanged and decreases
        symmetrically for gains and losses.

        """
        # Feature Preservation
        peak_matrix, peak_idx_matrix, tp_idx_matrix, tp_peak_matrix, F1_TP_per_spectra = self.peak_counter(spectra,
                                                                                                           wn,
                                                                                                           self.intensity_tolerance,
                                                                                                           self.reference_peaks,
                                                                                                           self.wn_tolerance_spectra)
        sel_proc = self.peak_selectivity(spectra, y, tp_idx_matrix, self.wn_tolerance_spectra)

        # Peaks Shape
        FWHM_matrix, R2_matrix = self.voigt_FWHM(spectra, wn, peak_idx_matrix)

        # Noise Evaluation
        mad_baseline = self.baseline_drift(spectra)
        snr_processed = self.snr(spectra, wn, tp_idx_matrix, tp_peak_matrix, window_cm=50)
        snr_ratio = snr_processed / snr_raw if snr_raw is not None else snr_processed

        # PLS
        RMSE_pct, coef = self.calibration_fit(spectra, y, self.n_splits, self.components)
        F1_coef = self.coefficient_peak(coef, wn, self.reference_peaks, self.wn_tolerance_coef)

        # Aggregation
        median_F1_TP = np.nanmedian(F1_TP_per_spectra)
        sel_proc_median = np.nanmedian(sel_proc)
        median_FWHM = np.nanmedian(FWHM_matrix)
        median_Voigt = np.nanmedian(R2_matrix)
        median_mad_baseline = np.nanmedian(mad_baseline)

        median_selectivity = np.exp(-np.abs(np.log(sel_proc_median / sel_raw_median))) if sel_raw_median is not None \
            else sel_proc_median

        res = np.array([
            float(round(median_F1_TP, 4)),
            float(round(median_selectivity, 4)),
            float(round(median_FWHM, 4)),
            float(round(median_Voigt, 4)),
            float(round(median_mad_baseline, 4)),
            float(round(snr_ratio, 4)),
            float(round(RMSE_pct, 4)),
            float(round(F1_coef, 4))
        ], dtype=np.float64)

        return res

