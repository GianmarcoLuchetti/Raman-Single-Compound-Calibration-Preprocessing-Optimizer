import time
import itertools
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import yaml
import pandas as pd
from joblib import Parallel, delayed
import utils as u
from evaluation_metrics import Policy
from preprocessing import Baseline_removal, Scattering_correction, Noise_filter
import warnings


def process_pipeline(
    i: int,
    base_cfg: Dict[str, Any],
    scat_cfg: Dict[str, Any],
    filt_cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    wn: np.ndarray,
    snr_raw: float,
    sel_raw_median: float,
    eval_config: Dict[str, Any],
    metric_names: List
) -> Optional[Dict[str, Any]]:
    """Preprocess spectra with one pipeline combination and evaluate it.

    Applies scatter correction, then per-spectrum baseline removal and
    noise filtering, then a global non-negativity shift. It then scores
    the result with `Policy.evaluate`. Designed as a parallel worker: all
    processing objects are created inside the call.

    Parameters
    ----------
    i : int
        Combination index, used only to label log messages.
    base_cfg : dict
        Baseline configuration, ``{'name': ..., 'params': ...}``, or
        ``{'name': 'none'}`` to skip the step.
    scat_cfg : dict
        Scatter-correction configuration, same format.
    filt_cfg : dict
        Noise-filter configuration, same format.
    X : np.ndarray, shape (n_spectra, n_wavenumbers)
        Raw spectra in acquisition order, in arbitrary intensity units.
        Not modified.
    y : np.ndarray, shape (n_spectra,) or (n_spectra, 1)
        Reference response values, e.g. analyte concentrations.
    wn : np.ndarray, shape (n_wavenumbers,)
        Wavenumber axis, in cm⁻¹. Used only in evaluation.
    snr_raw : float
        Median SNR of the raw spectra, passed to `Policy.evaluate`.
    sel_raw_median : float
        Peak selectivity of the raw spectra, passed to `Policy.evaluate`.
    eval_config : dict
        Configuration used to construct the `Policy`.
    metric_names : list of str
        Names for the entries of the metric vector, in the order returned
        by `Policy.evaluate`.

    Returns
    -------
    dict or None
        On success, a dictionary with keys ``'baseline_config'``,
        ``'scattering_config'`` and ``'filtering_config'`` holding the
        input configs, plus one key per name in `metric_names` mapped to
        its metric value. None if preprocessing or evaluation raised an
        exception.

    Raises
    ------
    KeyError
        If any config lacks ``'name'``, or `eval_config` lacks a key that
        `Policy` requires. Raised before the guarded blocks.

    Notes
    -----
    Exceptions raised during preprocessing or evaluation are caught,
    reported with ``print`` and turned into a None return. Status and
    timing messages are printed to standard output. Metric names are
    paired with metric values by `zip`, which stops at the shorter of the
    two.

    See Also
    --------
    apply_preprocessing : The same preprocessing as a standalone function.
    Policy.evaluate : Computes the metric vector.
    """

    combo_start = time.time()
    print(
        f"[{i}] 🚧 Starting: {base_cfg['name']} + {scat_cfg['name']} + {filt_cfg['name']}"
    )

    # Create objects inside worker (avoids pickling issues)
    baseline_obj = Baseline_removal()
    scattering_obj = Scattering_correction()
    filtering_obj = Noise_filter()
    policy_obj = Policy(eval_config)

    try:
        # Scattering correction
        if scat_cfg['name'] != 'none':
            X_scatter = scattering_obj.correct_scattering(X, scat_cfg)
        else:
            X_scatter = X.copy()

        # Baseline and filtering: per-spectrum operations
        X_corrected = np.zeros_like(X_scatter, dtype=np.float64)
        for idx in range(X_scatter.shape[0]):
            x_baseline = (baseline_obj.remove_baseline(X_scatter[idx], base_cfg)
                          if base_cfg['name'] != 'none' else X_scatter[idx])
            x_filtered = (filtering_obj.filter_noise(x_baseline, filt_cfg)
                          if filt_cfg['name'] != 'none' else x_baseline)

            X_corrected[idx] = x_filtered

        # Global shift: single value for entire dataset (not per-spectrum)
        global_min = X_corrected.min()
        if global_min < 0:
            X_corrected += abs(global_min)

    except Exception as e:
        print(
            f"[{i}] ❌ Preprocess error "
            f"({base_cfg['name']} + {scat_cfg['name']} + {filt_cfg['name']}): {e}"
        )
        return None

    # ---- Evaluation
    try:
        evaluation = policy_obj.evaluate(X_corrected, wn, y, snr_raw, sel_raw_median)
        combo_time = time.time() - combo_start
        print(
            f"[{i}] ✅ Evaluation complete in {combo_time:.2f}s — "
            f"{base_cfg['name']} + {scat_cfg['name']} + {filt_cfg['name']}"
        )

        # Convert np array to dict using metric names
        eval_dict = dict(zip(metric_names, evaluation))

        result = {
            "baseline_config": base_cfg,
            "scattering_config": scat_cfg,
            "filtering_config": filt_cfg,
            **eval_dict,
        }
        return result
    except Exception as e:
        print(
            f"[{i}] ❌ Evaluation failed "
            f"({base_cfg['name']} + {scat_cfg['name']} + {filt_cfg['name']}): {e}"
        )
        return None


def grid_search(cfg: dict, x_data: str, y_data: str) -> pd.DataFrame | None:
    """Run the preprocessing grid search and collect pipeline metrics.

    Expands the baseline, scatter-correction and filtering grids into all
    pipeline combinations, loads the spectra and reference values, and
    computes the raw-data SNR and selectivity used as references. It then
    preprocesses and evaluates every combination in parallel and gathers
    the successful results in a DataFrame.

    Parameters
    ----------
    cfg : dict
        Run configuration with keys:

        ``'evaluation_policy'`` : dict
            `Policy` configuration, plus ``'metrics'``, the list of
            metric names in the order returned by `Policy.evaluate`.
        ``'baseline'``, ``'scattering'``, ``'filtering'`` : dict
            Per-algorithm parameter grids for `expand_param_grid`.
    x_data : str
        File name of the spectra CSV, located with `get_data_path`. It
        must have a ``'Sample'`` index column and wavenumbers in cm⁻¹ as
        the remaining column headers.
    y_data : str
        File name of the reference-values CSV, located with
        `get_data_path`. It must have a ``'Sample'`` index column.

    Returns
    -------
    pd.DataFrame or None
        One row per successful pipeline. Columns ``'baseline_config'``,
        ``'scattering_config'`` and ``'filtering_config'`` hold the step
        configs, and one numeric column per metric name holds the
        values. None if no combination succeeded.

    Raises
    ------
    KeyError
        If a required configuration key is missing.
    FileNotFoundError, MultipleMatchesError
        If a data file cannot be located uniquely.

    Notes
    -----
    The reference SNR and selectivity are computed on the
    raw spectra with the same `Policy` settings as the pipelines. Rows of
    the two CSVs are paired by position, not by ``'Sample'`` label.
    Failed combinations are dropped. Progress and timing messages are
    printed to standard output.
    """

    warnings.filterwarnings("ignore")

    t0 = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    eval_config = cfg["evaluation_policy"]
    baseline_algos = cfg["baseline"]
    scattering_algos = cfg["scattering"]
    filtering_algos = cfg["filtering"]

    metrics_name = eval_config["metrics"]

    # --- Expand search space
    expanded_baselines = u.expand_param_grid(baseline_algos)
    expanded_scatters = u.expand_param_grid(scattering_algos)
    expanded_filters = u.expand_param_grid(filtering_algos)

    all_combos: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = list(
        itertools.product(expanded_baselines, expanded_scatters, expanded_filters)
    )

    print(f"\n### Total pipeline combinations to evaluate: {len(all_combos)} ###\n")

    # --- Load data
    df = pd.read_csv(
        u.get_data_path(x_data), index_col="Sample"
    )
    wn = df.columns.values.astype(float)
    X = df.values.astype(float)

    y = pd.read_csv(
        u.get_data_path(y_data), index_col="Sample"
    ).values.astype(float)

    # --- Compute snr_raw on raw data (before any preprocessing)
    policy_obj = Policy(eval_config)
    peak_matrix, peak_idx_matrix, tp_idx_matrix, tp_peak_matrix, _ = policy_obj.peak_counter(
        X, wn,
        eval_config['intensity_tolerance'],
        eval_config['reference_peaks'],
        eval_config['wn_tolerance_spectra']
    )
    sel_raw = policy_obj.peak_selectivity(X, y, tp_idx_matrix, eval_config['wn_tolerance_spectra'])
    sel_raw_median = np.nanmedian(sel_raw)
    snr_raw = policy_obj.snr(X, wn, tp_idx_matrix, tp_peak_matrix, window_cm=50)
    del policy_obj

    # --- Parallel execution
    n_jobs = min(multiprocessing.cpu_count(), max(1, len(all_combos)))

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_pipeline)(
            i + 1, base_cfg, scat_cfg, filt_cfg, X, y, wn, snr_raw, sel_raw_median,
            eval_config, metrics_name
        )
        for i, (base_cfg, scat_cfg, filt_cfg) in enumerate(all_combos)
    )

    results: List[Dict[str, Any]] = [r for r in results_list if r is not None]

    if not results:
        print(
            "\n⚠️ No successful results collected. Check preprocessing/evaluation errors above."
        )
        return None

    # --- Build results DataFrame
    results_df = pd.DataFrame(results)

    # Enforce numeric dtypes where possible
    non_numeric_cols = {
        "baseline_config",
        "scattering_config",
        "filtering_config",
    }
    for col in results_df.columns:
        if col not in non_numeric_cols:
            results_df[col] = pd.to_numeric(results_df[col], errors="ignore")

    print(f"\n### ✅ Grid Search Complete — Total Time: {time.time() - t0:.2f} sec ###")

    return results_df


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    with open(u.get_data_path("config.yml"), "rb") as f:
        cfg = yaml.safe_load(f)["pipeline_optimization"]

    X_DATA = cfg['raman_input']
    Y_DATA = cfg['hplc_input']
    DATA_OUTPUT = cfg['file_out']

    results = grid_search(cfg, X_DATA, Y_DATA, DATA_OUTPUT)
    # results.to_csv('test_results_glu.csv')
