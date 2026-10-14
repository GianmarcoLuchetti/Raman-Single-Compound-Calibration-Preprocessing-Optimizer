import warnings
import utils as u
import yaml
from grid_search import grid_search

from pathlib import Path  # NEW

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    with open(u.get_data_path("config.yml"), 'rb') as f:
        cfg = yaml.safe_load(f)

    cfg_screening = cfg['pipeline_screening']
    X_data = cfg_screening['raman_input']
    y_data = cfg_screening['hplc_input']
    results_dir = cfg_screening['results_dir']
    save = cfg_screening.get('save', False)  # NEW: defaults to False if missing

    screening_output = grid_search(cfg_screening, X_data, y_data)

    cfg_selection = cfg['pipeline_selection']
    hf = cfg_selection['hard_filters']
    des_bounds = cfg_selection['bounds']

    filtered, best_pipeline = u.selection(screening_output, hf, des_bounds)

    print(f'\nFiltered pipeline: {len(filtered)} / {len(screening_output)}')

    print(f'Selected scattering correction: {best_pipeline["scattering_config"].item()}')
    print(f'Selected baseline correction: {best_pipeline["baseline_config"].item()}')
    print(f'Selected noise filtering: {best_pipeline["filtering_config"].item()}\n')

    print(f"Measured metrics:\n"
          f"    Median F1 Score Spectra: {best_pipeline['Median_F1_TP'].item()}\n"
          f"    Median Selectivity: {best_pipeline['Median_selectivity'].item()}\n"
          f"    Median FWHM (cm\u207b\u00b9): {best_pipeline['Median_FWHM'].item()}\n"
          f"    Median Voigt R\u00b2: {best_pipeline['Median_Voigt'].item()}\n"
          f"    Median Baseline MAD: {best_pipeline['Median_mad_baseline'].item()}\n"
          f"    Median SNR Improvement: {best_pipeline['Snr_improvement'].item()}\n"
          f"    rRMSECV Worst Quartile (%): {best_pipeline['RMSE_pct'].item()}\n"
          f"    Median F1 Score Regression Coefficient: {best_pipeline['Median_F1_coef'].item()}\n"
          )

    print(f'Desirability metrics:\n'
          f'    Desirability Median F1 Score Spectra:{best_pipeline["Des_F1_TP_spectra"].item()}\n'
          f'    Desirability Median FWHM:{best_pipeline["Des_fwhm"].item()}\n'
          f'    Desirability Median Baseline MAD:{best_pipeline["Des_mad"].item()}\n'
          f'    Desirability Median F1 Score Regression Coefficient:{best_pipeline["Des_F1_TP_coef"].item()}\n')

    print(f'Geometric Mean: {best_pipeline["Geom_mean"].item()}\n')

    if save:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def val(col):
            return best_pipeline[col].item()

        best_summary = {
            'n_pipelines_screened': len(screening_output),
            'n_pipelines_filtered': len(filtered),
            'pipeline': {
                'scattering_correction': val('scattering_config'),
                'baseline_correction': val('baseline_config'),
                'noise_filtering': val('filtering_config'),
            },
            'measured_metrics': {
                'median_f1_score_spectra': val('Median_F1_TP'),
                'median_selectivity': val('Median_selectivity'),
                'median_fwhm_cm-1': val('Median_FWHM'),
                'median_voigt_r2': val('Median_Voigt'),
                'median_baseline_mad': val('Median_mad_baseline'),
                'median_snr_improvement': val('Snr_improvement'),
                'rrmsecv_worst_quartile_pct': val('RMSE_pct'),
                'median_f1_score_regression_coefficient': val('Median_F1_coef'),
            },
            'desirability': {
                'f1_score_spectra': val('Des_F1_TP_spectra'),
                'fwhm': val('Des_fwhm'),
                'baseline_mad': val('Des_mad'),  # check column name
                'f1_score_regression_coefficient': val('Des_F1_TP_coef'),
            },
            'geometric_mean': val('Geom_mean'),
        }

        with open(out_dir / 'best_pipeline.yml', 'w') as f:
            yaml.safe_dump(best_summary, f, sort_keys=False, allow_unicode=True)

        screening_output.to_csv(out_dir / 'screening_results.csv', index=False)
        filtered.sort_values('Geom_mean', ascending=False).to_csv(
            out_dir / 'filtered_ranking.csv', index=False
        )

        with open(out_dir / 'config_used.yml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f'Results saved to {out_dir.resolve()}')