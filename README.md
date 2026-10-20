# Raman-Single-Compound-Calibration-Preprocessing-Optimizer

## Description

The following repository provides the code from the publication: "Balancing prediction performance and spectral interpretability: Multi-metric 
guided design of Raman spectrum preprocessing pipelines to transfer single compound models to bioprocess data" (https://doi.org/10.1002/jrs.70223) [1]. 
The code is used to design data preprocessing pipelines for Raman rapid calibration with single-compound spectra [2],
using evaluation metrics measured after testing a specific pipeline. The tool provides an automated workflow for screening pipelines and
selecting the best based on a multi-criteria evaluation policy.

### Repository Structure
```bash
├── README.md
├── requirements.txt
└── src
    ├── config.yml
    ├── evaluation_metrics.py
    ├── grid_search.py
    ├── main.py
    ├── preprocessing.py
    └── utils.py
```


## Installation

The following software can be installed via GitHub executing the following code:

```git clone [repository url]```

After the installation run on your terminal:

```cd Raman-Single-Compound-Calibration-Preprocessing-Optimizer```

### Requirements 

The software to be executed requires the following libraries:
- pandas~=2.2.3
- numpy~=2.2.6
- pathlib~=1.0.1
- scipy~=1.15.3 
- pyyaml~=6.0.2 
- scikit-learn~=1.6.1 
- line-profiler~=5.0.0 
- chemotools~=0.2.5 
- joblib~=1.5.1 
- pybaselines~=1.2.0

To install the required libraries execute:
```pip install -r requirements.txt```



## Usage

The workflow is controlled entirely through `src/config.yml`: no code changes are required to run a new screening.

### 1. Prepare the data

The framework requires two datasets:

- **raman_input**: single-compound Raman spectra (e.g., glucose and ethanol), including a blank (e.g., medium).
- **hplc_input**: reference concentrations of taget analytes.

The spectra file should contain one spectrum per row, with Raman shifts as column headers. The HPLC file should contain one concentration for each spectrum.

| Index | 400.0 | 401.0 | ... | 1800.0 |
|---|---|---|---|---|
|S1| ... | ... | ... | ... |
|S2| ... | ... | ... | ... |
|S3| ... | ... | ... | ... |

| Index | Concentration |
|---|---|
|S1| ... |
|S2| ... |
|S3| ... |

### 2. Configure the run

Edit `src/config.yml` to set the datasets, the preprocessing search space and the evaluation settings:

```yaml
pipeline_screening:
  raman_input: 
  hplc_input: 
  results_dir:
  save: 
  evaluation_policy:
    n_components: 
...
  baseline:
    none:
      alg: ""
    asls:
      alg: "lam={lam}, p={p}, diff_order={diff_order}"
      lam: [<...>]
      p: [<...>]
      diff_order: [<...>]
...
pipeline_selection:
  hard_filters:
    Median_selectivity: [<...>]
...
  bounds:
    Des_F1_TP_spectra: {metric: ,direction: ,lower: ,upper: }
...
```

### 3. Run the screening

From the repository root:

```bash
cd src
python main.py
```

`main.py` performs three steps:

1. **Grid search** (`grid_search.py::grid_searc()`): generates every combination of scattering correction, baseline correction and filtering defined in `config.yml`, and applies each pipeline to the spectra dataset (`preprocessing.py`). Pipelines are evaluated in parallel with `joblib`.
2. **Evaluation** (`evaluation_metrics.py::Policy::evaluate()`): for each pipeline, computes
   - *spectral quality metrics*: SNR, FWHM, pseudo-Voigt fit R², MAD of the baseline, peak F1 score and selectivity;
   - *calibration performance metrics*: rRMSECV from cross-validated PLS on the calibration set and peak F1 score of the regression vector.
3. **Selection** (`utils.py::selection()`): discards pipelines failing the hard filter, then ranks the remainder by the geometric mean of Derringer–Suich desirability functions across all metrics.

### 4. Outputs

If `<save>` is `true`, results are written to `<results_dir>`:

- `screening_results.csv`: every screened pipeline with its spectral quality metrics, calibration performance metrics and desirability scores;
- `filtered_ranking.csv`: the pipelines that pass the hard filters, ranked by geometric mean desirability in descending order;
- `best_pipeline.yml`: the top-ranked pipeline, with its preprocessing configuration (scattering correction, baseline correction, noise filtering), measured metrics, individual desirabilities and geometric mean;
- `config_used.yml`: a copy of the configuration used for the run, so the results can be reproduced.



## References
[1] G. L. Sfondalmondo and M. E. Klijn, “Balancing Prediction Performance and Spectral Interpretability: Multimetric Guided Design of Raman Spectrum Preprocessing Pipelines to Transfer Single Compound Models to Bioprocess Data,” Journal of Raman Spectroscopy (2026): 1–19, https://doi.org/10.1002/jrs.70223.

[2] Klaverdijk, M., L. A. Smulders, M. Ottens, and M. E. Klijn. “Towards Rapid Calibration of Bioprocess Quantification Models Using Single Compound Raman Spectra: A Comparison of Four Approaches.” Biotechnology and Bioengineering (2026): 324–336, https://doi.org/10.1002/bit.70092.



## Copyright

Technische Universiteit Delft hereby disclaims all copyright interest in the program “Raman-Single-Compound-Calibration-Preprocessing-Optimizer” written by the Authors:
Gianmarco Luchetti Sfondalmondo and Marieke Klijn, Faculty of Applied Science.

© 2026 Gianmarco Luchetti Sfondalmondo and Marieke Klijn, Delft, The Netherlands.



## License

This software is distributed under a MIT license.



## Funding
This project is funded by Schmidt Sciences