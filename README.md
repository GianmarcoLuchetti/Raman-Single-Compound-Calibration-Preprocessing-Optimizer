# Balancing-prediction-performance-and-spectral-interpretability

## Description

The following repository provides the code from the publication: 'Balancing prediction performance and spectral interpretability: Multi-metric 
guided design of Raman spectrum preprocessing pipelines to transfer single compound models to bioprocess data' (**ADD DOI**) [1]. 
The code is used to design data preprocessing pipelines for the rapid calibration with single-compound Raman spectra [2],
using evaluation metrics measured after testing a specific pipeline. The tool provides an automated workflow for screening pipelines and
selecting the best based on a multi-criteria evaluation policy.


## Installation

The following software can be installed via GitHub executing the following code:

```git clone [repository url]```

After the installation run on your terminal:

```cd /Balancing-prediction-performance-and-spectral-interpretability```

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

## Copyright

Technische Universiteit Delft hereby disclaims all copyright interest in the program “Raman Preprocessing Pipelines Optimizator” (used to design data preprocessing pipelines via a data driven approach) written by the Author(s).
Gianmarco Luchetti Sfondalmondo, Dean of Faculty of Applied Science

© 2026 Gianmarco Luchetti Sfondalmondo, Delft, The Netherlands.

## License

This software is distributed under a MIT license.
