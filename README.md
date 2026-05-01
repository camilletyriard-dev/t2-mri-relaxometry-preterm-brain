# T2 MRI Relaxometry of the Preterm-Born Adolescent Brain

---

## Overview

This repository implements a complete computational pipeline for **MRI T2 relaxometry** applied to the developing brain, covering:

* Quality control and diagnostic imaging checks on multi-echo T2 data
* **Mono-exponential fitting** via analytical, log-linear, weighted least-squares, and non-linear least-squares methods
* **Two-compartment (bi-exponential) fitting** with fixed and free T2 values, initialisation sensitivity analysis, and parametric bootstrap confidence intervals
* **Model comparison** between one- and two-compartment models using AICc and Akaike weights, with uncertainty quantification via MCMC (Metropolis-Hastings)
* **Multi-compartment analysis** via NNLS T2 spectra and a three-compartment fixed-T2 model (Dingwall et al., 2016)
* **Segmentation-guided MAP priors** that regularise voxel-wise myelin water fraction estimates using tissue-specific Gaussian penalties
* **Inter-subject variation** in the EPICure cohort, including group comparisons (preterm vs term), sex-stratified analysis, regional white matter analysis with Bonferroni correction, and correlation with cognitive outcomes (FSIQ)

All analyses are reproducible from the EPICure preterm-born adolescent cohort data. The codebase is written in Python and is structured for clarity, modularity, and scientific correctness.

## Accompanying Report

This repository is accompanied by a scientific report describing the modelling framework, fitting procedures, uncertainty quantification, and cohort-level findings.

**Report:** [`docs/report.pdf`](docs/report.pdf)

---

## Scientific Background

T2 relaxation describes how quickly a population of nuclear spins dephases after excitation in an MRI experiment. The signal decay measured at echo time $t$ is:

$$S(t) = S_0 \cdot \exp(-t / T_2)$$

where $S_0$ is the equilibrium signal (proton density weighting) and $T_2$ is the spin-spin relaxation time constant, which varies across tissue types.

### Multi-Compartment Extension

A single imaging voxel may contain multiple tissue types (myelin water, intra/extracellular water, CSF) with distinct T2 values. The signal from a voxel containing $C$ non-interacting water pools is:

$$S(t) = S_0 \sum_{i=1}^{C} v_i \exp(-t / T_{2,i}), \qquad \sum_{i=1}^{C} v_i = 1$$

where $v_i$ is the volume fraction and $T_{2,i}$ the relaxation time of the $i$-th compartment.

### Models Implemented

| Model | Free Parameters | Fixed Values | Description |
| --- | --- | --- | --- |
| Mono-exponential | $S_0, T_2$ | — | Single-compartment baseline |
| Bi-exponential (fixed $T_2$) | $S_0, v$ | $T_{2,\text{short}} = 20$ ms, $T_{2,\text{long}} = 80$ ms | Myelin water fraction estimator |
| Bi-exponential (free $T_2$) | $S_0, v, T_{2,\text{short}}, T_{2,\text{long}}$ | — | Unconstrained two-compartment |
| Three-compartment (Dingwall) | $S_0, v_1, v_2$ | $T_2 \in \{20, 80, 2000\}$ ms | Myelin + IE water + CSF |
| NNLS spectrum | amplitudes on 101-point grid | $T_2$ grid: 10–2000 ms, log-spaced | Non-parametric distribution |

### Myelin Water Fraction

The myelin water fraction (MWF), defined as the signal fraction with $T_2 < 40{-}50$ ms, is a non-invasive proxy for myelin content. In normal white matter, MWF is typically 8–16% (Whittall et al., 1997; MacKay et al., 2006). The fixed-$T_2$ bi-exponential model identifies MWF directly as $v$, the short-$T_2$ volume fraction.

### Model Comparison

Models are compared using the corrected Akaike Information Criterion:

$$\text{AICc} = 2N + K \log(\text{SSR}/K) + \frac{2N(N+1)}{K - N - 1}$$

where $N$ is the number of free parameters (including $\sigma$), $K$ is the number of echoes, and SSR is the sum of squared residuals. Akaike weights convert AICc values into relative model probabilities.

### MAP Estimation with Segmentation Priors

Voxel-wise MWF estimation is regularised using a Gaussian prior $\mathcal{N}(\mu_v, \sigma_v^2)$ on the volume fraction $v$, where $\mu_v$ and $\sigma_v$ are derived per-voxel from the soft tissue segmentation and cohort-level compartment statistics. The MAP objective appends a prior residual to the data residuals:

$$\hat{\theta}_{\text{MAP}} = \arg\min_\theta \left[ \sum_k \left(\frac{S_k - \hat{S}_k(\theta)}{\sigma_n}\right)^2 + \left(\frac{v - \mu_v}{\sigma_v}\right)^2 \right]$$

### Uncertainty Quantification

Parameter precision is assessed by two methods:

* **Parametric bootstrap**: generate $B$ synthetic datasets from the fitted model with Gaussian noise at the estimated $\sigma$, refit each, and report the 2.5th–97.5th percentile interval.
* **MCMC (Metropolis-Hastings)**: sample the posterior $p(v \mid S) \propto \exp(-\chi^2/2\sigma^2) \cdot \text{Beta}(v; \alpha, \beta)$ with $S_0$ analytically profiled out at each proposal. Burn-in and thinning yield the posterior distribution of $v$.

---

## Results Summary

### Task 2 — Mono-Exponential T2 Estimates (Preterm Cohort, Tissue Means)

| Tissue | T2 OLS (ms) | T2 WLS (ms) | T2 NLLS (ms) |
| --- | --- | --- | --- |
| CSF | ~700–1200 | ~700–1200 | ~700–1200 |
| GM | ~65–80 | ~65–80 | ~65–80 |
| WM | ~55–70 | ~55–70 | ~55–70 |

All three multi-echo methods agree closely; the systematic curvature in WM log-residuals confirms multi-exponentiality.

### Task 4 — AICc Model Comparison (WM, Preterm Cohort)

| Model | N (incl. σ) | Cohort Akaike Weight |
| --- | --- | --- |
| Mono-exponential | 3 | ~0 |
| Bi-exponential (fixed T2) | 3 | ~1.0 |

The two-compartment model is preferred in virtually all subjects for WM tissue-mean curves.

### Task 7 — Preterm vs Term Group Comparison

Whole-WM MWF shows no significant difference between EP and term-born controls (consistent with Dingwall et al., 2016). Regional analysis (corpus callosum, internal capsule) reveals region-specific trends, consistent with Laureano et al. (2022) and Thalhammer et al. (2025).

---

## Repository Structure

```
t2-mri-relaxometry-preterm-brain/
│
├── README.md
├── CITATION.cff
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── README.md                        # Data access instructions
│   ├── TEs.txt                          # Shared adolescent echo-time schedule
│   ├── demo.xlsx                        # Participant demographics (subset)
│   ├── Epicure_psychological.xlsx       # Full cohort table (EP/FT, neuropsych)
│   └── nm_labels.xlsx                   # White-matter region label definitions
│
├── docs/
│   └── report.pdf                       # Accompanying scientific report
│
├── src/
│   ├── models.py                        # Signal models, fitters, bootstrap,
│   │                                    #   MCMC, AICc, NNLS, MAP estimation
│   ├── utils.py                         # NIfTI/TE loaders, tissue helpers,
│   │                                    #   demographics, cohort loader
│   ├── quality.py                       # Monotonicity, SNR, log-fit scoring,
│   │                                    #   partial volume, cohort QA table
│   ├── analysis.py                      # Cohort fitting loops, regional analysis,
│   │                                    #   MAP priors, group statistics
│   └── plotting.py                      # Every figure as a callable function
│
└── notebooks/
    ├── 01_imaging_factors.ipynb          # QA: monotonicity, SNR, partial volume
    ├── 02_one_compartment.ipynb          # Mono-exp fitting: analytical → NLLS
    ├── 03_two_compartment.ipynb          # Bi-exp fitting, bootstrap CIs
    ├── 04_model_comparison.ipynb         # AICc, Akaike weights, MCMC vs bootstrap
    ├── 05_multicompartment.ipynb         # NNLS spectra, 3-compartment, AICc
    ├── 06_priors.ipynb                   # MAP priors, spatial smoothing
    └── 07_inter_subject.ipynb            # EP vs FT: whole-WM, regional, IQ
```

---

## Getting Started

### Requirements

* Python 3.10 or later
* See [`requirements.txt`](requirements.txt) for package dependencies

### Data

The imaging data are from the EPICure preterm cohort and UCL adult-control dataset. See [`data/README.md`](data/README.md) for the expected file layout and access instructions.

### Running the Pipeline

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run notebooks in order: `01_imaging_factors.ipynb` through `07_inter_subject.ipynb`. Notebook 01 produces `preterm_ids.csv` and `preterm_exclusions.csv` in `data/`, which are consumed by all subsequent notebooks.