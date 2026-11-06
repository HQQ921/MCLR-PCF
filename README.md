# MCLR-PCF: Multi-View Common-Library Retrieval for Personalized-Common Collaborative Time Series

This repository contains the implementation for the paper: *Multi-View Common-Library Retrieval for Personalized-Common Collaborative Time Series*.

## Overview

This framework combines time series forecasting with Retrieval-Augmented Generation (RAG) and multi-view fusion. It retrieves similar historical patterns from a common library and fuses them with the target time series through multi-view attention and gated fusion.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data & Models

Large files (datasets and pre-trained model) are not included in this repository. Please download from the link below and place them at the project root:

| Content | Link |
|---------|------|
| Dataset | https://pan.quark.cn/s/1f6c6cc1ba64 |
| Pre-trained model (`all-mpnet-base-v2`) | https://pan.quark.cn/s/d51246d8730f |

Expected structure:

```
data/
├── processed/
│   ├── Agriculture/
│   │   ├── agriculture_generated_all.csv
│   │   ├── agriculture_generated_all_pred8.csv
│   │   ├── agriculture_generated_all_pred10.csv
│   │   ├── agriculture_generated_all_pred12.csv
│   │   └── pattern_library/
│   │       └── per_sample_neighbors.csv
│   ├── Climate/
│   │   ├── climate_generated_all.csv
│   │   ├── climate_generated_all_pred8.csv
│   │   ├── climate_generated_all_pred10.csv
│   │   ├── climate_generated_all_pred12.csv
│   │   └── pattern_library/
│   │       └── per_sample_neighbors.pkl
│   ├── Economy/
│   │   ├── economy_generated_all.csv
│   │   ├── economy_generated_all_pred8.csv
│   │   ├── economy_generated_all_pred10.csv
│   │   ├── economy_generated_all_pred12.csv
│   │   └── pattern_library/
│   │       └── per_sample_neighbors.csv
│   ├── Energy/
│   │   ├── energy_generated_all.csv
│   │   ├── energy_generated_all_pred24.csv
│   │   ├── energy_generated_all_pred36.csv
│   │   ├── energy_generated_all_pred48.csv
│   │   └── pattern_library/
│   │       └── per_sample_neighbors.pkl
│   ├── Health_US/
│   │   ├── health_us_generated_all.csv
│   │   ├── health_us_generated_all_pred24.csv
│   │   ├── health_us_generated_all_pred36.csv
│   │   ├── health_us_generated_all_pred48.csv
│   │   └── pattern_library/
│   │       └── per_sample_neighbors.pkl
│   ├── SocialGood/
│   │   ├── socialgood_generated_all.csv
│   │   ├── socialgood_generated_all_pred8.csv
│   │   ├── socialgood_generated_all_pred10.csv
│   │   ├── socialgood_generated_all_pred12.csv
│   │   └── pattern_library/
│   │       └── per_sample_neighbors.pkl
│   └── Traffic/
│       ├── traffic_generated_all.csv
│       ├── traffic_generated_all_8.csv
│       ├── traffic_generated_all_10.csv
│       ├── traffic_generated_all_12.csv
│       └── pattern_library/
│           └── per_sample_neighbors.csv
├── raw/
│   └── Time-MMD/
│       ├── numerical/
│       │   ├── Agriculture/Agriculture.csv
│       │   ├── Climate/Climate.csv
│       │   ├── Economy/Economy.csv
│       │   ├── Energy/Energy.csv
│       │   ├── Health_US/Health_US.csv
│       │   ├── SocialGood/SocialGood.csv
│       │   └── Traffic/Traffic.csv
│       └── textual/
│           ├── Agriculture/ (Agriculture_report.csv, Agriculture_search.csv)
│           ├── Climate/ (Climate_search.csv)
│           ├── Economy/ (Economy_report.csv, Economy_search.csv)
│           ├── Energy/ (Energy_report.csv, Energy_search.csv)
│           ├── Health_US/ (Health_US_report.csv, Health_US_search.csv)
│           ├── SocialGood/ (SocialGood_report.csv, SocialGood_search.csv)
│           └── Traffic/ (Traffic_report.csv, Traffic_search.csv)
```

```
models/
└── all-mpnet-base-v2/
    ├── config.json
    ├── ...
    └── 1_Pooling/
        └── config.json
```

## Run Experiments

Pre-configured experiment scripts are in `scripts/`. Each dataset has one script per prediction length:

| Dataset | Prediction Lengths |
|---------|-------------------|
| Traffic | 6, 8, 10, 12 |
| Agriculture | 6, 8, 10, 12 |
| Climate | 6, 8, 10, 12 |
| Economy | 6, 8, 10, 12 |
| Energy | 12, 24, 36, 48 |
| Health_US | 12, 24, 36, 48 |
| SocialGood | 6, 8, 10, 12 |

Run a script directly:

```bash
bash scripts/run_traffic_pred6.sh
```

Checkpoints are saved to `./checkpoints/forecast_rag/{dataset}_pred{len}/`.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `exp/` | Experiment classes (core: `exp_forecast_rag_perspective.py`) |
| `data_provider/` | Data loading |
| `data_pipeline/` | Data generation and retrieval |
| `models/` | Time series model architectures (TimesNet, etc.) |
| `layers/` | Neural network layer modules |
| `utils/` | Utility functions |
| `scripts/` | Experiment scripts and preprocessing utilities |
| `checkpoints/` | Saved model checkpoints |

## Pre-processing

### Generate samples

```bash
python data_pipeline/build_samples_with_material.py \
  --output_dir ./data/processed/Health_US --dataset_name health \
  --enable_background_fact_filter true --enable_background_patterns true \
  --enable_context_text true
```

### Convert retrieval results to neighbor index

```bash
python scripts/convert_retrieval_results.py \
  --data_path ./data/processed/SocialGood/socialgood_generated_all.csv \
  --train_results ./data/processed/SocialGood/retrieval/retrieval_results_train_union.pkl \
  --val_results ./data/processed/SocialGood/retrieval/retrieval_results_val_union.pkl \
  --test_results ./data/processed/SocialGood/retrieval/retrieval_results_test_union.pkl \
  --output_path ./data/processed/SocialGood/pattern_library/per_sample_neighbors.pkl
```

## Citation

If you use this code, please cite the paper.
