# Data Pipeline

This folder contains standalone, task-oriented data processing scripts.

Current scripts:

- `build_samples_with_material.py`
  - Builds samples with fields:
    `split, sample_id, date, window_start, window_end, x_ot, y_ot, x_ot_ori, y_ot_ori, llm_material, context_text, temporal_pattern, statistical_pattern, background_patterns, perspectives`
  - Uses:
    - numerical data from `--numerical_csv`
    - textual facts from `--textual_csv`
  - Optional:
    - `--min_date YYYY-MM-DD` to keep rows with `date >= min_date` before split/windowing.
    - `--scale true/false` to control whether `x_ot/y_ot` are z-scored by train split stats.
      `x_ot_ori/y_ot_ori` always keep original values.
    - `--dataset_name` / `--granularity` / `--temporal_date_col` to control `temporal_pattern`.
    - `statistical_pattern` is generated from `x_ot` using train-split percentile thresholds:
      trend (slope), volatility (std), and seasonality (weekly period=4, daily period=7).
    - `--enable_background_patterns true/false` to enable LLM-based background classification from `llm_material`.
      It writes:
      - `background_patterns`: sample-level union labels, like `1,4`
      - `perspectives`: per-fact JSON details (`fact`, `labels`, `label_names`)
    - `--enable_background_fact_filter true/false` to pre-filter textual facts before building `llm_material`.
      - Rules are loaded from `--background_filters_json` under dataset key (e.g., `health`).
      - This filtering step runs before LLM background classification and before context generation.
    - `--background_categories_json` must define categories for each dataset explicitly (no default fallback).
    - `--enable_context_text true/false` to enable LLM context generation.
      It builds prompt input from:
      - `x_ot_ori` (original sequence values)
      - `llm_material` truncated to last `N` facts (`--context_max_facts`, default `20`)
      Prompt templates are loaded from `--context_prompts_json` per dataset (must be explicitly defined).
    - Pattern frequencies are printed, and can be saved to
      `<base_name>_pattern_frequency.json` (`--save_pattern_frequency true/false`).
      The report includes:
      - `temporal_pattern` frequency
      - `statistical_pattern` frequency
      - `background_patterns` frequency
      - `background_label_frequency` (1..N categories)

Example:

```powershell
python data_pipeline/build_samples_with_material.py `
  --numerical_csv ./data/raw/Time-MMD/numerical/Energy/Energy.csv `
  --textual_csv ./data/raw/Time-MMD/textual/Energy/Energy_search.csv `
  --output_dir ./data/processed/Energy/generated `
  --seq_len 8 `
  --pred_len 6
```

Context generation example:

```powershell
python data_pipeline/build_samples_with_material.py `
  --numerical_csv ./data/raw/Time-MMD/numerical/Energy/Energy.csv `
  --textual_csv ./data/raw/Time-MMD/textual/Energy/Energy_search.csv `
  --output_dir ./data/processed/Energy `
  --base_name energy_generated_all `
  --min_date 1996-01-01 `
  --enable_context_text true `
  --context_prompts_json ./data_pipeline/context_prompts.json `
  --context_max_facts 20
```
