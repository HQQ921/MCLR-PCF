"""
Derive a longer forecast horizon dataset from an existing sample CSV without
recomputing temporal/statistical/background/context fields.

Examples:
python data_pipeline/derive_horizon_from_existing_samples.py \
  --input_csv ./data/processed/Energy/energy_generated_all.csv \
  --output_csv ./data/processed/Energy/energy_generated_all_pred24.csv \
  --source_pred_len 12 \
  --target_pred_len 24

python data_pipeline/derive_horizon_from_existing_samples.py \
  --input_csv ./data/processed/Energy/energy_generated_all.csv \
  --output_csv ./data/processed/Energy/energy_generated_all_pred36.csv \
  --source_pred_len 12 \
  --target_pred_len 36
"""

import argparse
import ast
import json
from typing import Any, List

import pandas as pd


def parse_list(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except Exception:
                continue
    return []


def dumps_list(values: List[float]) -> str:
    return json.dumps(values, ensure_ascii=False)


def derive_target_y(y_cache: List[List[float]], i: int, source_pred_len: int, target_pred_len: int) -> List[float]:
    """
    Build target horizon by 1-step rolling extension:
    - keep y_i[0:source_pred_len]
    - append tail value from subsequent rows:
      y_{i+1}[-1], y_{i+2}[-1], ...
    This matches the original sliding-window construction and only drops
    target_pred_len - source_pred_len rows at split tail.
    """
    base = y_cache[i]
    if len(base) != source_pred_len:
        return []

    out = list(base[:target_pred_len])
    if target_pred_len <= source_pred_len:
        return out

    need_extra = target_pred_len - source_pred_len
    for step in range(1, need_extra + 1):
        idx = i + step
        if idx >= len(y_cache):
            return []
        part = y_cache[idx]
        if len(part) != source_pred_len:
            return []
        out.append(float(part[-1]))
    return out[:target_pred_len]


def derive_split(df_split: pd.DataFrame, source_pred_len: int, target_pred_len: int) -> pd.DataFrame:
    rows = []
    n = len(df_split)
    max_offset = target_pred_len - source_pred_len
    keep_n = n - max_offset
    if keep_n <= 0:
        return df_split.iloc[0:0].copy()

    df_split = df_split.reset_index(drop=True).copy()

    has_y_ori = "y_ot_ori" in df_split.columns
    y_cache = [parse_list(v) for v in df_split["y_ot"]]
    y_ori_cache = [parse_list(v) for v in df_split["y_ot_ori"]] if has_y_ori else None

    for i in range(keep_n):
        y_target = derive_target_y(y_cache, i, source_pred_len, target_pred_len)
        if not y_target:
            continue

        out_row = df_split.iloc[i].copy()
        out_row["y_ot"] = dumps_list(y_target)

        if has_y_ori:
            y_ori_target = derive_target_y(y_ori_cache, i, source_pred_len, target_pred_len)
            if not y_ori_target:
                continue
            out_row["y_ot_ori"] = dumps_list(y_ori_target)

        rows.append(out_row)

    if not rows:
        return df_split.iloc[0:0].copy()
    return pd.DataFrame(rows, columns=df_split.columns)


def main():
    parser = argparse.ArgumentParser(
        description="Derive longer horizon labels from an existing generated sample CSV"
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Input sample CSV (e.g. pred_len=12)")
    parser.add_argument("--output_csv", type=str, required=True, help="Output sample CSV (e.g. pred_len=24)")
    parser.add_argument("--source_pred_len", type=int, default=12, help="Current y_ot length in input CSV")
    parser.add_argument("--target_pred_len", type=int, default=24, help="Target y_ot length in output CSV")
    parser.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Split column name (default: split)",
    )
    args = parser.parse_args()

    if args.target_pred_len < args.source_pred_len:
        raise ValueError("target_pred_len must be >= source_pred_len.")

    df = pd.read_csv(args.input_csv)
    if args.split_col not in df.columns:
        raise ValueError(f"Missing split column: {args.split_col}")
    if "y_ot" not in df.columns:
        raise ValueError("Missing required column: y_ot")

    out_splits = []
    for split_name, split_df in df.groupby(args.split_col, sort=False):
        derived = derive_split(
            split_df,
            source_pred_len=args.source_pred_len,
            target_pred_len=args.target_pred_len,
        )
        out_splits.append(derived)
        print(
            f"split={split_name}: input={len(split_df)}, output={len(derived)}, dropped_tail={len(split_df)-len(derived)}"
        )

    out_df = pd.concat(out_splits, axis=0, ignore_index=True) if out_splits else df.iloc[0:0].copy()
    out_df.to_csv(args.output_csv, index=False)

    print(f"Saved: {args.output_csv}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
