"""
Convert retrieval_results_*.pkl into per_sample_neighbors.pkl/csv.

Example:
python scripts/convert_retrieval_results.py --data_path ./data/processed/Health_US/health_us_generated_all.csv --train_results ./data/processed/Health_US/retrieval/retrieval_results_train_union.pkl --val_results ./data/processed/Health_US/retrieval/retrieval_results_val_union.pkl --test_results ./data/processed/Health_US/retrieval/retrieval_results_test_union.pkl --output_path ./data/processed/Health_US/pattern_library/per_sample_neighbors.pkl
"""
import argparse
import os
import pickle
from typing import Dict, Any, List

import pandas as pd


def load_results(path: str) -> Dict[int, Any]:
    obj = pd.read_pickle(path)
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported results format in {path}: {type(obj)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert retrieval_results to per_sample_neighbors format")
    parser.add_argument("--data_path", type=str,
                        default="./data/processed/Traffic/generated/traffic_generated_all.csv")
    parser.add_argument("--train_results", type=str,
                        default="./data/processed/Traffic/retrieval_results/retrieval_results_train.pkl")
    parser.add_argument("--val_results", type=str,
                        default="./data/processed/Traffic/retrieval_results/retrieval_results_val.pkl")
    parser.add_argument("--test_results", type=str,
                        default="./data/processed/Traffic/retrieval_results/retrieval_results_test.pkl")
    parser.add_argument("--output_path", type=str, default="./pattern_library/per_sample_neighbors.pkl")
    return parser.parse_args()


def build_neighbors(records: Dict[int, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
    id_to_row = {int(r.sample_id): i for i, r in df.iterrows()} if "sample_id" in df.columns else {}
    out = []
    for sample_id, payload in records.items():
        try:
            sid = int(sample_id)
        except Exception:
            continue
        row_idx = id_to_row.get(sid, None)
        window_start = df.iloc[row_idx]["window_start"] if row_idx is not None else None
        window_end = df.iloc[row_idx]["window_end"] if row_idx is not None else None

        retrieved = payload.get("retrieved_samples", [])
        neighbors = []
        for item in sorted(retrieved, key=lambda x: x.get("rank", 0)):
            nid = int(item.get("sample_id"))
            nrow = id_to_row.get(nid, None)
            neighbors.append({
                "neighbor_id": nid,
                "neighbor_cluster": -1,
                "distance": 1.0 - float(item.get("sequence_similarity", 0.0)),
                "window_end_nei": df.iloc[nrow]["window_end"] if nrow is not None else None,
                "sequence_similarity": float(item.get("sequence_similarity", 0.0)),
                "rank": int(item.get("rank", 0)),
                "matched_perspectives": item.get("matched_perspectives", []),
            })

        out.append({
            "sample_id": sid,
            "cluster_id": -1,
            "window_start": window_start,
            "window_end": window_end,
            "neighbors": neighbors,
            "effective_k": len(neighbors),
            "note": ""
        })

    return out


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path, parse_dates=["window_start", "window_end"])

    all_records: Dict[int, Any] = {}
    for p in [args.train_results, args.val_results, args.test_results]:
        if p and os.path.exists(p):
            all_records.update(load_results(p))
        else:
            print(f"Warning: results file not found: {p}")

    rows = build_neighbors(all_records, df)
    out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_df.to_pickle(args.output_path)
    csv_path = os.path.splitext(args.output_path)[0] + ".csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved per-sample neighbors to: {args.output_path}")
    print(f"Saved csv preview to: {csv_path}")


if __name__ == "__main__":
    main()
