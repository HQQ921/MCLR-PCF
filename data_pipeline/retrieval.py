"""
Sequence retrieval with optional three-view candidate filtering.

Default mode keeps the original behavior:
1) build candidates by temporal/background/statistical views (union)
2) rerank candidates by weighted text + sequence similarity.

Use --ts_only to do pure sequence retrieval (DTW similarity) over the whole
historical pool (still respecting time filter unless --disable_time_filter).
"""

import argparse
import ast
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dtaidistance import dtw


class UnionWeightedRetriever:
    def __init__(self, historical_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        self.historical_pool = historical_df.copy()
        self.config = {
            "k": 10,
            "enable_time_filter": True,
            "normalize_ts": False,
            "ts_only": False,
            "w_temporal": 0.5,
            "w_background": 0.2,
            "w_statistical": 0.3,
            "alpha": 0.1,
            "beta": 0.9,
            "background_top_n": 100,
        }
        if config:
            self.config.update(config)

        print("=" * 60)
        print("Initialized retriever")
        print("=" * 60)
        print(f"  Historical pool size: {len(self.historical_pool)}")
        print(f"  Top-K: {self.config['k']}")
        print(f"  Time filter: {self.config['enable_time_filter']}")
        print(f"  Normalize series for DTW: {self.config['normalize_ts']}")
        print(f"  TS-only mode: {self.config['ts_only']}")
        print(f"  Weights (temporal/background/statistical): "
              f"{self.config['w_temporal']:.2f}/{self.config['w_background']:.2f}/{self.config['w_statistical']:.2f}")
        print(f"  Final score alpha/beta: {self.config['alpha']:.2f}/{self.config['beta']:.2f}")
        print("=" * 60)

    @staticmethod
    def _parse_series(x: Any) -> np.ndarray:
        if isinstance(x, str):
            x = ast.literal_eval(x)
        return np.asarray(x, dtype=float)

    @staticmethod
    def normalize_sequence(seq: np.ndarray) -> np.ndarray:
        mean = np.mean(seq)
        std = np.std(seq)
        if std < 1e-8:
            return seq - mean
        return (seq - mean) / std

    def compute_sequence_similarity(self, x1: Any, x2: Any) -> float:
        try:
            seq1 = self._parse_series(x1)
            seq2 = self._parse_series(x2)
            if len(seq1) != len(seq2):
                return 0.0
            if self.config["normalize_ts"]:
                seq1 = self.normalize_sequence(seq1)
                seq2 = self.normalize_sequence(seq2)
            distance = dtw.distance(seq1, seq2)
            return float(1.0 / (1.0 + distance))
        except Exception:
            return 0.0

    def filter_by_time(self, pool: pd.DataFrame, query_sample: pd.Series) -> pd.DataFrame:
        if not self.config["enable_time_filter"]:
            return pool
        query_end = pd.to_datetime(query_sample["window_end"])
        return pool[pd.to_datetime(pool["window_end"]) < query_end]

    @staticmethod
    def compute_jaccard_similarity(set1_str: str, set2_str: str) -> float:
        if pd.isna(set1_str) or set1_str == "":
            return 0.0
        if pd.isna(set2_str) or set2_str == "":
            return 0.0
        set1 = set(str(set1_str).split(","))
        set2 = set(str(set2_str).split(","))
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        return len(set1 & set2) / union

    def retrieve_temporal(self, query_sample: pd.Series, pool: pd.DataFrame) -> Dict[int, float]:
        matched = pool[pool["temporal_pattern"] == query_sample["temporal_pattern"]]
        return {int(sid): 1.0 for sid in matched["sample_id"].values}

    def retrieve_background(self, query_sample: pd.Series, pool: pd.DataFrame) -> Dict[int, float]:
        query_patterns = query_sample.get("background_patterns", "")
        if pd.isna(query_patterns) or query_patterns == "":
            return {}
        sims = []
        for _, row in pool.iterrows():
            sim = self.compute_jaccard_similarity(query_patterns, row.get("background_patterns", ""))
            if sim > 0:
                sims.append({"sample_id": int(row["sample_id"]), "similarity": float(sim)})
        sims.sort(key=lambda x: x["similarity"], reverse=True)
        sims = sims[: int(self.config["background_top_n"])]
        return {x["sample_id"]: x["similarity"] for x in sims}

    def retrieve_statistical(self, query_sample: pd.Series, pool: pd.DataFrame) -> Dict[int, float]:
        matched = pool[pool["statistical_pattern"] == query_sample["statistical_pattern"]]
        return {int(sid): 1.0 for sid in matched["sample_id"].values}

    @staticmethod
    def get_matched_perspectives(sample_id: int, temp_dict: Dict[int, float], back_dict: Dict[int, float],
                                 stat_dict: Dict[int, float]) -> List[int]:
        out = []
        if sample_id in temp_dict:
            out.append(1)
        if sample_id in back_dict:
            out.append(2)
        if sample_id in stat_dict:
            out.append(3)
        return out

    def retrieve(self, query_sample: pd.Series) -> Dict[str, Any]:
        start_time = time.time()
        query_id = int(query_sample["sample_id"])

        pool = self.filter_by_time(self.historical_pool, query_sample)
        pool = pool[pool["sample_id"] != query_id]
        if len(pool) == 0:
            return self._empty_result(query_sample, 0)

        ts_only = bool(self.config.get("ts_only", False))
        if ts_only:
            temp_dict: Dict[int, float] = {}
            back_dict: Dict[int, float] = {}
            stat_dict: Dict[int, float] = {}
            candidate_df = pool
            union_size = len(candidate_df)
        else:
            temp_dict = self.retrieve_temporal(query_sample, pool)
            back_dict = self.retrieve_background(query_sample, pool)
            stat_dict = self.retrieve_statistical(query_sample, pool)
            all_candidate_ids = set(temp_dict.keys()) | set(back_dict.keys()) | set(stat_dict.keys())
            if len(all_candidate_ids) == 0:
                return self._empty_result(query_sample, len(pool), len(temp_dict), len(back_dict), len(stat_dict))
            candidate_df = pool[pool["sample_id"].isin(all_candidate_ids)]
            union_size = len(all_candidate_ids)

        w_temp = float(self.config["w_temporal"])
        w_back = float(self.config["w_background"])
        w_stat = float(self.config["w_statistical"])
        alpha = float(self.config["alpha"])
        beta = float(self.config["beta"])

        candidates = []
        for _, row in candidate_df.iterrows():
            sid = int(row["sample_id"])
            seq_score = self.compute_sequence_similarity(query_sample["x_ot"], row["x_ot"])

            if ts_only:
                temp_match = 0.0
                back_match = 0.0
                stat_match = 0.0
                text_score = 0.0
                final_score = seq_score
                matched_perspectives: List[int] = []
            else:
                temp_match = temp_dict.get(sid, 0.0)
                back_match = back_dict.get(sid, 0.0)
                stat_match = stat_dict.get(sid, 0.0)
                text_score = w_temp * temp_match + w_back * back_match + w_stat * stat_match
                final_score = alpha * text_score + beta * seq_score
                matched_perspectives = self.get_matched_perspectives(sid, temp_dict, back_dict, stat_dict)

            candidates.append(
                {
                    "sample_id": sid,
                    "text_score": float(text_score),
                    "sequence_similarity": float(seq_score),
                    "final_score": float(final_score),
                    "matched_perspectives": matched_perspectives,
                    "temporal_match": float(temp_match),
                    "background_match": float(back_match),
                    "statistical_match": float(stat_match),
                }
            )

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_k = candidates[: int(self.config["k"])]
        for i, item in enumerate(top_k, start=1):
            item["rank"] = i

        return {
            "query_info": {
                "window_start": query_sample["window_start"],
                "window_end": query_sample["window_end"],
                "temporal_pattern": query_sample.get("temporal_pattern", ""),
                "statistical_pattern": query_sample.get("statistical_pattern", ""),
                "background_patterns": query_sample.get("background_patterns", ""),
            },
            "retrieved_samples": top_k,
            "metadata": {
                "filtered_pool_size": int(len(pool)),
                "temporal_candidates": int(len(temp_dict)),
                "statistical_candidates": int(len(stat_dict)),
                "background_candidates": int(len(back_dict)),
                "union_size": int(union_size),
                "n_retrieved": int(len(top_k)),
                "retrieval_time": float(time.time() - start_time),
            },
        }

    @staticmethod
    def _empty_result(query_sample: pd.Series, pool_size: int = 0, temp_size: int = 0,
                      back_size: int = 0, stat_size: int = 0) -> Dict[str, Any]:
        return {
            "query_info": {
                "window_start": query_sample["window_start"],
                "window_end": query_sample["window_end"],
                "temporal_pattern": query_sample.get("temporal_pattern", ""),
                "statistical_pattern": query_sample.get("statistical_pattern", ""),
                "background_patterns": query_sample.get("background_patterns", ""),
            },
            "retrieved_samples": [],
            "metadata": {
                "filtered_pool_size": int(pool_size),
                "temporal_candidates": int(temp_size),
                "statistical_candidates": int(stat_size),
                "background_candidates": int(back_size),
                "union_size": 0,
                "n_retrieved": 0,
                "retrieval_time": 0.0,
            },
        }

    def retrieve_batch(self, query_df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[int, Any]:
        results: Dict[int, Any] = {}
        print(f"\n{'=' * 60}")
        print(f"Batch retrieval: {len(query_df)} samples")
        print(f"{'=' * 60}\n")

        for idx, (_, query_sample) in enumerate(query_df.iterrows(), start=1):
            qid = int(query_sample["sample_id"])
            results[qid] = self.retrieve(query_sample)
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{len(query_df)}")

        print(f"\n{'=' * 60}")
        print("Retrieval done")
        print(f"{'=' * 60}")

        self._print_statistics(results)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved: {save_path}")

        return results

    @staticmethod
    def _print_statistics(results: Dict[int, Any]) -> None:
        n_total = len(results)
        if n_total == 0:
            print("No retrieval results.")
            return
        n_empty = sum(1 for r in results.values() if r["metadata"]["n_retrieved"] == 0)
        n_ret = [r["metadata"]["n_retrieved"] for r in results.values()]
        union_sizes = [r["metadata"]["union_size"] for r in results.values()]

        print("\nRetrieval stats")
        print(f"  Total queries: {n_total}")
        print(f"  Empty retrieval: {n_empty} ({n_empty / n_total * 100:.1f}%)")
        print(f"  Avg retrieved: {np.mean(n_ret):.2f}")
        print(f"  Avg union size: {np.mean(union_sizes):.2f}")


def save_statistics(results: Dict[int, Any], output_path: str) -> None:
    rows = []
    for qid, result in results.items():
        ret = result.get("retrieved_samples", [])
        rows.append(
            {
                "query_sample_id": int(qid),
                "n_retrieved": int(result["metadata"].get("n_retrieved", 0)),
                "union_size": int(result["metadata"].get("union_size", 0)),
                "temporal_candidates": int(result["metadata"].get("temporal_candidates", 0)),
                "background_candidates": int(result["metadata"].get("background_candidates", 0)),
                "statistical_candidates": int(result["metadata"].get("statistical_candidates", 0)),
                "avg_final_score": float(np.mean([x["final_score"] for x in ret])) if ret else 0.0,
                "avg_text_score": float(np.mean([x["text_score"] for x in ret])) if ret else 0.0,
                "avg_sequence_similarity": float(np.mean([x["sequence_similarity"] for x in ret])) if ret else 0.0,
                "retrieval_time": float(result["metadata"].get("retrieval_time", 0.0)),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved statistics: {output_path}")


def main(args: argparse.Namespace) -> None:
    print("Loading data...")
    df = pd.read_csv(args.data_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print("Dataset overview")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")

    config = {
        "k": args.k,
        "enable_time_filter": (not args.disable_time_filter),
        "normalize_ts": args.normalize_ts,
        "ts_only": args.ts_only,
        "w_temporal": args.w_temporal,
        "w_background": args.w_background,
        "w_statistical": args.w_statistical,
        "alpha": args.alpha,
        "beta": args.beta,
        "background_top_n": args.background_top_n,
    }

    retriever = UnionWeightedRetriever(train_df, config)

    if args.test_single and len(test_df) > 0:
        sample = test_df.iloc[0]
        result = retriever.retrieve(sample)
        print("\nSingle-sample debug")
        print(f"  Query sample_id: {sample['sample_id']}")
        print(f"  Candidates in union: {result['metadata']['union_size']}")
        print(f"  Retrieved: {result['metadata']['n_retrieved']}")
        for item in result["retrieved_samples"][:5]:
            print(
                f"    rank={item['rank']} sid={item['sample_id']} final={item['final_score']:.4f} "
                f"seq={item['sequence_similarity']:.4f} text={item['text_score']:.4f}"
            )

    if args.retrieve_test:
        out = os.path.join(args.output_dir, "retrieval_results_test_union.pkl")
        stats = os.path.join(args.output_dir, "retrieval_statistics_test_union.csv")
        results = retriever.retrieve_batch(test_df, save_path=out)
        save_statistics(results, stats)

    if args.retrieve_train:
        out = os.path.join(args.output_dir, "retrieval_results_train_union.pkl")
        stats = os.path.join(args.output_dir, "retrieval_statistics_train_union.csv")
        results = retriever.retrieve_batch(train_df, save_path=out)
        save_statistics(results, stats)

    if args.retrieve_val and len(val_df) > 0:
        out = os.path.join(args.output_dir, "retrieval_results_val_union.pkl")
        stats = os.path.join(args.output_dir, "retrieval_statistics_val_union.csv")
        results = retriever.retrieve_batch(val_df, save_path=out)
        save_statistics(results, stats)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval with optional TS-only mode")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--normalize_ts", action="store_true")
    parser.add_argument("--ts_only", action="store_true",
                        help="Pure time-series retrieval; ignore three-view candidate union")
    parser.add_argument("--disable_time_filter", action="store_true",
                        help="Allow retrieving from all historical rows regardless of time")

    parser.add_argument("--w_temporal", type=float, default=0.5)
    parser.add_argument("--w_background", type=float, default=0.2)
    parser.add_argument("--w_statistical", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--background_top_n", type=int, default=100)

    parser.add_argument("--test_single", action="store_true")
    parser.add_argument("--retrieve_test", action="store_true")
    parser.add_argument("--retrieve_train", action="store_true")
    parser.add_argument("--retrieve_val", action="store_true")

    args = parser.parse_args()
    if not (args.retrieve_test or args.retrieve_train or args.retrieve_val):
        args.retrieve_test = True
        args.retrieve_train = True
        args.retrieve_val = True

    main(args)
