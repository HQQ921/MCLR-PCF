'''
python data_pipeline/build_samples_with_material.py --output_dir ./data/processed/Health_US --dataset_name health --enable_background_fact_filter true --enable_background_patterns true --enable_context_text true
需要补充3个json文件
'''
import argparse
import ast
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")


def detect_column(columns: List[str], candidates: List[str]) -> str:
    lower_to_original = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    raise ValueError(f"Missing required column. Tried: {candidates}")


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def clean_fact_text(text: str) -> str:
    return " ".join(str(text).split())


GRANULARITY = {
    "agriculture": "monthly",
    "climate": "weekly",
    "economy": "monthly",
    "energy": "weekly",
    "environment": "daily",
    "health": "weekly",
    "socialgood": "monthly",
    "traffic": "monthly",
}


def infer_dataset_name(numerical_csv: str) -> str:
    return os.path.splitext(os.path.basename(numerical_csv))[0].strip().lower()


def generate_temporal_pattern(date_str: str, granularity: str) -> str:
    date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(date):
        return ""

    month = int(date.month)
    if granularity == "monthly":
        return f"M{month:02d}"
    if granularity == "weekly":
        week = (int(date.day) - 1) // 7 + 1
        return f"M{month:02d}_W{week}"
    if granularity == "daily":
        week = (int(date.day) - 1) // 7 + 1
        day_type = "WD" if int(date.weekday()) < 5 else "WE"
        return f"M{month:02d}_W{week}_{day_type}"
    return ""


def load_background_categories(categories_json: str, dataset_name: str) -> dict:
    if not categories_json or not os.path.exists(categories_json):
        raise FileNotFoundError(
            f"Background categories file not found: {categories_json}. "
            "Please define categories for each dataset in this JSON."
        )

    with open(categories_json, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    dataset_key = (dataset_name or "").strip().lower()
    if not isinstance(data, dict):
        raise ValueError("Background categories JSON must be an object.")
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("Background categories JSON must contain object key 'datasets'.")
    if dataset_key not in datasets:
        raise ValueError(
            f"Dataset '{dataset_key}' is not defined in categories JSON. "
            "Please define all datasets explicitly."
        )
    dataset_map = datasets.get(dataset_key, {})
    if not isinstance(dataset_map, dict) or not dataset_map:
        raise ValueError(f"Dataset '{dataset_key}' categories must be a non-empty object.")

    categories = {}
    for k, v in dataset_map.items():
        if str(k).strip() and str(v).strip():
            categories[str(k)] = str(v)
    if not categories:
        raise ValueError(f"Dataset '{dataset_key}' has no valid categories.")
    return categories


def normalize_filter_text(text: str, normalize_cfg: dict) -> str:
    s = str(text)
    if normalize_cfg.get("strip", True):
        s = s.strip()
    if normalize_cfg.get("collapse_spaces", True):
        s = " ".join(s.split())
    if normalize_cfg.get("lower", True):
        s = s.lower()
    return s


def to_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for x in value:
        sx = str(x).strip()
        if sx:
            out.append(sx)
    return out


def match_filter_rule(text_raw: str, text_normalized: str, rule: dict) -> bool:
    rule_type = str(rule.get("type", "regex_any")).strip().lower()
    case_sensitive = bool(rule.get("case_sensitive", False))

    if rule_type in {"regex_any", "regex_all", "exclude_regex"}:
        patterns = to_string_list(rule.get("patterns", []))
        if not patterns:
            return False
        target = text_raw
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            if rule_type in {"regex_any", "exclude_regex"}:
                return any(re.search(p, target, flags=flags) is not None for p in patterns)
            return all(re.search(p, target, flags=flags) is not None for p in patterns)
        except re.error as e:
            raise ValueError(f"Invalid regex in rule '{rule.get('id', '')}': {e}") from e

    if rule_type == "keyword_any":
        keywords = to_string_list(rule.get("keywords", []))
        if not keywords:
            return False
        target = text_raw if case_sensitive else text_normalized
        if not case_sensitive:
            keywords = [k.lower() for k in keywords]
        return any(k in target for k in keywords)

    raise ValueError(f"Unsupported filter rule type: {rule_type}")


def load_background_filters(filters_json: str, dataset_name: str) -> dict:
    if not filters_json or not os.path.exists(filters_json):
        raise FileNotFoundError(
            f"Background filters file not found: {filters_json}. "
            "Please define dataset-specific filters in this JSON."
        )

    with open(filters_json, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Background filters JSON must be an object.")

    defaults = data.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("Background filters JSON key 'defaults' must be an object.")

    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("Background filters JSON must contain object key 'datasets'.")

    dataset_key = (dataset_name or "").strip().lower()
    if dataset_key not in datasets:
        raise ValueError(
            f"Dataset '{dataset_key}' is not defined in filters JSON. "
            "Please define all enabled datasets explicitly."
        )

    cfg = datasets.get(dataset_key, {})
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset '{dataset_key}' filter config must be an object.")

    normalize_cfg = defaults.get("normalize", {})
    if normalize_cfg is None:
        normalize_cfg = {}
    if not isinstance(normalize_cfg, dict):
        raise ValueError("Filter normalize config must be an object.")
    normalize_cfg = {
        "lower": bool(normalize_cfg.get("lower", True)),
        "strip": bool(normalize_cfg.get("strip", True)),
        "collapse_spaces": bool(normalize_cfg.get("collapse_spaces", True)),
    }

    logic = str(cfg.get("logic", defaults.get("logic", "any"))).strip().lower()
    if logic not in {"any", "all"}:
        raise ValueError(f"Unsupported filter logic: {logic}")

    min_matched_rules = int(cfg.get("min_matched_rules", defaults.get("min_matched_rules", 1)))
    if min_matched_rules <= 0:
        min_matched_rules = 1

    on_no_match = str(cfg.get("on_no_match", defaults.get("on_no_match", "drop"))).strip().lower()
    if on_no_match not in {"drop", "keep"}:
        raise ValueError(f"Unsupported on_no_match: {on_no_match}")

    invalid_values = cfg.get("invalid_values", defaults.get("invalid_values", []))
    invalid_values = {normalize_filter_text(v, normalize_cfg) for v in to_string_list(invalid_values)}

    rules = cfg.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(f"Dataset '{dataset_key}' filters key 'rules' must be a list.")
    normalized_rules = []
    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Dataset '{dataset_key}' rule #{idx} must be an object.")
        rid = str(rule.get("id", f"rule_{idx}")).strip() or f"rule_{idx}"
        rtype = str(rule.get("type", "regex_any")).strip().lower()
        action = str(rule.get("action", "")).strip().lower() or "include"
        if rtype == "exclude_regex" and str(rule.get("action", "")).strip() == "":
            action = "exclude"
        if action not in {"include", "exclude"}:
            raise ValueError(f"Dataset '{dataset_key}' rule '{rid}' has unsupported action: {action}")

        rule_cfg = dict(rule)
        rule_cfg["id"] = rid
        rule_cfg["type"] = rtype
        rule_cfg["action"] = action
        normalized_rules.append(rule_cfg)

    return {
        "dataset_name": dataset_key,
        "enabled": bool(cfg.get("enabled", True)),
        "normalize": normalize_cfg,
        "invalid_values": invalid_values,
        "logic": logic,
        "min_matched_rules": min_matched_rules,
        "on_no_match": on_no_match,
        "rules": normalized_rules,
    }


def apply_background_fact_filter(
    text_df: pd.DataFrame,
    dataset_name: str,
    enable_filter: bool,
    filters_json: str,
) -> Tuple[pd.DataFrame, dict]:
    total_rows = int(len(text_df))
    passthrough_stats = {
        "enabled": False,
        "dataset_name": (dataset_name or "").strip().lower(),
        "total_rows": total_rows,
        "kept_rows": total_rows,
        "dropped_rows": 0,
        "kept_ratio": 1.0 if total_rows > 0 else 0.0,
        "matched_include_rule_frequency": {},
        "matched_exclude_rule_frequency": {},
    }
    if not enable_filter:
        return text_df, passthrough_stats

    filter_cfg = load_background_filters(filters_json, dataset_name)
    if not filter_cfg["enabled"]:
        return text_df, passthrough_stats

    include_rules = [r for r in filter_cfg["rules"] if r.get("action") == "include"]
    exclude_rules = [r for r in filter_cfg["rules"] if r.get("action") == "exclude"]
    include_counts = {r["id"]: 0 for r in include_rules}
    exclude_counts = {r["id"]: 0 for r in exclude_rules}

    keep_mask = []
    for fact in text_df["fact"].astype(str).tolist():
        text_raw = str(fact)
        text_normalized = normalize_filter_text(text_raw, filter_cfg["normalize"])

        if text_normalized in filter_cfg["invalid_values"]:
            keep_mask.append(False)
            continue

        matched_includes = []
        matched_excludes = []

        for rule in include_rules:
            if match_filter_rule(text_raw, text_normalized, rule):
                matched_includes.append(rule["id"])
                include_counts[rule["id"]] += 1

        for rule in exclude_rules:
            if match_filter_rule(text_raw, text_normalized, rule):
                matched_excludes.append(rule["id"])
                exclude_counts[rule["id"]] += 1

        if matched_excludes:
            keep_mask.append(False)
            continue

        if not include_rules:
            keep = filter_cfg["on_no_match"] == "keep"
        else:
            if len(matched_includes) == 0:
                keep = filter_cfg["on_no_match"] == "keep"
            elif filter_cfg["logic"] == "all":
                keep = len(matched_includes) == len(include_rules)
            else:
                keep = len(matched_includes) >= filter_cfg["min_matched_rules"]

        keep_mask.append(keep)

    filtered_df = text_df.loc[keep_mask].reset_index(drop=True).copy()
    kept_rows = int(len(filtered_df))
    dropped_rows = total_rows - kept_rows
    stats = {
        "enabled": True,
        "dataset_name": filter_cfg["dataset_name"],
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": dropped_rows,
        "kept_ratio": float(kept_rows / total_rows) if total_rows > 0 else 0.0,
        "include_rule_count": len(include_rules),
        "exclude_rule_count": len(exclude_rules),
        "logic": filter_cfg["logic"],
        "min_matched_rules": filter_cfg["min_matched_rules"],
        "on_no_match": filter_cfg["on_no_match"],
        "matched_include_rule_frequency": include_counts,
        "matched_exclude_rule_frequency": exclude_counts,
    }
    return filtered_df, stats


def extract_facts_from_llm_material(llm_material: str) -> List[str]:
    if not isinstance(llm_material, str) or not llm_material.strip():
        return []
    facts = []
    for raw_line in llm_material.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"fact\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        fact = match.group(1).strip() if match else line
        if fact and fact.lower() != "na":
            facts.append(fact)
    return facts


def classify_fact_with_llm(
    fact_text: str,
    categories: dict,
    client,
    model_name: str,
    retries: int = 3,
    timeout: int = 30,
) -> List[str]:
    if not fact_text.strip():
        return []

    category_lines = "\n".join([f"{k}. {v}" for k, v in sorted(categories.items(), key=lambda x: int(x[0]))])
    # prompt = (
    #     "请判断以下文本属于哪些背景类别（必须且只能返回 1 个编号（1~N））：\n"
    #     f"{category_lines}\n\n"
    #     "要求：\n"
    #     "- 仅返回类别编号，用英文逗号分隔，例如：1\n"
    #     "- 若不属于任何类别，返回：none\n"
    #     "- 不要解释，不要输出其他内容,只接受单个编号；若返回多个，就随机选一个\n\n"
    #     f"文本：{fact_text}"
    # )
    prompt = (
        "你是一个文本分类器。请根据以下类别定义，判断给定文本属于哪些类别：\n\n"
        f"类别定义：{category_lines}\n\n"
        "要求：\n"
        "- 若匹配一个或多个类别，返回编号，用英文逗号分隔，例如：1,3 \n"
        "- 若不属于任何类别，返回：none\n"
        "- 只输出编号或 none，不要输出任何解释或其他内容 \n\n"
        f"文本：{fact_text}"
    )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
                timeout=timeout,
            )
            output = response.choices[0].message.content.strip()
            if output.lower() == "none":
                return []
            nums = []
            for part in output.split(","):
                num = part.strip()
                if num in categories:
                    nums.append(num)
            return sorted(set(nums), key=lambda x: int(x))
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM classify failed after {retries} retries: {last_error}")


def build_background_annotations(
    df: pd.DataFrame,
    categories: dict,
    enable_background: bool,
    api_key: str,
    base_url: str,
    model_name: str,
) -> pd.DataFrame:
    if not enable_background:
        df["perspectives"] = ""
        if "background_patterns" not in df.columns:
            df["background_patterns"] = ""
        return df

    if not api_key:
        raise ValueError("Background generation is enabled but no API key provided.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = {}
    perspectives_col = []
    pattern_col = []

    for llm_material in df["llm_material"]:
        facts = extract_facts_from_llm_material(llm_material)
        per_fact = []
        labels_union = set()
        for fact in facts:
            if fact in cache:
                labels = cache[fact]
            else:
                labels = classify_fact_with_llm(fact, categories, client, model_name)
                cache[fact] = labels
            labels_union.update(labels)
            per_fact.append(
                {
                    "fact": fact,
                    "labels": labels,
                    "label_names": [categories[x] for x in labels],
                }
            )
        perspectives_col.append(json.dumps(per_fact, ensure_ascii=False))
        pattern_col.append(",".join(sorted(labels_union, key=lambda x: int(x))))

    df["perspectives"] = perspectives_col
    df["background_patterns"] = pattern_col
    return df


def load_context_prompts(prompts_json: str, dataset_name: str) -> Dict[str, str]:
    if not prompts_json or not os.path.exists(prompts_json):
        raise FileNotFoundError(
            f"Context prompts file not found: {prompts_json}. "
            "Please define prompts for each dataset in this JSON."
        )

    with open(prompts_json, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Context prompts JSON must be an object.")
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("Context prompts JSON must contain object key 'datasets'.")

    dataset_key = (dataset_name or "").strip().lower()
    if dataset_key not in datasets:
        raise ValueError(
            f"Dataset '{dataset_key}' is not defined in context prompts JSON. "
            "Please define all datasets explicitly."
        )

    cfg = datasets.get(dataset_key, {})
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset '{dataset_key}' prompt config must be an object.")

    system_prompt = str(cfg.get("system_prompt", "")).strip()
    user_template = str(cfg.get("user_prompt_template", "")).strip()
    if not system_prompt or not user_template:
        raise ValueError(
            f"Dataset '{dataset_key}' prompt config must contain non-empty "
            "'system_prompt' and 'user_prompt_template'."
        )

    return {
        "system_prompt": system_prompt,
        "user_prompt_template": user_template,
    }


def parse_numeric_list(value) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except Exception:
                continue
    return []


def truncate_llm_material(llm_material: str, max_facts: int) -> Dict[str, object]:
    if not isinstance(llm_material, str) or not llm_material.strip():
        return {
            "text": "",
            "total_count": 0,
            "kept_count": 0,
        }

    lines = [line.strip() for line in llm_material.splitlines() if line.strip()]
    total = len(lines)
    if max_facts <= 0:
        kept_lines = lines
    else:
        kept_lines = lines[-max_facts:]

    return {
        "text": "\n".join(kept_lines),
        "total_count": total,
        "kept_count": len(kept_lines),
    }


def render_context_user_prompt(
    row: pd.Series,
    dataset_name: str,
    user_prompt_template: str,
    max_facts: int,
) -> str:
    trunc = truncate_llm_material(row.get("llm_material", ""), max_facts=max_facts)
    facts_block = trunc["text"] if trunc["text"] else "No matched fact in this window."
    x_ot_ori = parse_numeric_list(row.get("x_ot_ori"))
    x_ot_ori_text = json.dumps(x_ot_ori, ensure_ascii=False)

    return user_prompt_template.format(
        dataset_name=dataset_name,
        split=row.get("split", ""),
        sample_id=row.get("sample_id", ""),
        date=row.get("date", ""),
        window_start=row.get("window_start", ""),
        window_end=row.get("window_end", ""),
        x_ot_ori=x_ot_ori_text,
        llm_material=facts_block,
        total_fact_count=trunc["total_count"],
        kept_fact_count=trunc["kept_count"],
        max_facts=max_facts,
    )


def call_llm_text(
    client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int = 3,
) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM context generation failed after {retries} retries: {last_error}")


def build_context_text(
    df: pd.DataFrame,
    dataset_name: str,
    enable_context_text: bool,
    prompts_json: str,
    max_facts: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    if not enable_context_text:
        if "context_text" not in df.columns:
            df["context_text"] = ""
        return df

    if not api_key:
        raise ValueError("Context generation is enabled but no API key provided.")

    prompt_cfg = load_context_prompts(prompts_json, dataset_name)
    system_prompt = prompt_cfg["system_prompt"]
    user_prompt_template = prompt_cfg["user_prompt_template"]

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = {}
    use_disk_cache = bool(cache_dir)
    if use_disk_cache:
        os.makedirs(cache_dir, exist_ok=True)

    outputs = []
    total_rows = len(df)
    for idx, row in df.iterrows():
        user_prompt = render_context_user_prompt(
            row=row,
            dataset_name=dataset_name,
            user_prompt_template=user_prompt_template,
            max_facts=max_facts,
        )
        key_src = json.dumps(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        key = hashlib.sha1(key_src.encode("utf-8-sig")).hexdigest()

        text = None
        if key in cache:
            text = cache[key]
        elif use_disk_cache:
            disk_path = os.path.join(cache_dir, f"{key}.txt")
            if os.path.exists(disk_path):
                with open(disk_path, "r", encoding="utf-8-sig") as f:
                    text = f.read().strip()

        if text is None:
            text = call_llm_text(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            cache[key] = text
            if use_disk_cache:
                disk_path = os.path.join(cache_dir, f"{key}.txt")
                with open(disk_path, "w", encoding="utf-8-sig") as f:
                    f.write(text)

        outputs.append(text)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
            print(f"Context generation progress: {idx + 1}/{total_rows}")

    df["context_text"] = outputs
    return df


def count_split_labels(label_str: str) -> List[str]:
    if not isinstance(label_str, str) or not label_str.strip():
        return []
    return [x.strip() for x in label_str.split(",") if x.strip()]


def build_pattern_frequency_report(df: pd.DataFrame, categories: dict) -> dict:
    report = {
        "temporal_pattern": {
            "unique_count": int(df["temporal_pattern"].nunique(dropna=True)),
            "counts": {str(k): int(v) for k, v in df["temporal_pattern"].value_counts(dropna=False).to_dict().items()},
        },
        "statistical_pattern": {
            "unique_count": int(df["statistical_pattern"].nunique(dropna=True)),
            "counts": {str(k): int(v) for k, v in df["statistical_pattern"].value_counts(dropna=False).to_dict().items()},
        },
        "background_patterns": {
            "unique_count": int(df["background_patterns"].nunique(dropna=True)),
            "counts": {str(k): int(v) for k, v in df["background_patterns"].value_counts(dropna=False).to_dict().items()},
        },
        "background_label_frequency": {},
    }

    label_counts = {k: 0 for k in categories.keys()}
    for val in df["background_patterns"]:
        for label in count_split_labels(val):
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = label_counts.get(label, 0) + 1

    for label, count in sorted(label_counts.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])):
        report["background_label_frequency"][label] = {
            "name": categories.get(label, ""),
            "count": int(count),
        }
    return report


def compute_trend(values: np.ndarray) -> float:
    x = np.arange(len(values))
    slope, _, _, _, _ = linregress(x, values)
    return float(slope)


def compute_volatility(values: np.ndarray) -> float:
    return float(np.std(values))


def compute_seasonality(values: np.ndarray, period: int) -> float:
    if period is None or period <= 0:
        return 0.0
    if len(values) < 2 * period:
        return 0.0
    try:
        decomposition = seasonal_decompose(
            values,
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )
        seasonal_var = float(np.var(decomposition.seasonal))
        residual_var = float(np.var(decomposition.resid))
        denom = seasonal_var + residual_var
        if denom <= 1e-12:
            return 0.0
        strength = seasonal_var / denom
        return float(max(0.0, min(1.0, strength)))
    except Exception:
        return 0.0


def compute_stat_features(x_ot, granularity: str) -> dict:
    values = np.asarray(x_ot, dtype=float)
    features = {
        "trend": compute_trend(values),
        "volatility": compute_volatility(values),
    }
    if granularity == "weekly":
        features["seasonality"] = compute_seasonality(values, period=4)
    elif granularity == "daily":
        features["seasonality"] = compute_seasonality(values, period=7)
    return features


def compute_stat_thresholds(train_df: pd.DataFrame, granularity: str) -> dict:
    trends = []
    volatilities = []
    seasonalities = []
    for x_ot in train_df["x_ot"]:
        features = compute_stat_features(x_ot, granularity)
        trends.append(features["trend"])
        volatilities.append(features["volatility"])
        if "seasonality" in features:
            seasonalities.append(features["seasonality"])

    thresholds = {
        "trend_low": float(np.percentile(trends, 33.33)),
        "trend_high": float(np.percentile(trends, 66.67)),
        "vol_low": float(np.percentile(volatilities, 33.33)),
        "vol_high": float(np.percentile(volatilities, 66.67)),
    }
    if seasonalities:
        thresholds["season_low"] = float(np.percentile(seasonalities, 33.33))
        thresholds["season_high"] = float(np.percentile(seasonalities, 66.67))
    return thresholds


def classify_stat_pattern(features: dict, granularity: str, thresholds: dict) -> str:
    if features["trend"] > thresholds["trend_high"]:
        trend = "I"
    elif features["trend"] < thresholds["trend_low"]:
        trend = "D"
    else:
        trend = "S"

    if features["volatility"] > thresholds["vol_high"]:
        vol = "H"
    elif features["volatility"] < thresholds["vol_low"]:
        vol = "L"
    else:
        vol = "M"

    pattern = f"S_{trend}_{vol}"

    if granularity in {"weekly", "daily"} and "seasonality" in features:
        season_low = thresholds.get("season_low", 0.0)
        season_high = thresholds.get("season_high", 0.0)
        if features["seasonality"] > season_high:
            season = "S"
        elif features["seasonality"] < season_low:
            season = "N"
        else:
            season = "W"
        pattern = f"{pattern}_{season}"

    return pattern


def build_llm_material(text_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> str:
    # Rule: keep textual rows whose start_date falls inside [window_start, window_end].
    mask = (text_df["start_date"] >= window_start) & (text_df["start_date"] <= window_end)
    matched = text_df.loc[mask, ["start_date", "end_date", "fact"]]
    if matched.empty:
        return ""

    lines = []
    for row in matched.itertuples(index=False):
        s = row.start_date.strftime("%Y-%m-%d")
        e = row.end_date.strftime("%Y-%m-%d")
        f = clean_fact_text(row.fact)
        lines.append(f"start_date: {s}; end_date: {e}; fact: {f}")
    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser(description="Build processed samples with llm_material from numerical+textual tables")
    parser.add_argument(
        "--numerical_csv",
        type=str,
        default="./data/raw/Time-MMD/numerical/Environment/Environment.csv",
        help="Path to numerical CSV",
    )
    parser.add_argument(
        "--textual_csv",
        type=str,
        default="./data/raw/Time-MMD/textual/Environment/Environment_search.csv",
        help="Path to textual CSV (must include start_date/end_date/fact)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default=None,
        help="Output file prefix. If not set, auto-generated from numerical_csv filename.",
    )
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=(0.7, 0.1, 0.2),
        help="train/val/test ratio",
    )
    parser.add_argument(
        "--scale",
        type=parse_bool,
        default=True,
        help="Whether to z-score OT with train split statistics",
    )
    parser.add_argument(
        "--min_date",
        type=str,
        default=None,
        help="Optional date filter: keep rows with date >= min_date (format: YYYY-MM-DD). Default: no filter.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for temporal granularity mapping (e.g., Energy/Traffic). "
             "If not set, inferred from numerical_csv filename.",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default=None,
        choices=["monthly", "weekly", "daily"],
        help="Temporal granularity override. If set, it takes precedence over dataset mapping.",
    )
    parser.add_argument(
        "--temporal_date_col",
        type=str,
        default="window_start",
        choices=["date", "window_start", "window_end"],
        help="Which date column to use for temporal_pattern generation.",
    )
    parser.add_argument(
        "--enable_background_patterns",
        type=parse_bool,
        default=False,
        help="Whether to generate background_patterns and perspectives via LLM classification.",
    )
    parser.add_argument(
        "--background_categories_json",
        type=str,
        default="./data_pipeline/background_categories.json",
        help="JSON file that stores per-dataset background category definitions.",
    )
    parser.add_argument(
        "--enable_background_fact_filter",
        type=parse_bool,
        default=False,
        help="Whether to pre-filter textual facts using dataset-specific rules before building llm_material.",
    )
    parser.add_argument(
        "--background_filters_json",
        type=str,
        default="./data_pipeline/background_filters.json",
        help="JSON file that stores per-dataset textual fact filter rules.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="API key for background classification (default: env DASHSCOPE_API_KEY).",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="qwen3-coder-plus",
        help="Model name for background classification.",
    )
    parser.add_argument(
        "--enable_context_text",
        type=parse_bool,
        default=False,
        help="Whether to generate context_text via LLM using x_ot_ori + truncated llm_material.",
    )
    parser.add_argument(
        "--context_prompts_json",
        type=str,
        default="./data_pipeline/context_prompts.json",
        help="JSON file that stores per-dataset context prompts.",
    )
    parser.add_argument(
        "--context_max_facts",
        type=int,
        default=96,
        help="Keep only the last N facts from llm_material for context generation (<=0 keeps all).",
    )
    parser.add_argument(
        "--context_openai_api_key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="API key for context generation (default: env DASHSCOPE_API_KEY).",
    )
    parser.add_argument(
        "--context_openai_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL for context generation.",
    )
    parser.add_argument(
        "--context_openai_model",
        type=str,
        default="qwen3-coder-plus",
        help="Model name for context generation.",
    )
    parser.add_argument(
        "--context_temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for context generation.",
    )
    parser.add_argument(
        "--context_max_tokens",
        type=int,
        default=300,
        help="Max tokens for context generation.",
    )
    parser.add_argument(
        "--context_timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for each context generation request.",
    )
    parser.add_argument(
        "--context_cache_dir",
        type=str,
        default="./data_pipeline/.context_cache",
        help="Optional cache directory for context generation outputs.",
    )
    parser.add_argument(
        "--save_pattern_frequency",
        type=parse_bool,
        default=True,
        help="Whether to save pattern frequency report to JSON.",
    )
    args = parser.parse_args()

    if abs(sum(args.split_ratio) - 1.0) > 1e-8:
        raise ValueError(f"split_ratio must sum to 1.0, got {args.split_ratio}")

    os.makedirs(args.output_dir, exist_ok=True)

    num_df = pd.read_csv(args.numerical_csv)
    date_col = detect_column(num_df.columns.tolist(), ["date"])
    target_col = detect_column(num_df.columns.tolist(), ["OT"])

    num_df[date_col] = pd.to_datetime(num_df[date_col], errors="coerce")
    num_df[target_col] = pd.to_numeric(num_df[target_col], errors="coerce") 
    num_df = num_df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    if args.min_date:
        cutoff = pd.to_datetime(args.min_date, errors="raise")
        num_df = num_df[num_df[date_col] >= cutoff].reset_index(drop=True)

    dataset_name = (args.dataset_name or infer_dataset_name(args.numerical_csv)).strip().lower()
    granularity = args.granularity if args.granularity else GRANULARITY.get(dataset_name, "monthly")

    text_df = pd.read_csv(args.textual_csv)
    start_col = detect_column(text_df.columns.tolist(), ["start_date"])
    end_col = detect_column(text_df.columns.tolist(), ["end_date"])
    fact_col = detect_column(text_df.columns.tolist(), ["fact"])

    text_df[start_col] = pd.to_datetime(text_df[start_col], errors="coerce")
    text_df[end_col] = pd.to_datetime(text_df[end_col], errors="coerce")
    text_df[fact_col] = text_df[fact_col].fillna("").astype(str)
    text_df = text_df.dropna(subset=[start_col, end_col]).copy()
    text_df = text_df.rename(columns={start_col: "start_date", end_col: "end_date", fact_col: "fact"})
    text_df = text_df.sort_values("start_date").reset_index(drop=True)
    text_df, filter_stats = apply_background_fact_filter(
        text_df=text_df,
        dataset_name=dataset_name,
        enable_filter=args.enable_background_fact_filter,
        filters_json=args.background_filters_json,
    )

    values = num_df[target_col].astype(float).to_numpy()
    values_raw = values.copy()
    dates = num_df[date_col]

    n = len(num_df)
    num_train = int(n * args.split_ratio[0])
    num_test = int(n * args.split_ratio[2])
    num_val = n - num_train - num_test

    border1s = [0, num_train - args.seq_len, n - num_test - args.seq_len]
    border2s = [num_train, num_train + num_val, n]
    split_names = ["train", "val", "test"]

    train_slice = values_raw[border1s[0]:border2s[0]]
    mean = train_slice.mean()
    std = train_slice.std()
    if std < 1e-12:
        std = 1.0

    if args.scale:
        values = (values_raw - mean) / std
    else:
        values = values_raw.copy()

    rows = []
    sample_id = 0
    for split_idx, split_name in enumerate(split_names):
        b1 = max(0, border1s[split_idx])
        b2 = max(0, border2s[split_idx])
        seg_values = values[b1:b2]
        seg_values_raw = values_raw[b1:b2]
        seg_dates = dates.iloc[b1:b2].reset_index(drop=True)

        total = len(seg_values)
        for start in range(0, total - args.seq_len - args.pred_len + 1):
            end_hist = start + args.seq_len
            end_pred = end_hist + args.pred_len

            x_ot = seg_values[start:end_hist].tolist()
            y_ot = seg_values[end_hist:end_pred].tolist()
            x_ot_ori = seg_values_raw[start:end_hist].tolist()
            y_ot_ori = seg_values_raw[end_hist:end_pred].tolist()

            date_value = seg_dates.iloc[start]
            window_start = seg_dates.iloc[start]
            window_end = seg_dates.iloc[end_hist - 1]
            llm_material = build_llm_material(text_df, window_start, window_end)

            rows.append(
                {
                    "split": split_name,
                    "sample_id": sample_id,
                    "date": date_value.strftime("%Y-%m-%d"),
                    "window_start": window_start.strftime("%Y-%m-%d"),
                    "window_end": window_end.strftime("%Y-%m-%d"),
                    "x_ot": x_ot,
                    "y_ot": y_ot,
                    "x_ot_ori": x_ot_ori,
                    "y_ot_ori": y_ot_ori,
                    "llm_material": llm_material,
                    # Initialize four text views as empty placeholders.
                    "context_text": "",
                    "temporal_pattern": "",
                    "statistical_pattern": "",
                    "background_patterns": "",
                    "perspectives": "",
                }
            )
            sample_id += 1

    out_df = pd.DataFrame(rows)

    categories = load_background_categories(args.background_categories_json, dataset_name)
    out_df["temporal_pattern"] = out_df[args.temporal_date_col].apply(
        lambda x: generate_temporal_pattern(x, granularity)
    )

    # Statistical pattern from x_ot, thresholds learned on train split.
    train_df = out_df[out_df["split"] == "train"]
    if len(train_df) == 0:
        raise ValueError("No train samples found. Cannot compute statistical thresholds.")
    stat_thresholds = compute_stat_thresholds(train_df, granularity)

    stat_patterns = []
    for x_ot in out_df["x_ot"]:
        try:
            f = compute_stat_features(x_ot, granularity)
            stat_patterns.append(classify_stat_pattern(f, granularity, stat_thresholds))
        except Exception:
            stat_patterns.append("S_UNKNOWN")
    out_df["statistical_pattern"] = stat_patterns

    # Background patterns and per-fact perspectives (optional LLM step).
    out_df = build_background_annotations(
        df=out_df,
        categories=categories,
        enable_background=args.enable_background_patterns,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        model_name=args.openai_model,
    )

    # Context text from x_ot_ori + truncated llm_material (optional LLM step).
    out_df = build_context_text(
        df=out_df,
        dataset_name=dataset_name,
        enable_context_text=args.enable_context_text,
        prompts_json=args.context_prompts_json,
        max_facts=args.context_max_facts,
        api_key=args.context_openai_api_key,
        base_url=args.context_openai_base_url,
        model_name=args.context_openai_model,
        temperature=args.context_temperature,
        max_tokens=args.context_max_tokens,
        timeout=args.context_timeout,
        cache_dir=args.context_cache_dir,
    )

    out_df = out_df[
        [
            "split",
            "sample_id",
            "date",
            "window_start",
            "window_end",
            "x_ot",
            "y_ot",
            "x_ot_ori",
            "y_ot_ori",
            "llm_material",
            "context_text",
            "temporal_pattern",
            "statistical_pattern",
            "background_patterns",
            "perspectives",
        ]
    ]

    if args.base_name and str(args.base_name).strip():
        base_name = args.base_name.strip()
    else:
        base_name = os.path.splitext(os.path.basename(args.numerical_csv))[0].lower() + "_generated"

    all_path = os.path.join(args.output_dir, f"{base_name}_all.csv")
    out_df.to_csv(all_path, index=False)

    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(args.output_dir, f"{base_name}_{split_name}.csv")
        out_df[out_df["split"] == split_name].to_csv(split_path, index=False)

    freq_report = build_pattern_frequency_report(out_df, categories)
    freq_path = os.path.join(args.output_dir, f"{base_name}_pattern_frequency.json")
    if args.save_pattern_frequency:
        with open(freq_path, "w", encoding="utf-8-sig") as f:
            json.dump(freq_report, f, ensure_ascii=False, indent=2)

    print(f"Saved: {all_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Split counts: {out_df['split'].value_counts().to_dict()}")
    print(f"Train mean (OT): {mean:.10f}")
    print(f"Train std  (OT): {std:.10f}")
    print(f"Scale applied to x_ot/y_ot: {args.scale}")
    print(f"Temporal pattern granularity: {granularity} (dataset: {dataset_name})")
    print(f"Temporal date column: {args.temporal_date_col}")
    print(
        "Stat thresholds: "
        f"trend=({stat_thresholds['trend_low']:.6f},{stat_thresholds['trend_high']:.6f}), "
        f"vol=({stat_thresholds['vol_low']:.6f},{stat_thresholds['vol_high']:.6f})"
        + (
            f", season=({stat_thresholds['season_low']:.6f},{stat_thresholds['season_high']:.6f})"
            if "season_low" in stat_thresholds
            else ""
        )
    )
    print(f"Statistical patterns: {out_df['statistical_pattern'].nunique()} unique")
    print(f"Background fact filter enabled: {args.enable_background_fact_filter}")
    if args.enable_background_fact_filter:
        print(f"Background filters json: {args.background_filters_json}")
        print(
            "Filtered textual facts: "
            f"{filter_stats['kept_rows']}/{filter_stats['total_rows']} "
            f"(dropped={filter_stats['dropped_rows']}, kept_ratio={filter_stats['kept_ratio']:.4f})"
        )
        print(f"Filter include-rule matches: {filter_stats['matched_include_rule_frequency']}")
        print(f"Filter exclude-rule matches: {filter_stats['matched_exclude_rule_frequency']}")
    print(f"Background patterns enabled: {args.enable_background_patterns}")
    if args.enable_background_patterns:
        print(f"Background categories json: {args.background_categories_json}")
        print(f"Background patterns: {out_df['background_patterns'].nunique()} unique")
    print(f"Context text enabled: {args.enable_context_text}")
    if args.enable_context_text:
        non_empty_context = int((out_df["context_text"].fillna("").astype(str).str.strip() != "").sum())
        print(f"Context prompts json: {args.context_prompts_json}")
        print(f"Context max facts (tail): {args.context_max_facts}")
        print(f"Context model: {args.context_openai_model}")
        print(f"Non-empty context_text rows: {non_empty_context}/{len(out_df)}")
    if args.save_pattern_frequency:
        print(f"Pattern frequency report: {freq_path}")
    print(f"Temporal pattern frequency: {freq_report['temporal_pattern']['counts']}")
    print(f"Statistical pattern frequency: {freq_report['statistical_pattern']['counts']}")
    print(f"Background pattern frequency: {freq_report['background_patterns']['counts']}")
    print(f"Background label frequency: {freq_report['background_label_frequency']}")
    if args.min_date:
        print(f"Date filter applied: date >= {args.min_date}")


if __name__ == "__main__":
    main()
