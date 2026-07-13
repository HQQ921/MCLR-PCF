"""
Perspective-aware RAG forecasting experiment.

Implements the user-specified pipeline:
1) Personalized encoding: TS encoder + frozen text encoder + projection + cross-attention.
2) Perspective common encoding: Q (query x_ot), K (neighbor x_ot), V (neighbor y_ot) with
   perspective-specific attention and matched_perspectives masking.
3) Adaptive perspective fusion: weights from h_personalized, per-view gates, weighted sum -> h_general.
4) Personalized-general fusion: gated fusion -> h_fused.
5) Decoder: MLP -> pred_len.
"""
import ast
import json
import math
import os
import re
import random
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.exp_basic import Exp_Basic


class PerspectiveAttention(nn.Module):
    """Single-view attention with independent Q/K/V projections."""
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sim_scores: Optional[torch.Tensor] = None,
        sim_alpha: float = 0.0,
        soft_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [d_model]
            k: [K, d_model]
            v: [K, d_model]
            mask: [K] bool, True for valid neighbors
        Returns:
            view feature: [d_model]
        """
        q_proj = self.q_proj(q)  # [d_model]
        k_proj = self.k_proj(k)  # [K, d_model]
        v_proj = self.v_proj(v)  # [K, d_model]
        scores = torch.matmul(k_proj, q_proj) / math.sqrt(q_proj.shape[0])  # [K]
        if sim_scores is not None and sim_alpha != 0.0:
            scores = scores + sim_alpha * sim_scores
        if soft_weights is not None:
            scores = scores + torch.log(soft_weights.clamp_min(1e-6))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # Handle all-masked case
        if torch.isinf(scores).all():
            return torch.zeros_like(q_proj)
        attn = torch.softmax(scores, dim=0).unsqueeze(1)  # [K,1]
        return (v_proj * attn).sum(dim=0)


class Exp_Forecast_RAG_Perspective(Exp_Basic):
    """
    Perspective-aware RAG forecasting experiment.
    """
    def __init__(self, args):
        super().__init__(args)

        self.use_text = True
        self.use_rag = True
        self.freeze_text_encoder = getattr(args, "freeze_text_encoder", True)
        self.freeze_ts_encoder = getattr(args, "freeze_ts_encoder", False)
        self.load_pretrained = getattr(args, "load_pretrained", True)

        # TS encoder (shared for Q/K/V)
        self.ts_encoder = self._build_model()
        self.ts_encoder = self.ts_encoder.to(self.device)
        if self.freeze_ts_encoder:
            for p in self.ts_encoder.parameters():
                p.requires_grad = False
            self.ts_encoder.eval()
        else:
            self.ts_encoder.train()

        # Text encoder (frozen by default)
        self.text_encoder, self.tokenizer = self._load_llm()
        self.text_encoder = self.text_encoder.to(self.device)
        if self.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
        else:
            self.text_encoder.train()

        # Projections and attention
        if self.args.llm_dim == self.args.d_model:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.args.llm_dim, self.args.d_model, bias=False)
        self.text_proj = self.text_proj.to(self.device)
        # Gate for personalized fusion (TS + text)
        self.personal_gate = nn.Linear(self.args.d_model * 2, self.args.d_model).to(self.device)

        # Perspective-specific attention heads
        self.perspective_heads = nn.ModuleDict({
            "time": PerspectiveAttention(self.args.d_model),
            "background": PerspectiveAttention(self.args.d_model),
            "statistical": PerspectiveAttention(self.args.d_model),
        }).to(self.device)
        # Per-view projections before entering each perspective head
        self.view_proj = nn.ModuleDict({
            "time": nn.Linear(self.args.d_model, self.args.d_model, bias=False),
            "background": nn.Linear(self.args.d_model, self.args.d_model, bias=False),
            "statistical": nn.Linear(self.args.d_model, self.args.d_model, bias=False),
        }).to(self.device)

        # Perspective importance and gates
        self.importance_net = nn.Linear(self.args.d_model, 3).to(self.device)
        self.gate_time = nn.Linear(self.args.d_model * 2, self.args.d_model).to(self.device)
        self.gate_background = nn.Linear(self.args.d_model * 2, self.args.d_model).to(self.device)
        self.gate_statistical = nn.Linear(self.args.d_model * 2, self.args.d_model).to(self.device)
        self.view_fusion_type = getattr(args, "view_fusion_type", "gated")
        self.token_pick = getattr(args, "token_pick", "ts")
        view_heads = getattr(args, "view_fusion_heads", None)
        if view_heads is None:
            view_heads = max(1, min(4, self.args.n_heads))
        if self.args.d_model % int(view_heads) != 0:
            raise ValueError(f"view_fusion_heads={view_heads} must divide d_model={self.args.d_model}")
        self.view_fusion_heads = int(view_heads)
        if self.view_fusion_type == "attention":
            self.view_fusion_attn = nn.MultiheadAttention(
                embed_dim=self.args.d_model,
                num_heads=self.view_fusion_heads,
                dropout=0.1,
                batch_first=True,
            ).to(self.device)
        elif self.view_fusion_type == "token":
            self.view_fusion_attn = nn.MultiheadAttention(
                embed_dim=self.args.d_model,
                num_heads=self.view_fusion_heads,
                dropout=0.1,
                batch_first=True,
            ).to(self.device)
            self.view_type_embed = nn.Embedding(4, self.args.d_model).to(self.device)
            self.view_fusion_ln = nn.LayerNorm(self.args.d_model).to(self.device)
            self.view_fusion_dropout = nn.Dropout(0.1).to(self.device)
        elif self.view_fusion_type == "late":
            self.view_pred_heads = nn.ModuleDict({
                "time": nn.Sequential(
                    nn.Linear(self.args.d_model, self.args.d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.args.d_model * 2, self.args.pred_len * self.args.enc_in),
                    nn.Unflatten(1, (self.args.pred_len, self.args.enc_in)),
                ),
                "background": nn.Sequential(
                    nn.Linear(self.args.d_model, self.args.d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.args.d_model * 2, self.args.pred_len * self.args.enc_in),
                    nn.Unflatten(1, (self.args.pred_len, self.args.enc_in)),
                ),
                "statistical": nn.Sequential(
                    nn.Linear(self.args.d_model, self.args.d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.args.d_model * 2, self.args.pred_len * self.args.enc_in),
                    nn.Unflatten(1, (self.args.pred_len, self.args.enc_in)),
                ),
            }).to(self.device)

        # Personalized-general fusion gate
        self.fusion_gate = nn.Linear(self.args.d_model * 2, self.args.d_model).to(self.device)
        gate_bias = getattr(args, "gate_init_bias", -1.0)
        with torch.no_grad():
            self.personal_gate.bias.data.fill_(gate_bias)
            self.gate_time.bias.data.fill_(gate_bias)
            self.gate_background.bias.data.fill_(gate_bias)
            self.gate_statistical.bias.data.fill_(gate_bias)
            self.fusion_gate.bias.data.fill_(gate_bias)

        # Prediction head (MLP -> pred_len)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.args.d_model * 2, self.args.pred_len * self.args.enc_in),
            nn.Unflatten(1, (self.args.pred_len, self.args.enc_in)),
        ).to(self.device)
        # Output bias calibration (shift only)
        self.output_bias = nn.Linear(self.args.d_model, self.args.pred_len * self.args.enc_in).to(self.device)
        with torch.no_grad():
            nn.init.zeros_(self.output_bias.weight)
            nn.init.zeros_(self.output_bias.bias)

        # Optional step-wise gate for residual rag injection.
        self.use_stepwise_rag_gate = bool(getattr(args, "use_stepwise_rag_gate", False))
        self.stepwise_gate_hidden = int(getattr(args, "stepwise_gate_hidden", 256))
        self.stepwise_gate_init_bias = float(getattr(args, "stepwise_gate_init_bias", -2.0))
        if self.use_stepwise_rag_gate:
            self.stepwise_rag_gate_head = nn.Sequential(
                nn.Linear(self.args.d_model * 2, self.stepwise_gate_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.stepwise_gate_hidden, self.args.pred_len * self.args.enc_in),
                nn.Unflatten(1, (self.args.pred_len, self.args.enc_in)),
            ).to(self.device)
            with torch.no_grad():
                final_linear = self.stepwise_rag_gate_head[3]
                if hasattr(final_linear, "bias") and final_linear.bias is not None:
                    final_linear.bias.fill_(self.stepwise_gate_init_bias)
        else:
            self.stepwise_rag_gate_head = None

        # Optional scale/shift calibration
        self.use_calibration = getattr(args, "use_calibration", False)
        if self.use_calibration:
            self.calib_scale = nn.Linear(self.args.d_model, self.args.pred_len * self.args.enc_in).to(self.device)
            self.calib_shift = nn.Linear(self.args.d_model, self.args.pred_len * self.args.enc_in).to(self.device)
            with torch.no_grad():
                nn.init.zeros_(self.calib_scale.weight)
                nn.init.zeros_(self.calib_scale.bias)
                nn.init.zeros_(self.calib_shift.weight)
                nn.init.zeros_(self.calib_shift.bias)
        else:
            self.calib_scale = None
            self.calib_shift = None

        # Future encoding for neighbor y_ot
        self.future_encode = getattr(args, "future_encode", "truncate")
        self.future_trunc_mode = getattr(args, "future_trunc_mode", "head")
        if self.future_encode == "pool":
            self.future_proj = nn.Sequential(
                nn.Linear(self.args.pred_len, self.args.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.args.d_model, self.args.d_model),
            ).to(self.device)
        else:
            self.future_proj = None

        # RAG neighbors and raw data
        self.sample_neighbors_df = None
        self.raw_df = None
        self.sample_to_row = {}
        self.neighbor_ts_cache: Dict[int, torch.Tensor] = {}
        self.neighbor_future_cache: Dict[int, torch.Tensor] = {}
        self.neighbor_stats_cache: Dict[int, tuple] = {}
        self.sample_series_cache: Dict[int, np.ndarray] = {}
        self.neighbor_scaler = None
        self.debug_neighbors = getattr(args, "debug_neighbors", False)
        self.debug_fusion_stats = getattr(args, "debug_fusion_stats", False)
        self.last_fusion_stats = None
        self.force_empty_perspectives = getattr(args, "force_empty_perspectives", False)
        self.force_all_perspectives = getattr(args, "force_all_perspectives", False)
        self.disable_perspective_mask = getattr(args, "disable_perspective_mask", False)
        self.soft_perspective_mask = getattr(args, "soft_perspective_mask", False)
        self.unmatched_weight = getattr(args, "unmatched_weight", 0.2)
        self.use_residual_rag = getattr(args, "use_residual_rag", False)
        self.rag_lambda = getattr(args, "rag_lambda", 0.3)
        self.view_mode = getattr(args, "view_mode", "all")  # all | time_only
        view_set = getattr(args, "view_set", "all")
        if view_set == "all" and self.view_mode == "time_only":
            view_set = "time"
        if view_set in {"time", "background", "statistical"}:
            self.active_views = {view_set}
        elif view_set == "time_background":
            self.active_views = {"time", "background"}
        elif view_set == "time_statistical":
            self.active_views = {"time", "statistical"}
        elif view_set == "background_statistical":
            self.active_views = {"background", "statistical"}
        else:
            self.active_views = {"time", "background", "statistical"}
        self.view_dropout = getattr(args, "view_dropout", 0.0)
        self.use_trend_norm = getattr(args, "use_trend_norm", False)
        self.use_delta_pred = getattr(args, "use_delta_pred", False)
        self.use_similarity_prior = getattr(args, "use_similarity_prior", False)
        self.similarity_alpha = getattr(args, "similarity_alpha", 1.0)
        self.use_neighbor_quality_gate = getattr(args, "use_neighbor_quality_gate", False)
        self.neighbor_quality_feature = str(getattr(args, "neighbor_quality_feature", "all")).strip().lower()
        self.neighbor_quality_alpha = float(getattr(args, "neighbor_quality_alpha", 1.0))
        self.neighbor_quality_hidden = int(getattr(args, "neighbor_quality_hidden", 8))
        if self.neighbor_quality_feature in {"seqsim", "cos"}:
            self.neighbor_quality_in_dim = 1
        else:
            self.neighbor_quality_in_dim = 6
        if self.use_neighbor_quality_gate:
            self.neighbor_quality_net = nn.Sequential(
                nn.Linear(self.neighbor_quality_in_dim, self.neighbor_quality_hidden),
                nn.ReLU(),
                nn.Linear(self.neighbor_quality_hidden, 1),
            ).to(self.device)
        else:
            self.neighbor_quality_net = None
        self.view_residual = getattr(args, "view_residual", False)
        self.residual_alpha_time = float(getattr(args, "residual_alpha_time", 1.0))
        self.residual_alpha_back = float(getattr(args, "residual_alpha_back", 0.1))
        self.residual_alpha_stat = float(getattr(args, "residual_alpha_stat", 0.1))
        self.stat_weight_cap = float(getattr(args, "stat_weight_cap", 1.0))
        self.use_view_gate = not getattr(args, "disable_view_gate", False)
        self.jitter_neighbor_id = getattr(args, "jitter_neighbor_id", False)
        self.jitter_min = getattr(args, "jitter_min", 2)
        self.jitter_max = getattr(args, "jitter_max", 7)
        jitter_seed = getattr(args, "jitter_seed", None)
        if jitter_seed is not None:
            random.seed(jitter_seed)
        self._load_per_sample_neighbors()

        # Optimizer
        trainable = []
        if not self.freeze_ts_encoder:
            trainable += list(self.ts_encoder.parameters())
        if not self.freeze_text_encoder:
            trainable += list(self.text_encoder.parameters())
        trainable += list(self.text_proj.parameters())
        trainable += list(self.personal_gate.parameters())
        trainable += list(self.perspective_heads.parameters())
        trainable += list(self.view_proj.parameters())
        trainable += list(self.importance_net.parameters())
        trainable += list(self.gate_time.parameters())
        trainable += list(self.gate_background.parameters())
        trainable += list(self.gate_statistical.parameters())
        if self.view_fusion_type == "attention":
            trainable += list(self.view_fusion_query.parameters())
            trainable += list(self.view_fusion_attn.parameters())
        elif self.view_fusion_type == "late":
            trainable += list(self.view_pred_heads.parameters())
        trainable += list(self.fusion_gate.parameters())
        trainable += list(self.prediction_head.parameters())
        trainable += list(self.output_bias.parameters())
        if self.stepwise_rag_gate_head is not None:
            trainable += list(self.stepwise_rag_gate_head.parameters())
        if self.neighbor_quality_net is not None:
            trainable += list(self.neighbor_quality_net.parameters())
        if self.use_calibration:
            trainable += list(self.calib_scale.parameters())
            trainable += list(self.calib_shift.parameters())
        if self.future_proj is not None:
            trainable += list(self.future_proj.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 1e-4),
        )
        loss_type = getattr(self.args, "loss_type", "mse")
        if loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            delta = getattr(self.args, "huber_delta", 1.0)
            self.criterion = nn.SmoothL1Loss(beta=delta)
        elif loss_type == "mix":
            alpha = getattr(self.args, "loss_alpha", 0.5)
            mse = nn.MSELoss()
            mae = nn.L1Loss()
            self.criterion = lambda pred, y: alpha * mse(pred, y) + (1.0 - alpha) * mae(pred, y)
        else:
            self.criterion = nn.MSELoss()

    def _load_llm(self):
        from transformers import AutoModel, AutoTokenizer

        llm_path = getattr(self.args, "llm_local_path", None) or "sentence-transformers/all-mpnet-base-v2"
        local_only = self.args.llm_local_path is not None
        model = AutoModel.from_pretrained(llm_path, local_files_only=local_only)
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=local_only)
        return model, tokenizer

    def _encode_text(self, text_list: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        if self.freeze_text_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
        else:
            outputs = self.text_encoder(**inputs)
        attention_mask = inputs["attention_mask"]
        text_features = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        text_features = text_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return text_features

    def _encode_timeseries(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series with shared encoder; pool by last hidden state.
        Args:
            x: [B, seq_len, 1]
        Returns:
            [B, d_model]
        """
        if hasattr(self.ts_encoder, "encoder"):
            enc_out = self.ts_encoder.enc_embedding(x, None)  # [B, T, d_model]
            enc_out, _ = self.ts_encoder.encoder(enc_out, attn_mask=None)
            return enc_out[:, -1, :]  # last hidden state
        if hasattr(self.ts_encoder, "enc_embedding") and hasattr(self.ts_encoder, "layer"):
            enc_out = self.ts_encoder.enc_embedding(x, None)  # [B, T, d_model]
            for i in range(self.ts_encoder.layer):
                enc_out = self.ts_encoder.layer_norm(self.ts_encoder.model[i](enc_out))
            return enc_out[:, -1, :]  # last hidden state
        raise AttributeError(f"Unsupported ts_encoder structure: {type(self.ts_encoder)}")

    def _apply_scaler(self, seq_list):
        if self.neighbor_scaler is None:
            return np.array(seq_list, dtype=np.float32)
        arr = np.array(seq_list, dtype=np.float32).reshape(-1, 1)
        scaled = self.neighbor_scaler.transform(arr).reshape(-1)
        return scaled

    def _tail_trend(self, seq: np.ndarray, tail: int = 6) -> float:
        if seq is None or len(seq) < 2:
            return 0.0
        n = min(int(tail), len(seq))
        y = np.asarray(seq[-n:], dtype=np.float32)
        x = np.arange(n, dtype=np.float32)
        x_center = x - x.mean()
        y_center = y - y.mean()
        denom = float((x_center ** 2).sum())
        if denom < 1e-12:
            return 0.0
        slope = float((x_center * y_center).sum() / denom)
        return slope

    def _get_scaled_series(self, sample_id: int) -> Optional[np.ndarray]:
        sid = int(sample_id)
        if sid in self.sample_series_cache:
            return self.sample_series_cache[sid]
        row_idx = self.sample_to_row.get(sid, None)
        if row_idx is None:
            return None
        x = self.raw_df.loc[row_idx, "x_ot_parsed"]
        if not isinstance(x, list):
            return None
        arr = self._apply_scaler(x)
        self.sample_series_cache[sid] = np.asarray(arr, dtype=np.float32)
        return self.sample_series_cache[sid]

    def _align_seq_len(self, seq, target_len, mode="head"):
        """Pad/truncate sequence to target_len for TS encoder."""
        arr = np.array(seq, dtype=np.float32)
        if arr.shape[0] == target_len:
            return arr
        if arr.shape[0] > target_len:
            if mode == "tail":
                return arr[-target_len:]
            return arr[:target_len]
        # pad with last value
        pad_val = arr[-1] if arr.shape[0] > 0 else 0.0
        pad = np.full((target_len - arr.shape[0],), pad_val, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def _encode_neighbor_history(self, sample_id: int) -> Optional[torch.Tensor]:
        if sample_id in self.neighbor_ts_cache:
            return self.neighbor_ts_cache[sample_id]
        row_idx = self.sample_to_row.get(int(sample_id), None)
        if row_idx is None:
            return None
        x = self.raw_df.loc[row_idx, "x_ot_parsed"]
        x_scaled = self._apply_scaler(x)
        if self.use_trend_norm:
            mean = float(np.mean(x_scaled))
            std = float(np.std(x_scaled) + 1e-6)
            x_scaled = (x_scaled - mean) / std
            self.neighbor_stats_cache[int(sample_id)] = (mean, std)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device).view(1, -1, 1)
        with torch.no_grad():
            ts_hidden = self._encode_timeseries(x_tensor)
        self.neighbor_ts_cache[sample_id] = ts_hidden.squeeze(0)
        return self.neighbor_ts_cache[sample_id]

    def _encode_neighbor_future(self, sample_id: int) -> Optional[torch.Tensor]:
        if self.future_encode != "pool" and sample_id in self.neighbor_future_cache:
            return self.neighbor_future_cache[sample_id]
        row_idx = self.sample_to_row.get(int(sample_id), None)
        if row_idx is None or "y_ot_parsed" not in self.raw_df.columns:
            return None
        y = self.raw_df.loc[row_idx, "y_ot_parsed"]
        if not isinstance(y, list):
            return None
        y_scaled = self._apply_scaler(y)
        if self.use_trend_norm:
            stats = self.neighbor_stats_cache.get(int(sample_id), None)
            if stats is None:
                x = self.raw_df.loc[row_idx, "x_ot_parsed"]
                x_scaled = self._apply_scaler(x)
                mean = float(np.mean(x_scaled))
                std = float(np.std(x_scaled) + 1e-6)
                self.neighbor_stats_cache[int(sample_id)] = (mean, std)
            else:
                mean, std = stats
            y_scaled = (y_scaled - mean) / std
        if self.future_encode == "pool":
            if self.future_proj is None:
                raise ValueError("future_encode=pool requires future_proj")
            y_scaled = self._align_seq_len(y_scaled, self.args.pred_len)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device).view(1, -1)
            y_hidden = self.future_proj(y_tensor)
            return y_hidden.squeeze(0)
        # Align to encoder expected length
        y_scaled = self._align_seq_len(y_scaled, self.args.seq_len, mode=self.future_trunc_mode)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device).view(1, -1, 1)
        with torch.no_grad():
            y_hidden = self._encode_timeseries(y_tensor)
        self.neighbor_future_cache[sample_id] = y_hidden.squeeze(0)
        return self.neighbor_future_cache[sample_id]

    def _parse_neighbors(self, x):
        if not isinstance(x, str):
            return x if x is not None else []
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return []
        if "Timestamp(" in s:
            s = re.sub(r"Timestamp\('([^']+)'\)", r'"\1"', s)
            s = re.sub(r'Timestamp\("([^"]+)"\)', r'"\1"', s)
        try:
            return ast.literal_eval(s)
        except Exception:
            try:
                return json.loads(s.replace("'", '"'))
            except Exception:
                return []

    def _normalize_matched_perspectives(self, val):
        if val is None:
            return []
        if isinstance(val, list):
            out = []
            for v in val:
                if isinstance(v, (int, float)):
                    out.append(int(v))
                elif isinstance(v, str) and v.strip().isdigit():
                    out.append(int(v.strip()))
            return out
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    out = []
                    for v in parsed:
                        if isinstance(v, (int, float)):
                            out.append(int(v))
                        elif isinstance(v, str) and v.strip().isdigit():
                            out.append(int(v.strip()))
                    return out
            except Exception:
                return []
        return []

    def _load_per_sample_neighbors(self):
        neighbors_path = getattr(self.args, "per_sample_neighbors_path", "./pattern_library/per_sample_neighbors.pkl")
        data_csv_path = os.path.join(getattr(self.args, "root_path", "./"), getattr(self.args, "data_path", ""))
        if not neighbors_path or not os.path.exists(neighbors_path):
            raise FileNotFoundError(f"per-sample neighbors not found: {neighbors_path}")
        if neighbors_path.endswith(".pkl"):
            try:
                self.sample_neighbors_df = pd.read_pickle(neighbors_path)
            except Exception as exc:
                csv_fallback = os.path.splitext(neighbors_path)[0] + ".csv"
                if os.path.exists(csv_fallback):
                    print(f"Warning: failed to load pkl ({exc}); falling back to csv: {csv_fallback}")
                    self.sample_neighbors_df = pd.read_csv(csv_fallback)
                else:
                    raise
        else:
            self.sample_neighbors_df = pd.read_csv(neighbors_path)
        if isinstance(self.sample_neighbors_df.iloc[0]["neighbors"], str):
            self.sample_neighbors_df["neighbors"] = self.sample_neighbors_df["neighbors"].apply(self._parse_neighbors)

        # Normalize matched_perspectives in each neighbor
        def _fix_neighbors(nei_list):
            if not isinstance(nei_list, list):
                return []
            out = []
            for n in nei_list:
                if not isinstance(n, dict):
                    continue
                n = dict(n)
                n["matched_perspectives"] = self._normalize_matched_perspectives(n.get("matched_perspectives", []))
                out.append(n)
            return out

        self.sample_neighbors_df["neighbors"] = self.sample_neighbors_df["neighbors"].apply(_fix_neighbors)

        self.raw_df = pd.read_csv(data_csv_path)
        self.raw_df["x_ot_parsed"] = self.raw_df["x_ot"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        if "y_ot" in self.raw_df.columns:
            self.raw_df["y_ot_parsed"] = self.raw_df["y_ot"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.sample_to_row = {int(r.sample_id): idx for idx, r in self.raw_df.iterrows()} if "sample_id" in self.raw_df.columns else {}

    def _get_neighbors(self, sample_id: int) -> List[Dict[str, Any]]:
        row = self.sample_neighbors_df[self.sample_neighbors_df["sample_id"] == int(sample_id)]
        if row.empty:
            return []
        neighbors = row.iloc[0]["neighbors"]
        if not isinstance(neighbors, list):
            return []
        # rank sort and top_k
        if any(isinstance(n, dict) and "rank" in n for n in neighbors):
            neighbors = sorted(neighbors, key=lambda n: n.get("rank", float("inf")))
        top_k = getattr(self.args, "retrieval_top_k", None)
        if top_k is not None:
            neighbors = neighbors[:top_k]
        return neighbors

    def _build_perspective_common(
        self,
        q: torch.Tensor,
        neighbors: List[Dict[str, Any]],
        query_sample_id: int,
        perspective_id: int,
        head: PerspectiveAttention,
        proj: nn.Module,
        return_k: bool = False,
    ) -> torch.Tensor:
        # Apply per-view projection before attention
        q = proj(q)
        k_list, v_list, mask_list, sim_list, weight_list = [], [], [], [], []
        quality_weight_list = []
        query_series = self._get_scaled_series(int(query_sample_id))
        for nei in neighbors:
            nei_id = nei.get("neighbor_id")
            if nei_id is None:
                continue
            matched = nei.get("matched_perspectives", [])
            if self.disable_perspective_mask:
                matched_flag = True
            elif self.force_empty_perspectives:
                matched_flag = False
            elif self.force_all_perspectives:
                matched_flag = True
            else:
                matched_flag = perspective_id in matched
            # Hard mask: skip unmatched neighbors before attention
            if not self.soft_perspective_mask and not matched_flag:
                continue
            if self.jitter_neighbor_id:
                low = min(self.jitter_min, self.jitter_max)
                high = max(self.jitter_min, self.jitter_max)
                nei_id = int(nei_id) - random.randint(low, high)
                if nei_id < 0:
                    continue
            k = self._encode_neighbor_history(int(nei_id))
            if k is None:
                continue
            v = self._encode_neighbor_future(int(nei_id))
            if v is None:
                v = k
            mask_list.append(matched_flag)
            if self.soft_perspective_mask:
                weight_list.append(1.0 if matched_flag else float(self.unmatched_weight))
            k_list.append(proj(k))
            v_list.append(proj(v))
            # similarity prior (sequence_similarity preferred; fallback final_score)
            sim_val = nei.get("sequence_similarity", None)
            if sim_val is None:
                sim_val = nei.get("final_score", 0.0)
            try:
                sim_float = float(sim_val)
            except Exception:
                sim_float = 0.0
            sim_list.append(sim_float)
            if self.use_neighbor_quality_gate and self.neighbor_quality_net is not None:
                cos_qk = float(
                    F.cosine_similarity(
                        q.view(1, -1),
                        proj(k).view(1, -1),
                        dim=1,
                        eps=1e-8,
                    ).item()
                )
                nei_series = self._get_scaled_series(int(nei_id))
                if query_series is None or nei_series is None or len(query_series) == 0 or len(nei_series) == 0:
                    last_gap = 0.0
                    mean_gap = 0.0
                    std_gap = 0.0
                    trend_gap = 0.0
                else:
                    last_gap = float(abs(float(query_series[-1]) - float(nei_series[-1])))
                    mean_gap = float(abs(float(np.mean(query_series)) - float(np.mean(nei_series))))
                    std_gap = float(abs(float(np.std(query_series)) - float(np.std(nei_series))))
                    trend_gap = float(abs(self._tail_trend(query_series) - self._tail_trend(nei_series)))
                if self.neighbor_quality_feature == "cos":
                    feat_vals = [cos_qk]
                elif self.neighbor_quality_feature == "seqsim":
                    feat_vals = [sim_float]
                else:
                    feat_vals = [sim_float, cos_qk, last_gap, mean_gap, std_gap, trend_gap]
                feat = torch.tensor([feat_vals], device=q.device, dtype=torch.float32)
                quality_logit = self.neighbor_quality_net(feat).squeeze(-1).squeeze(-1)
                quality_gate = torch.sigmoid(quality_logit).clamp_min(1e-6)
                quality_weight_list.append(quality_gate)
        if not k_list:
            return (torch.zeros_like(q), 0) if return_k else torch.zeros_like(q)
        K = torch.stack(k_list, dim=0)
        V = torch.stack(v_list, dim=0)
        mask = torch.tensor(mask_list, device=q.device, dtype=torch.bool)
        sim_scores = None
        if self.use_similarity_prior and sim_list:
            sim_scores = torch.tensor(sim_list, device=q.device, dtype=torch.float32)
        soft_weights = None
        base_weights = None
        if self.soft_perspective_mask and weight_list:
            base_weights = torch.tensor(weight_list, device=q.device, dtype=torch.float32)
            mask = None
        if self.use_neighbor_quality_gate and quality_weight_list:
            quality_weights = torch.stack(quality_weight_list).to(q.device)
            if self.neighbor_quality_alpha != 1.0:
                quality_weights = quality_weights.pow(max(0.0, self.neighbor_quality_alpha))
            base_weights = quality_weights if base_weights is None else base_weights * quality_weights
        if base_weights is not None:
            soft_weights = base_weights.clamp_min(1e-6)
        out = head(q, K, V, mask, sim_scores=sim_scores, sim_alpha=self.similarity_alpha, soft_weights=soft_weights)
        valid_k = len(k_list) if self.soft_perspective_mask else int(mask.sum().item())
        return (out, valid_k) if return_k else out

    def _compute_general(self, h_personalized: torch.Tensor, ts_features: torch.Tensor, sample_ids: List[int], return_views: bool = False):
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.tolist()
        # Per-sample three-view common encoding
        h_t_list, h_b_list, h_s_list = [], [], []
        valid_k_t, valid_k_b, valid_k_s = [], [], []
        used_count = 0
        total_k = 0
        for b_idx, sid in enumerate(sample_ids):
            neighbors = self._get_neighbors(int(sid))
            q = ts_features[b_idx]
            if "time" in self.active_views:
                h_t, kt = self._build_perspective_common(
                    q, neighbors, int(sid), 1, self.perspective_heads["time"], self.view_proj["time"], return_k=True
                )
            else:
                h_t = torch.zeros_like(q)
                kt = 0
            if "background" in self.active_views:
                h_b, kb = self._build_perspective_common(
                    q, neighbors, int(sid), 2, self.perspective_heads["background"], self.view_proj["background"], return_k=True
                )
            else:
                h_b = torch.zeros_like(q)
                kb = 0
            if "statistical" in self.active_views:
                h_s, ks = self._build_perspective_common(
                    q, neighbors, int(sid), 3, self.perspective_heads["statistical"], self.view_proj["statistical"], return_k=True
                )
            else:
                h_s = torch.zeros_like(q)
                ks = 0
            h_t_list.append(h_t)
            h_b_list.append(h_b)
            h_s_list.append(h_s)
            valid_k_t.append(kt)
            valid_k_b.append(kb)
            valid_k_s.append(ks)
            if neighbors:
                used_count += 1
                total_k += len(neighbors)
        h_t = torch.stack(h_t_list, dim=0)
        h_b = torch.stack(h_b_list, dim=0)
        h_s = torch.stack(h_s_list, dim=0)
        k_prior = torch.tensor([valid_k_t, valid_k_b, valid_k_s], device=ts_features.device, dtype=torch.float32).T

        base_feat = ts_features
        if self.view_fusion_type == "attention":
            view_seq = torch.stack([h_t, h_b, h_s], dim=1)  # [B, 3, d_model]
            if self.view_dropout > 0 and self.ts_encoder.training:
                # randomly drop one view per sample with prob=view_dropout
                drop_mask = (torch.rand(view_seq.shape[0], device=view_seq.device) < self.view_dropout)
                if drop_mask.any():
                    drop_idx = torch.randint(0, 3, (view_seq.shape[0],), device=view_seq.device)
                    view_seq[drop_mask, drop_idx[drop_mask]] = 0.0
            q = base_feat.unsqueeze(1)  # [B, 1, d_model]
            attn_out, _ = self.view_fusion_attn(query=q, key=view_seq, value=view_seq)
            h_general = attn_out.squeeze(1)
            weights = torch.zeros((ts_features.shape[0], 3), device=ts_features.device)
            weights[:, 0] = 1.0
        elif self.view_fusion_type == "token":
            tokens = torch.stack([base_feat, h_t, h_b, h_s], dim=1)  # [B, 4, d_model]
            type_ids = torch.arange(4, device=tokens.device)
            type_emb = self.view_type_embed(type_ids).unsqueeze(0)
            tokens = tokens + type_emb
            if self.view_dropout > 0 and self.ts_encoder.training:
                drop_mask = (torch.rand(tokens.shape[0], device=tokens.device) < self.view_dropout)
                if drop_mask.any():
                    drop_idx = torch.randint(1, 4, (tokens.shape[0],), device=tokens.device)
                    tokens[drop_mask, drop_idx[drop_mask]] = 0.0
            attn_out, _ = self.view_fusion_attn(query=tokens, key=tokens, value=tokens)
            tokens = self.view_fusion_ln(tokens + self.view_fusion_dropout(attn_out))
            if self.token_pick == "ts":
                h_general = tokens[:, 0]
            elif self.token_pick == "time":
                h_general = tokens[:, 1]
            elif self.token_pick == "background":
                h_general = tokens[:, 2]
            elif self.token_pick == "statistical":
                h_general = tokens[:, 3]
            elif self.token_pick == "random":
                bsz = tokens.shape[0]
                idx = torch.randint(1, 4, (bsz,), device=tokens.device)
                h_general = tokens[torch.arange(bsz, device=tokens.device), idx]
            else:
                raise ValueError(f"Unknown token_pick: {self.token_pick}")
            weights = torch.zeros((ts_features.shape[0], 3), device=ts_features.device)
            weights[:, 0] = 1.0
        elif self.view_fusion_type == "single":
            if self.active_views == {"background"}:
                h_general = h_b
                idx = 1
            elif self.active_views == {"statistical"}:
                h_general = h_s
                idx = 2
            else:
                h_general = h_t
                idx = 0
            weights = torch.zeros((ts_features.shape[0], 3), device=ts_features.device)
            weights[:, idx] = 1.0
        else:
            # Gated fusion (default)
            weights = torch.softmax(self.importance_net(base_feat), dim=1)  # [B,3]
            active_mask = torch.tensor(
                [1.0 if v in self.active_views else 0.0 for v in ("time", "background", "statistical")],
                device=weights.device,
            ).view(1, 3)
            weights = weights * active_mask
            # valid_k prior (normalize and reweight)
            if k_prior is not None:
                k_sum = k_prior.sum(dim=1, keepdim=True)
                k_weights = torch.zeros_like(k_prior)
                nonzero = k_sum.squeeze(1) > 0
                if nonzero.any():
                    k_weights[nonzero] = k_prior[nonzero] / k_sum[nonzero].clamp_min(1e-6)
                weights = weights * k_weights
            w_sum = weights.sum(dim=1, keepdim=True)
            fallback = active_mask / active_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = torch.where(w_sum > 0, weights / w_sum.clamp_min(1e-6), fallback)
            if self.view_dropout > 0 and self.ts_encoder.training:
                # randomly drop one view per sample with prob=view_dropout (no in-place ops)
                drop_mask = (torch.rand(weights.shape[0], device=weights.device) < self.view_dropout)
                if drop_mask.any():
                    drop_idx = torch.randint(0, 3, (weights.shape[0],), device=weights.device)
                    drop_onehot = F.one_hot(drop_idx, num_classes=3).float()
                    drop_onehot = drop_onehot * drop_mask.unsqueeze(1).float()
                    masked_weights = weights * (1.0 - drop_onehot)
                    weights = masked_weights / masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            if self.stat_weight_cap < 1.0:
                stat_w = torch.clamp(weights[:, 2], max=self.stat_weight_cap)
                weights = torch.stack([weights[:, 0], weights[:, 1], stat_w], dim=1)
                w_sum = weights.sum(dim=1, keepdim=True)
                fallback = active_mask / active_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                weights = torch.where(w_sum > 0, weights / w_sum.clamp_min(1e-6), fallback)
            if self.view_fusion_type == "late":
                # late fusion: no feature-level gating here
                h_general = (
                    weights[:, 0:1] * h_t
                    + weights[:, 1:2] * h_b
                    + weights[:, 2:3] * h_s
                )
            else:
                if self.use_view_gate:
                    gate_t = torch.sigmoid(self.gate_time(torch.cat([base_feat, h_t], dim=1)))
                    gate_b = torch.sigmoid(self.gate_background(torch.cat([base_feat, h_b], dim=1)))
                    gate_s = torch.sigmoid(self.gate_statistical(torch.cat([base_feat, h_s], dim=1)))
                    h_t_g = gate_t * h_t
                    h_b_g = gate_b * h_b
                    h_s_g = gate_s * h_s
                else:
                    h_t_g = h_t
                    h_b_g = h_b
                    h_s_g = h_s
                h_general = (
                    weights[:, 0:1] * h_t_g
                    + weights[:, 1:2] * h_b_g
                    + weights[:, 2:3] * h_s_g
                )
        if self.debug_neighbors:
            batch_size = len(sample_ids)
            avg_k = (total_k / used_count) if used_count > 0 else 0.0
            avg_kt = float(np.mean(valid_k_t)) if valid_k_t else 0.0
            avg_kb = float(np.mean(valid_k_b)) if valid_k_b else 0.0
            avg_ks = float(np.mean(valid_k_s)) if valid_k_s else 0.0
            print(
                f"[RAG-persp] batch={batch_size} used={used_count} avg_k={avg_k:.2f} "
                f"avg_kt={avg_kt:.2f} avg_kb={avg_kb:.2f} avg_ks={avg_ks:.2f}"
            )
        if return_views:
            return h_general, weights, (h_t, h_b, h_s)
        return h_general

    def _inject_rag_residual(
        self,
        base_pred: torch.Tensor,
        rag_pred: torch.Tensor,
        h_personalized: torch.Tensor,
        h_general: torch.Tensor,
    ):
        """
        Residual rag injection:
        - default: base + lambda * rag
        - optional step-wise gate: base + lambda * gate_t * rag
        """
        if self.use_stepwise_rag_gate and self.stepwise_rag_gate_head is not None:
            gate_in = torch.cat([h_personalized, h_general], dim=1)
            gate_t = torch.sigmoid(self.stepwise_rag_gate_head(gate_in))
            pred = base_pred + self.rag_lambda * gate_t * rag_pred
            return pred, gate_t
        pred = base_pred + self.rag_lambda * rag_pred
        return pred, None

    def forward(self, x_enc, context_text, sample_ids):
        # Per-sample trend normalization (optional)
        if self.use_trend_norm:
            mean = x_enc.mean(dim=1, keepdim=True)
            std = x_enc.std(dim=1, keepdim=True).clamp_min(1e-6)
            x_in = (x_enc - mean) / std
        else:
            mean = None
            std = None
            x_in = x_enc

        # TS encode
        ts_features = self._encode_timeseries(x_in)

        # Text encode + projection
        if context_text is None:
            context_text = [""] * x_enc.shape[0]
        text_features = self._encode_text(context_text)
        text_proj = self.text_proj(text_features)

        # Gated fusion (old version style): h_personalized = ts + gate * text
        gate_p = torch.sigmoid(self.personal_gate(torch.cat([ts_features, text_proj], dim=1)))
        h_personalized = ts_features + gate_p * text_proj

        # Common encoding from neighbors
        h_general = torch.zeros_like(h_personalized)
        view_weights = None
        view_feats = None
        if sample_ids is not None:
            if self.view_residual or self.view_fusion_type == "late":
                h_general, view_weights, view_feats = self._compute_general(
                    h_personalized, ts_features, sample_ids, return_views=True
                )
            else:
                h_general = self._compute_general(h_personalized, ts_features, sample_ids)

        # Personalized-general fusion
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_personalized, h_general], dim=1)))
        h_fused = gate * h_personalized + (1.0 - gate) * h_general

        # Decode
        step_gate = None
        if self.view_residual:
            base_pred = self.prediction_head(h_personalized)
            if sample_ids is None:
                rag_pred = torch.zeros_like(base_pred)
            else:
                h_t, h_b, h_s = view_feats
                pred_t = self.prediction_head(h_t)
                pred_b = self.prediction_head(h_b)
                pred_s = self.prediction_head(h_s)
                rag_pred = (
                    self.residual_alpha_time * pred_t
                    + self.residual_alpha_back * pred_b
                    + self.residual_alpha_stat * pred_s
                )
            pred, step_gate = self._inject_rag_residual(base_pred, rag_pred, h_personalized, h_general)
        elif self.use_residual_rag:
            base_pred = self.prediction_head(h_personalized)
            if self.view_fusion_type == "late":
                h_t, h_b, h_s = view_feats
                pred_t = self.view_pred_heads["time"](h_t)
                pred_b = self.view_pred_heads["background"](h_b)
                pred_s = self.view_pred_heads["statistical"](h_s)
                rag_pred = (
                    view_weights[:, 0:1, None] * pred_t
                    + view_weights[:, 1:2, None] * pred_b
                    + view_weights[:, 2:3, None] * pred_s
                )
            else:
                rag_pred = self.prediction_head(h_general)
            pred, step_gate = self._inject_rag_residual(base_pred, rag_pred, h_personalized, h_general)
        else:
            pred = self.prediction_head(h_fused)
            base_pred = None
            rag_pred = None
        if self.debug_fusion_stats:
            with torch.no_grad():
                stats = {
                    "gate_mean": gate.mean().item(),
                    "gate_min": gate.min().item(),
                    "gate_max": gate.max().item(),
                    "h_p_norm": h_personalized.norm(dim=1).mean().item(),
                    "h_g_norm": h_general.norm(dim=1).mean().item(),
                }
                if base_pred is not None and rag_pred is not None:
                    stats["base_pred_norm"] = base_pred.norm(dim=(1, 2)).mean().item()
                    stats["rag_pred_norm"] = rag_pred.norm(dim=(1, 2)).mean().item()
                if step_gate is not None:
                    stats["step_gate_mean"] = step_gate.mean().item()
                    stats["step_gate_min"] = step_gate.min().item()
                    stats["step_gate_max"] = step_gate.max().item()
                self.last_fusion_stats = stats
        bias = self.output_bias(ts_features).view(-1, self.args.pred_len, self.args.enc_in)
        pred = pred + bias
        if self.use_calibration:
            scale = self.calib_scale(ts_features).view(-1, self.args.pred_len, self.args.enc_in)
            shift = self.calib_shift(ts_features).view(-1, self.args.pred_len, self.args.enc_in)
            pred = pred * (1.0 + scale) + shift
        if self.use_delta_pred:
            last_val = x_in[:, -1:, :]
            pred = pred + last_val
        if self.use_trend_norm:
            pred = pred * std + mean
        return pred

    def train_one_epoch(self, train_loader, epoch):
        self.ts_encoder.train(not self.freeze_ts_encoder)
        self.text_encoder.train(not self.freeze_text_encoder)
        self.text_proj.train()
        self.personal_gate.train()
        self.perspective_heads.train()
        self.view_proj.train()
        self.importance_net.train()
        self.gate_time.train()
        self.gate_background.train()
        self.gate_statistical.train()
        if self.view_fusion_type == "attention":
            self.view_fusion_attn.train()
        elif self.view_fusion_type == "token":
            self.view_fusion_attn.train()
            self.view_type_embed.train()
            self.view_fusion_ln.train()
        elif self.view_fusion_type == "late":
            self.view_pred_heads.train()
        if self.future_proj is not None:
            self.future_proj.train()
        if self.stepwise_rag_gate_head is not None:
            self.stepwise_rag_gate_head.train()
        self.fusion_gate.train()
        self.output_bias.train()
        if self.use_calibration:
            self.calib_scale.train()
            self.calib_shift.train()
        self.prediction_head.train()

        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x_enc = batch["x_enc"].to(self.device)
            y = batch["y"].to(self.device)
            context_text = batch.get("context_text", None)
            sample_ids = batch.get("sample_id", None)

            pred = self.forward(x_enc, context_text, sample_ids)
            if self.debug_fusion_stats and batch_idx == 0 and self.last_fusion_stats:
                s = self.last_fusion_stats
                msg = (
                    f"[debug] gate_mean={s['gate_mean']:.3f} "
                    f"(min={s['gate_min']:.3f}, max={s['gate_max']:.3f}) "
                    f"h_p_norm={s['h_p_norm']:.3f} h_g_norm={s['h_g_norm']:.3f}"
                )
                if "base_pred_norm" in s:
                    msg += f" base_pred_norm={s['base_pred_norm']:.3f} rag_pred_norm={s['rag_pred_norm']:.3f}"
                if "step_gate_mean" in s:
                    msg += (
                        f" step_gate_mean={s['step_gate_mean']:.3f}"
                        f"(min={s['step_gate_min']:.3f}, max={s['step_gate_max']:.3f})"
                    )
                print(msg)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.ts_encoder.eval()
        self.text_encoder.eval()
        self.text_proj.eval()
        self.personal_gate.eval()
        self.perspective_heads.eval()
        self.view_proj.eval()
        self.importance_net.eval()
        self.gate_time.eval()
        self.gate_background.eval()
        self.gate_statistical.eval()
        if self.view_fusion_type == "attention":
            self.view_fusion_attn.eval()
        elif self.view_fusion_type == "token":
            self.view_fusion_attn.eval()
            self.view_type_embed.eval()
            self.view_fusion_ln.eval()
        elif self.view_fusion_type == "late":
            self.view_pred_heads.eval()
        if self.future_proj is not None:
            self.future_proj.eval()
        if self.stepwise_rag_gate_head is not None:
            self.stepwise_rag_gate_head.eval()
        self.fusion_gate.eval()
        self.output_bias.eval()
        if self.use_calibration:
            self.calib_scale.eval()
            self.calib_shift.eval()
        self.prediction_head.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_enc = batch["x_enc"].to(self.device)
                y = batch["y"].to(self.device)
                context_text = batch.get("context_text", None)
                sample_ids = batch.get("sample_id", None)

                pred = self.forward(x_enc, context_text, sample_ids)
                loss = self.criterion(pred, y)
                mse = ((pred - y) ** 2).mean()
                mae = (pred - y).abs().mean()
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
        if self.debug_fusion_stats and self.last_fusion_stats:
            s = self.last_fusion_stats
            msg = (
                f"[debug] gate_mean={s['gate_mean']:.3f} "
                f"(min={s['gate_min']:.3f}, max={s['gate_max']:.3f}) "
                f"h_p_norm={s['h_p_norm']:.3f} h_g_norm={s['h_g_norm']:.3f}"
            )
            if "base_pred_norm" in s:
                msg += f" base_pred_norm={s['base_pred_norm']:.3f} rag_pred_norm={s['rag_pred_norm']:.3f}"
            if "step_gate_mean" in s:
                msg += (
                    f" step_gate_mean={s['step_gate_mean']:.3f}"
                    f"(min={s['step_gate_min']:.3f}, max={s['step_gate_max']:.3f})"
                )
            print(msg)

        return {
            "loss": total_loss / len(val_loader),
            "mse": total_mse / len(val_loader),
            "mae": total_mae / len(val_loader),
        }

    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load TS encoder weights from a contrastive checkpoint."""
        print(f"\nLoading pretrained encoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ts_state = None
        if isinstance(checkpoint, dict):
            if "ts_encoder" in checkpoint and isinstance(checkpoint["ts_encoder"], dict):
                ts_state = checkpoint["ts_encoder"]
            else:
                prefixed = {}
                for key, value in checkpoint.items():
                    if key.startswith("ts_encoder."):
                        prefixed[key.replace("ts_encoder.", "")] = value
                if prefixed:
                    ts_state = prefixed
        if ts_state is None:
            ts_state = checkpoint
        # strip module. prefix if present
        if isinstance(ts_state, dict) and any(k.startswith("module.") for k in ts_state.keys()):
            ts_state = {k.replace("module.", ""): v for k, v in ts_state.items()}
        # filter out keys with mismatched shapes
        if isinstance(ts_state, dict):
            model_state = self.ts_encoder.state_dict()
            filtered_state = {}
            skipped = []
            for k, v in ts_state.items():
                if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        skipped.append(k)
                elif k in model_state:
                    filtered_state[k] = v
                else:
                    skipped.append(k)
            if skipped:
                print(f"Skipped {len(skipped)} keys due to shape/key mismatch (e.g., {skipped[:3]})")
            ts_state = filtered_state
        missing, unexpected = self.ts_encoder.load_state_dict(ts_state, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else missing)
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else unexpected)
        print("Pretrained encoder loaded")
