#!/bin/bash
python3 run_forecast_rag_perspective.py \
  --data_path data/processed/Agriculture/agriculture_generated_all_pred12.csv \
  --pred_len 12 \
  --per_sample_neighbors_path ./data/processed/Agriculture/pattern_library/per_sample_neighbors.csv \
  --llm_local_path ./models/all-mpnet-base-v2 \
  --retrieval_top_k 1 \
  --use_trend_norm \
  --disable_view_gate \
  --view_residual \
  --use_delta_pred \
  --gate_init_bias -5.0 \
  --use_neighbor_quality_gate \
  --neighbor_quality_alpha 2.0 \
  --rag_lambda 0.2 \
  --pretrained_encoder False \
  --checkpoints ./checkpoints/forecast_rag/agriculture_pred12
