#!/bin/bash
python3 run_forecast_rag_perspective.py \
  --data_path data/processed/Traffic/generated/traffic_generated_all_8.csv \
  --pred_len 8 \
  --rag_lambda 4.0 \
  --residual_alpha_stat 0.01 \
  --residual_alpha_back 0.01 \
  --neighbor_quality_alpha 0.3 \
  --per_sample_neighbors_path ./data/processed/Traffic/pattern_library/per_sample_neighbors.csv \
  --llm_local_path ./models/all-mpnet-base-v2 \
  --retrieval_top_k 10 \
  --use_trend_norm \
  --disable_view_gate \
  --view_residual \
  --use_delta_pred \
  --gate_init_bias -5.0 \
  --use_neighbor_quality_gate \
  --pretrained_encoder False \
  --checkpoints ./checkpoints/forecast_rag/traffic_pred8
