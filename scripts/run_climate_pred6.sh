#!/bin/bash
python3 run_forecast_rag_perspective.py \
  --data_path data/processed/Climate/climate_generated_all.csv \
  --pred_len 6 \
  --gate_init_bias -5.0 \
  --learning_rate 3e-5 \
  --llm_local_path ./models/all-mpnet-base-v2 \
  --neighbor_quality_alpha 0.3 \
  --patience 10 \
  --per_sample_neighbors_path ./data/processed/Climate/pattern_library/per_sample_neighbors.pkl \
  --pretrained_encoder False \
  --rag_lambda 1.0 \
  --residual_alpha_back 0.1 \
  --residual_alpha_stat 0.2 \
  --retrieval_top_k 4 \
  --train_epochs 80 \
  --use_neighbor_quality_gate \
  --view_residual \
  --checkpoints ./checkpoints/forecast_rag/climate_pred6
