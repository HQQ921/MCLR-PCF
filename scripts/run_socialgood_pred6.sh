#!/bin/bash
python3 run_forecast_rag_perspective.py \
  --data_path data/processed/SocialGood/socialgood_generated_all.csv \
  --pred_len 6 \
  --per_sample_neighbors_path ./data/processed/SocialGood/pattern_library/per_sample_neighbors.pkl \
  --llm_local_path ./models/all-mpnet-base-v2 \
  --view_residual \
  --use_neighbor_quality_gate \
  --gate_init_bias -10.0 \
  --residual_alpha_back 0.05 \
  --residual_alpha_stat 0.05 \
  --learning_rate 3e-5 \
  --train_epochs 80 \
  --patience 10 \
  --rag_lambda 0.8 \
  --pretrained_encoder False \
  --retrieval_top_k 7 \
  --neighbor_quality_alpha 0.3 \
  --checkpoints ./checkpoints/forecast_rag/socialgood_pred6
