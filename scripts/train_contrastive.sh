#!/bin/bash

# ========================================
# 对比学习训练脚本示例
# ========================================

# GPU设置
GPU=0

# 模型选择
MODEL="TimesNet"        # 可选: TimesNet, iTransformer, Autoformer, DLinear, PatchTST
LLM_MODEL="BERT"        # 可选: BERT, LLAMA2, GPT2, GPT2-M

# 数据路径
ROOT_PATH="./data/Traffic/"
DATA_PATH="Traffic_samples_with_context.csv"

# 训练参数
BATCH_SIZE=32
EPOCHS=100
TS_ENCODER_LR=1e-4
PROJ_LR=1e-3

# 对比学习参数
LAMBDA_TS_TS=0.8        # TS-TS对比权重（主任务）
LAMBDA_TS_TEXT=0.2      # TS-Text对比权重（辅助任务）
TEMPERATURE=0.07        # 温度参数
PROJ_DIM=128            # 投影维度

# 保存路径
SAVE_DIR="./checkpoints/contrastive/${MODEL}_${LLM_MODEL}_ts${LAMBDA_TS_TS}_text${LAMBDA_TS_TEXT}/"

echo "=========================================="
echo "开始训练对比学习模型"
echo "=========================================="
echo "时序模型: ${MODEL}"
echo "文本模型: ${LLM_MODEL}"
echo "损失权重: TS-TS=${LAMBDA_TS_TS}, TS-Text=${LAMBDA_TS_TEXT}"
echo "保存路径: ${SAVE_DIR}"
echo "=========================================="

python run_contrastive.py \
  --model ${MODEL} \
  --llm_model ${LLM_MODEL} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --seq_len 12 \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --ts_encoder_lr ${TS_ENCODER_LR} \
  --proj_lr ${PROJ_LR} \
  --lambda_ts_ts ${LAMBDA_TS_TS} \
  --lambda_ts_text ${LAMBDA_TS_TEXT} \
  --temperature ${TEMPERATURE} \
  --proj_dim ${PROJ_DIM} \
  --d_model 512 \
  --d_ff 2048 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --patience 10 \
  --gpu ${GPU} \
  --save_dir ${SAVE_DIR} \
  --seed 2024

echo "训练完成!"
