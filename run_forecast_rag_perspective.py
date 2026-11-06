"""
Run perspective-aware RAG forecasting.

Example:
python run_forecast_rag_perspective.py ^
  --data_path data/processed/Traffic/generated/traffic_generated_all.csv ^
  --per_sample_neighbors_path ./pattern_library/per_sample_neighbors.pkl ^
  --llm_local_path ./models/all-mpnet-base-v2
"""
import argparse
import os
import random
import time
import torch
from exp.exp_forecast_rag_perspective import Exp_Forecast_RAG_Perspective
from data_provider.data_loader import data_provider


def get_args():
    parser = argparse.ArgumentParser(description="Perspective-aware RAG Forecasting")

    # Data
    parser.add_argument("--task_name", type=str, default="long_term_forecast",
                        help="Task name for TimesNet")
    parser.add_argument("--root_path", type=str, default="./", help="Root path of data")
    parser.add_argument("--data_path", type=str,
                        default="data/processed/Traffic/generated/traffic_generated_all.csv",
                        help="Data file path")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--label_len", type=int, default=4)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--dec_in", type=int, default=1)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--scale", type=bool, default=False)

    # Model
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=3)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_true",
                        help="Use distilling in Informer-style models")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_kernels", type=int, default=3)

    # Encoders
    parser.add_argument("--llm_local_path", type=str, default=None)
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--pretrained_encoder", type=str,
                        default="./checkpoints/contrastive/best_model.pth")
    parser.add_argument("--freeze_ts_encoder", type=bool, default=False)
    parser.add_argument("--freeze_text_encoder", type=bool, default=True)
    parser.add_argument("--load_pretrained", type=bool, default=True)

    # Retrieval
    parser.add_argument("--per_sample_neighbors_path", type=str,
                        default="./pattern_library/per_sample_neighbors.pkl")
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--debug_neighbors", action="store_true",
                        help="Print per-batch neighbor usage stats")
    parser.add_argument("--force_empty_perspectives", action="store_true",
                        help="Force matched_perspectives to be treated as empty (no view matches)")
    parser.add_argument("--force_all_perspectives", action="store_true",
                        help="Force matched_perspectives to be treated as [1,2,3] (all views match)")
    parser.add_argument("--disable_perspective_mask", action="store_true",
                        help="Disable perspective masking (all neighbors kept for every view)")
    parser.add_argument("--soft_perspective_mask", action="store_true",
                        help="Use soft mask instead of hard masking for unmatched perspectives")
    parser.add_argument("--unmatched_weight", type=float, default=0.2,
                        help="Weight for unmatched perspectives when soft mask is enabled")
    parser.add_argument("--use_residual_rag", action="store_true",
                        help="Use base_pred + rag_lambda * rag_pred instead of fused prediction")
    parser.add_argument("--rag_lambda", type=float, default=0.5,
                        help="Residual weight for rag_pred when use_residual_rag is on")
    parser.add_argument("--gate_init_bias", type=float, default=-1.0,
                        help="Init bias for gate sigmoids; larger -> more open gates")
    parser.add_argument("--view_fusion_type", type=str, default="gated",
                        choices=["gated", "attention", "late", "token", "single"],
                        help="Fusion type for three-view common encoding")
    parser.add_argument("--view_fusion_heads", type=int, default=None,
                        help="Number of heads for view fusion attention (default: min(4, n_heads))")
    parser.add_argument("--token_pick", type=str, default="ts",
                        choices=["ts", "time", "background", "statistical", "random"],
                        help="Token to use as h_general when view_fusion_type=token")
    parser.add_argument("--view_mode", type=str, default="all",
                        choices=["all", "time_only"],
                        help="Use only time view (time_only) or all three views (all)")
    parser.add_argument("--view_set", type=str, default="all",
                        choices=["all", "time", "background", "statistical",
                                 "time_background", "time_statistical", "background_statistical"],
                        help="Select active views for ablation (gated fusion recommended)")
    parser.add_argument("--view_residual", action="store_true",
                        help="Use time view prediction as main output and add background/statistical residuals")
    parser.add_argument("--residual_alpha_time", type=float, default=1.0,
                        help="Residual weight for time view when view_residual is enabled")
    parser.add_argument("--residual_alpha_back", type=float, default=0.2,
                        help="Residual weight for background view when view_residual is enabled")
    parser.add_argument("--residual_alpha_stat", type=float, default=1.0,
                        help="Residual weight for statistical view when view_residual is enabled")
    parser.add_argument("--view_dropout", type=float, default=0.0,
                        help="Drop one view with given probability during training (e.g., 0.3)")
    parser.add_argument("--stat_weight_cap", type=float, default=1.0,
                        help="Upper bound for statistical view weight in gated fusion (1.0 = no cap)")
    parser.add_argument("--disable_view_gate", action="store_true",
                        help="Disable per-view gates inside gated fusion (use weights only)")
    parser.add_argument("--use_trend_norm", action="store_true",
                        help="Normalize each sample by its own mean/std (trend-focused encoding)")
    parser.add_argument("--use_delta_pred", action="store_true",
                        help="Predict delta from last normalized value and add back")
    parser.add_argument("--use_calibration", action="store_true",
                        help="Apply scale/shift calibration head to predictions")
    parser.add_argument("--use_similarity_prior", action="store_true",
                        help="Add retrieval similarity score into attention logits")
    parser.add_argument("--similarity_alpha", type=float, default=0.0,
                        help="Weight for similarity prior added to attention logits")
    parser.add_argument("--use_neighbor_quality_gate", action="store_true",
                        help="Learn per-neighbor quality weights from sequence_similarity")
    parser.add_argument("--neighbor_quality_alpha", type=float, default=0.3,
                        help="Exponent for learned neighbor quality weight (>=0)")
    parser.add_argument("--neighbor_quality_hidden", type=int, default=8,
                        help="Hidden size of neighbor quality MLP")
    parser.add_argument("--future_encode", type=str, default="truncate",
                        choices=["truncate", "pool"],
                        help="Neighbor y_ot encoding: truncate to seq_len or pool full y_ot")
    parser.add_argument("--future_trunc_mode", type=str, default="head",
                        choices=["head", "tail"],
                        help="When future_encode=truncate and y_ot is longer than seq_len, keep head or tail")
    parser.add_argument("--debug_random_neighbor", action="store_true",
                        help="Print neighbors for one random test sample_id")
    parser.add_argument("--debug_neighbor_max", type=int, default=10,
                        help="Max neighbors to print when debug_random_neighbor is enabled")
    parser.add_argument("--debug_fusion_stats", action="store_true",
                        help="Print fusion/gate stats from the first train batch and last val/test batch")
    parser.add_argument("--jitter_neighbor_id", action="store_true",
                        help="Randomly subtract jitter_min~jitter_max from neighbor_id during retrieval")
    parser.add_argument("--jitter_min", type=int, default=2,
                        help="Minimum jitter to subtract from neighbor_id")
    parser.add_argument("--jitter_max", type=int, default=7,
                        help="Maximum jitter to subtract from neighbor_id")
    parser.add_argument("--jitter_seed", type=int, default=None,
                        help="Random seed for neighbor_id jitter (optional)")

    # Training
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "mae", "huber", "mix"],
                        help="Loss type for training")
    parser.add_argument("--huber_delta", type=float, default=1.0,
                        help="Delta (beta) for SmoothL1/Huber loss")
    parser.add_argument("--loss_alpha", type=float, default=0.5,
                        help="Alpha for mix loss: alpha*MSE + (1-alpha)*MAE")
    parser.add_argument("--select_metric", type=str, default="loss",
                        choices=["loss", "mse", "mae", "mix"],
                        help="Metric used for early stopping and best model selection")
    parser.add_argument("--select_alpha", type=float, default=0.5,
                        help="Alpha for mix selection: alpha*MSE + (1-alpha)*MAE")

    # Checkpoints
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/forecast_rag/perspective")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    return args


def main():
    t_start = time.time()
    args = get_args()
    os.makedirs(args.checkpoints, exist_ok=True)

    print("=" * 80)
    print("Training Mode: perspective_rag")
    print(f"Checkpoint: {args.checkpoints}")
    print("=" * 80)

    print("\n1. Loading data...")
    t_data_start = time.time()
    train_data, train_loader = data_provider(args, flag="train")
    val_data, val_loader = data_provider(args, flag="val")
    test_data, test_loader = data_provider(args, flag="test")
    t_data_end = time.time()

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Data loading time: {t_data_end - t_data_start:.1f}s")

    print("\n2. Building model...")
    exp = Exp_Forecast_RAG_Perspective(args)
    exp.neighbor_scaler = getattr(train_data, "scaler", None)
    exp.ts_encoder.to(args.device)
    exp.text_encoder.to(args.device)
    exp.prediction_head.to(args.device)

    if args.load_pretrained and os.path.exists(args.pretrained_encoder):
        print(f"\n3. Loading pretrained encoder from {args.pretrained_encoder}")
        exp.load_pretrained_encoder(args.pretrained_encoder)
    else:
        print("\n3. Training from scratch (no pretrained encoder).")

    print("\n4. Training...")
    t_train_start = time.time()
    best_val_metric = float("inf")
    patience_counter = 0
    best_path = os.path.join(args.checkpoints, "best_model.pth")

    for epoch in range(1, args.train_epochs + 1):
        t_epoch_start = time.time()
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.train_epochs}")
        print(f"{'='*80}")

        train_loss = exp.train_one_epoch(train_loader, epoch)
        val_metrics = exp.validate(val_loader)
        val_loss = val_metrics["loss"]
        val_mse = val_metrics["mse"]
        val_mae = val_metrics["mae"]
        t_epoch_end = time.time()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
        print(f"Epoch time: {t_epoch_end - t_epoch_start:.1f}s")

        if args.select_metric == "mse":
            val_score = val_mse
        elif args.select_metric == "mae":
            val_score = val_mae
        elif args.select_metric == "mix":
            val_score = args.select_alpha * val_mse + (1.0 - args.select_alpha) * val_mae
        else:
            val_score = val_loss

        if val_score < best_val_metric:
            best_val_metric = val_score
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "ts_encoder": exp.ts_encoder.state_dict(),
                "text_encoder": exp.text_encoder.state_dict(),
                "prediction_head": exp.prediction_head.state_dict(),
                "args": vars(args),
            }
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {os.path.abspath(best_path)} (select_metric={args.select_metric}, score={val_score:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    t_train_end = time.time()
    print(f"\nTotal training time: {t_train_end - t_train_start:.1f}s")

    print("\n5. Testing...")
    t_test_start = time.time()
    if os.path.exists(best_path):
        best_checkpoint = torch.load(best_path, map_location=args.device)
        exp.ts_encoder.load_state_dict(best_checkpoint["ts_encoder"], strict=False)
        exp.text_encoder.load_state_dict(best_checkpoint["text_encoder"], strict=False)
        exp.prediction_head.load_state_dict(best_checkpoint["prediction_head"], strict=False)

    test_metrics = exp.validate(test_loader)
    t_test_end = time.time()
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test MSE: {test_metrics['mse']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    print(f"  Testing time: {t_test_end - t_test_start:.1f}s")

    t_end = time.time()
    print(f"\n{'='*80}")
    print(f"Total runtime: {t_end - t_start:.1f}s ({(t_end - t_start)/60:.1f}min)")
    print(f"{'='*80}")

    if args.debug_random_neighbor:
        sample_ids = getattr(test_data, "sample_ids", None)
        if sample_ids:
            sid = random.choice(sample_ids)
            neighbors = exp._get_neighbors(int(sid))
            print(f"\n[debug] Random test sample_id: {sid}")
            if not neighbors:
                print("[debug] No neighbors found.")
            else:
                max_n = min(len(neighbors), args.debug_neighbor_max)
                for i, nei in enumerate(neighbors[:max_n], start=1):
                    print(
                        f"[debug] {i:02d} neighbor_id={nei.get('neighbor_id')} "
                        f"rank={nei.get('rank')} sim={nei.get('sequence_similarity')} "
                        f"matched_perspectives={nei.get('matched_perspectives')}"
                    )
        else:
            print("[debug] test_data has no sample_ids; cannot print neighbors.")

    results_path = os.path.join(args.checkpoints, "test_results.txt")
    with open(results_path, "w") as f:
        f.write("Mode: perspective_rag\n")
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test MSE: {test_metrics['mse']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"Best Val Metric ({args.select_metric}): {best_val_metric:.4f}\n")
    print(f"Test results saved to {results_path}")


if __name__ == "__main__":
    main()
