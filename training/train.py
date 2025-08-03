#!/usr/bin/env python3
"""
RandLA-Net Training Script for Stair Detection
===============================================

This script trains a RandLA-Net model for 3-class stair segmentation:
- Class 0: Background
- Class 1: Riser  
- Class 2: Tread

Training Features:
- Multi-scale point cloud sampling
- Weighted cross-entropy loss for class imbalance
- Early stopping with IoU monitoring
- Mixed precision training (AMP)
- Cosine annealing learning rate schedule
- Model checkpointing and validation

Usage:
    python train.py --data_path /path/to/stairs/dataset --epochs 100

Author: Vincent Yeung
"""

#!/usr/bin/env python3

import os
import sys
import argparse

# ─── Make sure this script's folder is first on PYTHONPATH ────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import glob, pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train or fine-tune RandLA-Net model for stair detection')
parser.add_argument('--fine-tune', action='store_true', help='Enable fine-tuning mode')
parser.add_argument('--resume-epoch', type=int, default=None, help='Resume from specific epoch')
parser.add_argument('--background-weight', type=float, default=None, help='Weight for background class')
parser.add_argument('--riser-weight', type=float, default=None, help='Weight for riser class')
parser.add_argument('--tread-weight', type=float, default=None, help='Weight for tread class')
parser.add_argument('--focal-gamma', type=float, default=None, help='Gamma parameter for focal loss')
parser.add_argument('--learning-rate', type=float, default=None, help='Initial learning rate')
args = parser.parse_args()

import hyperparameters as H

# Apply command-line parameter overrides if provided
if args.background_weight is not None:
    H.CLASS_WEIGHTS[0] = args.background_weight
    print(f"🔄 Overriding background weight: {H.CLASS_WEIGHTS[0]}")

if args.riser_weight is not None:
    H.CLASS_WEIGHTS[1] = args.riser_weight
    print(f"🔄 Overriding riser weight: {H.CLASS_WEIGHTS[1]}")

if args.tread_weight is not None:
    H.CLASS_WEIGHTS[2] = args.tread_weight
    print(f"🔄 Overriding tread weight: {H.CLASS_WEIGHTS[2]}")

if args.focal_gamma is not None:
    H.FOCAL_GAMMA = args.focal_gamma
    print(f"🔄 Overriding focal gamma: {H.FOCAL_GAMMA}")

if args.learning_rate is not None:
    H.LR_INITIAL = args.learning_rate
    print(f"🔄 Overriding learning rate: {H.LR_INITIAL}")

print(">>> D_OUT:",             H.D_OUT)
print(">>> NUM_LAYERS:",        H.NUM_LAYERS)
print(">>> SUB_SAMPLING_RATIO:", H.SUB_SAMPLING_RATIO)
print("hyp.sub_sampling_ratio =", H.SUB_SAMPLING_RATIO)
from dataset import StairPatchDataset
from model import build_model
from loss import FocalCSCELoss  # Use Focal Loss instead of CSCELoss
from metrics import compute_class_iou
from utils import move_to_device, save_colored_point_cloud


files = glob.glob("data/train/*.pickle")[:10]  # just sample first 10
for fn in files:
    data = pickle.load(open(fn, "rb"))
    labels = data["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    freqs = dict(zip(unique.tolist(), counts.tolist()))
    total = labels.shape[0]
    print(f"{fn} →", {k: f"{v/total*100:.2f}%" for k,v in freqs.items()})

def main():
    # ─── Logging & Checkpoints ────────────────────────────────────────────────
    # Create a unique run name for fine-tuning
    run_name = "exp1"
    if args.fine_tune:
        run_name = f"fine_tune_bg{H.CLASS_WEIGHTS[0]}_riser{H.CLASS_WEIGHTS[1]}_tread{H.CLASS_WEIGHTS[2]}_gamma{H.FOCAL_GAMMA}_lr{H.LR_INITIAL}"
        if args.resume_epoch:
            run_name += f"_from_epoch{args.resume_epoch}"
    
    writer    = SummaryWriter(f"runs/{run_name}")
    ckpt_dir  = "checkpoints"
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    
    # For fine-tuning, we want to load a specific epoch's checkpoint if specified
    if args.fine_tune and args.resume_epoch:
        source_ckpt_path = os.path.join(ckpt_dir, f"epoch_{args.resume_epoch}.pth")
        fine_tune_ckpt_path = os.path.join(ckpt_dir, f"best_fine_tuned.pth")
        if os.path.exists(source_ckpt_path):
            print(f"🔍 Fine-tuning from checkpoint: {source_ckpt_path}")
            ckpt_path = fine_tune_ckpt_path
        else:
            print(f"⚠️ Checkpoint for epoch {args.resume_epoch} not found. Using best checkpoint.")
    
    vis_dir   = "val_vis"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ─── Data Loaders ─────────────────────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        StairPatchDataset("train"),
        batch_size   = H.BATCH_SIZE,
        shuffle      = True,
        num_workers  = 2,
        pin_memory   = True,
        drop_last    = True
    )
    val_loader = torch.utils.data.DataLoader(
        StairPatchDataset("val"),
        batch_size   = H.BATCH_SIZE,
        shuffle      = False,
        num_workers  = 2,
        pin_memory   = True
    )

    # ─── Model, Optimizer, Scheduler, Loss, AMP ───────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = H.LR_INITIAL,
        weight_decay = H.WEIGHT_DECAY
    )
    if H.SCHEDULER["name"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = H.NUM_EPOCHS,
            eta_min = H.SCHEDULER["eta_min"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = H.LR_STEP_SIZE,
            gamma     = H.LR_GAMMA
        )
    scaler    = GradScaler(enabled=H.USE_AMP)
    w         = torch.tensor(H.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    gamma     = H.FOCAL_GAMMA  # Get gamma parameter for focal loss
    criterion = FocalCSCELoss(weight=w, gamma=gamma).to(device)

    # ─── Resume logic ─────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    best_iou    = 0.0  # Track best IoU for early stopping
    stale       = 0
    
    # If fine-tuning from a specific epoch, load that checkpoint
    if args.fine_tune and args.resume_epoch:
        source_ckpt_path = os.path.join(ckpt_dir, f"epoch_{args.resume_epoch}.pth")
        if os.path.exists(source_ckpt_path):
            ck = torch.load(source_ckpt_path, map_location=device)
            model.load_state_dict(ck["model_state"])
            # Don't load optimizer/scheduler state - we want fresh ones with new parameters
            print(f"✅ Fine-tuning from epoch {args.resume_epoch}, with adjusted parameters")
            # Reset best values since we're starting fresh tracking with new parameters
            best_val = float("inf")
            best_iou = 0.0
            start_epoch = 1  # Start counting from 1 again for the fine-tuning run
        else:
            print(f"⚠️ Checkpoint for epoch {args.resume_epoch} not found. Using best checkpoint if available.")
            if os.path.exists(ckpt_path):
                ck = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ck["model_state"])
                # Don't load optimizer/scheduler state for fine-tuning
                best_val = float("inf")
                best_iou = 0.0
                start_epoch = 1
                print(f"✅ Fine-tuning from best available checkpoint with adjusted parameters")
    # Regular resume logic
    elif os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        best_val = ck["best_val_loss"]
        # Load best_val_iou if it exists, otherwise initialize to 0
        best_iou = ck.get("best_val_iou", 0.0)
        start_epoch = ck["epoch"] + 1
        print(f"✅ Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}, best loss: {best_val:.4f}")

    # ─── Training Loop ────────────────────────────────────────────────────────
    for ep in range(start_epoch, H.NUM_EPOCHS + 1):
        # — TRAIN —
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {ep}"):
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            with autocast(enabled=H.USE_AMP):
                out   = model(batch)                        # (B, N, C)
                loss  = criterion(
                    out,
                    batch["labels"],
                    normals   = batch.get("normals", None),
                    curvature = batch.get("curvature", None),
                    epoch     = ep
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        avg_tr = tr_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_tr, ep)
        print(f"✅ Epoch {ep} Train Loss: {avg_tr:.4f}")

        # — VALIDATION —
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        all_ious = []

        with torch.no_grad():
            for b, batch in enumerate(tqdm(val_loader, desc=f"Val {ep}")):
                batch = move_to_device(batch, device)
                with autocast(enabled=H.USE_AMP):
                    out   = model(batch)
                    loss  = criterion(
                        out,
                        batch["labels"],
                        normals   = batch.get("normals", None),
                        curvature = batch.get("curvature", None),
                        epoch     = ep
                    )
                v_loss += loss.item()

                preds = out.argmax(dim=-1)
                # ─── DEBUG: see how many of each class the model is predicting ─────────
                unique, counts = torch.unique(preds, return_counts=True)
                print(f"[DEBUG] Val preds distribution:", dict(zip(unique.tolist(), counts.tolist())))
                mask  = batch["labels"] >= 0
                correct += (preds[mask] == batch["labels"][mask]).sum().item()
                total   += int(mask.sum().item())

                ious = compute_class_iou(
                    preds[mask].cpu(),
                    batch["labels"][mask].cpu(),
                    H.NUM_CLASSES
                )
                all_ious.append(ious)

                # dump PLY every 10 epochs on first batch
                if ep % 10 == 0 and b == 0:
                    coords  = batch["points"][0].cpu().numpy()
                    vis_lab = preds[0].cpu().numpy()
                    fname   = os.path.join(vis_dir, f"ep{ep:03d}_sample0.ply")
                    print(f"[DEBUG] Saving PLY to {fname}")
                    save_colored_point_cloud(coords, vis_lab, fname)

        avg_v    = v_loss / len(val_loader)
        acc      = correct / total if total > 0 else 0.0
        mean_iou = np.mean(all_ious, axis=0) if all_ious else np.zeros(H.NUM_CLASSES)

        # log scalars
        writer.add_scalar("Loss/Val",     avg_v, ep)
        writer.add_scalar("Accuracy/Val", acc,  ep)
        for cid, ci in enumerate(mean_iou):
            writer.add_scalar(f"IoU/Class_{cid}", ci, ep)

        # print metrics
        iou_str = ", ".join([f"Class {i}: {ci:.4f}" for i, ci in enumerate(mean_iou)])
        print(f"📊 Epoch {ep} Per-Class IoU → {iou_str}")
        print(f"🧪 Epoch {ep} Val Loss: {avg_v:.4f}, Accuracy: {acc*100:.2f}%")

        # Save checkpoint for every epoch
        ck = {
            "epoch":           ep,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss":   best_val,
            "best_val_iou":    best_iou
        }
        torch.save(ck, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pth"))

        # Save a checkpoint for every epoch during fine-tuning
        if args.fine_tune:
            fine_tune_ck = {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": avg_v,
                "best_val_iou": np.mean(mean_iou),
                "class_weights": H.CLASS_WEIGHTS,
                "focal_gamma": H.FOCAL_GAMMA,
                "lr_initial": H.LR_INITIAL,
            }
            torch.save(fine_tune_ck, os.path.join(ckpt_dir, f"fine_tune_epoch_{ep}.pth"))
        
        # Calculate mean IoU across all classes for early stopping
        avg_iou = np.mean(mean_iou)
        
        # Best model logic - determine whether to use loss or IoU based on config
        if H.EARLY_STOP_METRIC.lower() == "iou":
            # Use IoU for early stopping (higher is better)
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(ck, ckpt_path)
                print(f"💾 Best model updated based on IoU: {avg_iou:.4f}")
                stale = 0
            else:
                stale += 1
                print(f"⚠️ No improvement in IoU for {stale} epoch(s).")
        else:
            # Use loss for early stopping (lower is better)
            if avg_v < best_val:
                best_val = avg_v
                torch.save(ck, ckpt_path)
                print(f"💾 Best model updated based on loss: {avg_v:.4f}")
                stale = 0
            else:
                stale += 1
                print(f"⚠️ No improvement in loss for {stale} epoch(s).")

        scheduler.step()

    writer.close()
    print("🎉 Training complete. Best Val Loss:", best_val)


if __name__ == "__main__":
    main()