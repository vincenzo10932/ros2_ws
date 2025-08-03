# hyperparameters.py
import numpy as np

# ─── Data Sampling ───────────────────────────────────────────────────────────────
TRAIN_N_POINTS      = 16384
VAL_N_POINTS        = TRAIN_N_POINTS
PATCHES_PER_SCAN    = 50     # yields 26×50 = 1300 train patches per epoch

# ─── Classes & Weights ──────────────────────────────────────────────────────────
NUM_CLASSES         = 3      # {0: Background, 1: Riser, 2: Tread}
CLASS_WEIGHTS       = [
    10.0,  # Background
    1.54,  # Riser
    4.00  # Tread
]

# ─── Architecture ────────────────────────────────────────────────────────────────
D_OUT               = [8, 64, 128, 256, 512, 512]  
NUM_LAYERS          = len(D_OUT)
K_N                 = 16     # k in k-NN
SUB_SAMPLING_RATIO  = [4, 4, 4, 2, 2, 2] # downsampling ratios per layer

# ─── Optimization ────────────────────────────────────────────────────────────────
OPTIMIZER    = "adam"
LR_INITIAL   = 1e-3
WEIGHT_DECAY = 1e-4

# ─── Training Loop ───────────────────────────────────────────────────────────────
BATCH_SIZE          = 16
NUM_EPOCHS          = 100
EARLY_STOP_PATIENCE = 10     # stop if no IoU gain ≥0.5% for 10 epochs
USE_AMP             = True

# ─── Scheduler ───────────────────────────────────────────────────────────────────
SCHEDULER = {
    "name":   "cosine", 
    "T_max":  NUM_EPOCHS,
    "eta_min":1e-5
}

# ─── CSCE Warm‑up & Geometry Loss Weights ────────────────────────────────────────
CSCE_WARMUP = 30    # epochs before adding geometry terms
CSCE_ALPHA  = 0.05  # weight for normal‑alignment loss
CSCE_BETA   = 0.05  # weight for curvature loss

# ─── Augmentations ───────────────────────────────────────────────────────────────
P_RGP      = 0.5;    RGP_SIGMA  = 0.001
P_RRS      = 0.5;    RRS_MEAN   = 2.0;  RRS_STD    = 1.0
P_RRNS     = 0.5;    RRNS_FRAC  = 0.3
P_RBS      = 0.5;    RBS_RADIUS = [0.1, 0.3]
P_RRPG     = 0.5;    RRPG_RATIO = [0.5, 1.0]
P_BG_PATCH = 0.0

Z_JITTER    = 0.2    # meters
PITCH_RANGE = np.pi / 36  # ±5°

# ─── Inference / OctoMap ─────────────────────────────────────────────────────────
INFER_CHUNK_SIZE   = TRAIN_N_POINTS  # 16,384 - process all points for 14,367 dataset
OCTOMAP_CHUNK_SIZE = 6000   # Reduced for faster octree operations
OCTOMAP_RESOLUTION = 0.05  # meters
OCC_THRESHOLD      = 0.5   # occupancy probability threshold
