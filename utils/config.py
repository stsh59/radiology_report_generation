"""
Global configuration and paths for the Medical-SigLIP project.
OPTIMIZED VERSION with tuned hyperparameters.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
# DATA_ROOT: Override via DATA_ROOT env var; default is original path for backward compatibility
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/home/sb2ek/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2"
))
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data paths - MIMIC-CXR Multi-View
IMAGES_DIR_MIMIC = DATA_ROOT / "mimic-cxr-dataset"
MIMIC_TRAIN_CSV = PROJECT_ROOT / "multiview_train.csv"
MIMIC_VAL_CSV = PROJECT_ROOT / "multiview_val.csv"
MIMIC_TEST_CSV = PROJECT_ROOT / "multiview_test.csv"

# Model configurations
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
BIOGPT_MODEL_NAME = "microsoft/biogpt"

# Image preprocessing - CORRECTED to ImageNet values (SigLIP pretrained on ImageNet)
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Text preprocessing - INCREASED for longer reports
MAX_TEXT_LENGTH = 512
MAX_GENERATION_LENGTH = 512

# Training hyperparameters - STAGE-SPECIFIC
BATCH_SIZE = 4
BATCH_SIZE_STAGE1 = 8  # Larger for contrastive (more negatives)
BATCH_SIZE_STAGE2 = 4  # Smaller for generation (memory)

LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 5e-5  # Lower for generation stability

NUM_EPOCHS_STAGE1 = 10
NUM_EPOCHS_STAGE2 = 25  # More epochs for generation

WARMUP_STEPS_STAGE1 = 500
WARMUP_STEPS_STAGE2 = 2000  # Longer warmup for generation

WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS_STAGE1 = 8  # Effective batch 64
GRADIENT_ACCUMULATION_STEPS_STAGE2 = 4

# PEFT configurations - SEPARATE for vision and LLM
LORA_R_VISION = 16
LORA_ALPHA_VISION = 32
LORA_R_LLM = 64        # Higher capacity for text generation
LORA_ALPHA_LLM = 128   # Scaled with rank
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES_VISION = ["q_proj", "v_proj", "k_proj", "out_proj"]
LORA_TARGET_MODULES_LLM = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

# Legacy compatibility
LORA_R = LORA_R_VISION
LORA_ALPHA = LORA_ALPHA_VISION

# Perceiver Resampler - OPTIMIZED
NUM_PROJECTION_QUERIES = 64    # Increased from 32
PROJECTION_LAYERS = 4          # Increased from 2
PROJECTION_HEADS = 8
PROJECTION_DROPOUT = 0.15      # Slightly higher for regularization

# Generation parameters - PURE BEAM SEARCH (Deterministic for medical applications)
GENERATION_NUM_BEAMS = 5           # Slightly more beams for better quality
GENERATION_DO_SAMPLE = False       # Deterministic - no sampling
GENERATION_REPETITION_PENALTY = 1.2
GENERATION_NO_REPEAT_NGRAM_SIZE = 3
GENERATION_LENGTH_PENALTY = 1.0    # Neutral length preference
GENERATION_EARLY_STOPPING = True

# Label smoothing
LABEL_SMOOTHING = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 3

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration - use torch availability, not just env var
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 1


def ensure_dirs():
    """
    Explicitly create required directories.
    Call this from training scripts, NOT on import.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)