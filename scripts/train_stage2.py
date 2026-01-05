"""
Stage 2 Training Script: Generative Fine-tuning.
OPTIMIZED with stage-specific hyperparameters, early stopping, and increased epochs.
"""
import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pathlib import Path
import argparse
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import *
from data.datamodule import MultiViewDataModule
from models.generative import ReportGenLightning
from utils.logger import setup_logger

logger = setup_logger(__name__)

def _canonicalize_state_key(k: str) -> str:
    """
    Normalize checkpoint keys across Lightning + PEFT wrappers.
    Goal: compare/load weights even if prefixes differ (e.g., model.*, base_model.model.*).
    """
    prefixes = (
        "model.model.base_model.model.",
        "model.base_model.model.",
        "base_model.model.",
        "model.model.",
        "model.",
    )
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k

def main(args):
    pl.seed_everything(RANDOM_SEED, workers=True)
    
    # Explicitly create required directories
    ensure_dirs()
    
    # DataModule
    logger.info("Initializing DataModule (Stage 2)...")
    
    # Init Tokenizer for BioGPT
    from transformers import BioGptTokenizer
    tokenizer = BioGptTokenizer.from_pretrained(BIOGPT_MODEL_NAME)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    dm = MultiViewDataModule(
        train_csv=str(MIMIC_TRAIN_CSV),
        val_csv=str(MIMIC_VAL_CSV),
        test_csv=str(MIMIC_TEST_CSV),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=4,
        max_views=3,
        max_length=MAX_TEXT_LENGTH,  # 512
        image_root=str(IMAGES_DIR_MIMIC)
    )
    
    # Detect LoRA and checkpoint type BEFORE model creation
    stage1_has_lora = False
    stage1_checkpoint_data = None  # Full checkpoint dict (not just state_dict)
    vision_state_dict = None  # Vision-only weights to load
    stage1_checkpoint_provided = args.stage1_checkpoint is not None
    
    if args.stage1_checkpoint:
        checkpoint_path = Path(args.stage1_checkpoint)
        
        # === Check for merged checkpoint (preferred for clean weight transfer) ===
        merged_checkpoint = checkpoint_path.parent / "merged_for_stage2.pt"
        if merged_checkpoint.exists() and checkpoint_path != merged_checkpoint:
            logger.info(f"Found merged checkpoint: {merged_checkpoint}")
            logger.info("Using merged checkpoint for clean weight transfer (no shape mismatches)")
            args.stage1_checkpoint = str(merged_checkpoint)
            checkpoint_path = merged_checkpoint
        
        logger.info(f"Pre-loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        stage1_checkpoint_data = torch.load(args.stage1_checkpoint, map_location=DEVICE)
        
        # Detect checkpoint format
        is_merged = stage1_checkpoint_data.get('merged', False) if isinstance(stage1_checkpoint_data, dict) else False
        is_vision_only = stage1_checkpoint_data.get('vision_only', False) if isinstance(stage1_checkpoint_data, dict) else False
        
        # === NEW: Handle vision-only checkpoint (preferred, production-grade) ===
        if is_vision_only and 'vision_model_state_dict' in stage1_checkpoint_data:
            logger.info("✅ Detected vision-only Stage 1 checkpoint (expected format)")
            vision_state_dict = stage1_checkpoint_data['vision_model_state_dict']
            logger.info(f"  Loaded {len(vision_state_dict)} vision-only parameters from Stage 1")
            stage1_has_lora = False  # Merged checkpoint, no LoRA architecture needed
            
        # === Fallback: Handle legacy merged checkpoint with full state_dict ===
        elif is_merged and 'state_dict' in stage1_checkpoint_data:
            logger.info("Detected legacy merged checkpoint (full state_dict)")
            full_state = stage1_checkpoint_data['state_dict']
            # Filter to only vision_model keys
            vision_state_dict = {
                k.replace("vision_model.", ""): v 
                for k, v in full_state.items() 
                if k.startswith("vision_model.")
            }
            if not vision_state_dict:
                raise ValueError("Legacy checkpoint has no vision_model keys - cannot transfer to Stage 2")
            logger.info(f"  Extracted {len(vision_state_dict)} vision parameters from legacy checkpoint")
            stage1_has_lora = False
            
        # === Handle raw Lightning checkpoint (with potential LoRA) ===
        elif 'state_dict' in stage1_checkpoint_data:
            logger.warning("Raw Lightning checkpoint detected - may have LoRA/quantization issues")
            raw_state = stage1_checkpoint_data['state_dict']
            stage1_has_lora = any(("lora_" in k) or ("lora_A" in k) or ("lora_B" in k) for k in raw_state.keys())
            if stage1_has_lora:
                logger.info("Stage 1 checkpoint contains LoRA adapter weights. Will create model with vision_lora_enabled=True.")
            # Will use old loading path below
            vision_state_dict = None  # Signal to use legacy loader
        else:
            raise ValueError(f"Unknown checkpoint format: keys = {list(stage1_checkpoint_data.keys())[:10]}")
    
    # Model (uses config defaults for LR, warmup, Perceiver)
    # Pass vision_lora_enabled=True if Stage 1 has LoRA (this saves to hparams for checkpoint loading)
    # freeze_vision=None enables auto-detect: freeze if stage1 provided, else unfreeze
    logger.info("Initializing Model (ReportGen)...")
    
    # Determine freeze_vision: explicit flag takes precedence, otherwise auto-detect
    if args.unfreeze_vision:
        freeze_vision = False
    else:
        freeze_vision = None  # Auto-detect based on stage1_checkpoint_provided
    
    model = ReportGenLightning(
        siglip_model_name=SIGLIP_MODEL_NAME,
        biogpt_model_name=BIOGPT_MODEL_NAME,
        learning_rate=args.lr,
        warmup_steps=WARMUP_STEPS_STAGE2,
        freeze_vision=freeze_vision,
        vision_lora_enabled=stage1_has_lora,
        stage1_checkpoint_provided=stage1_checkpoint_provided
    )
    
    # Load Stage 1 checkpoint weights if provided
    if args.stage1_checkpoint and vision_state_dict is not None:
        # === NEW: Clean vision-only loading path (production-grade) ===
        logger.info(f"Loading Stage 1 vision weights from {args.stage1_checkpoint}")
        
        # Get target vision model's expected keys
        # Note: model.vision_encoder.model is SiglipModel, we need its vision_model
        target_vision_model = model.vision_encoder.model.vision_model
        target_state = target_vision_model.state_dict()
        
        # Verify all source keys are vision_model keys (no text_model contamination)
        logger.info(f"  Source keys: {len(vision_state_dict)}")
        logger.info(f"  Target keys: {len(target_state)}")
        
        # Match keys between source and target
        source_keys = set(vision_state_dict.keys())
        target_keys = set(target_state.keys())
        
        matched_keys = source_keys & target_keys
        missing_in_source = target_keys - source_keys
        extra_in_source = source_keys - target_keys
        
        match_ratio = len(matched_keys) / len(target_keys) if target_keys else 0.0
        
        logger.info(f"  Matched: {len(matched_keys)} / {len(target_keys)} ({match_ratio*100:.1f}%)")
        
        if missing_in_source:
            logger.warning(f"  Missing in source: {len(missing_in_source)} keys: {list(missing_in_source)[:10]}")
        if extra_in_source:
            logger.info(f"  Extra in source (ignored): {len(extra_in_source)} keys: {list(extra_in_source)[:5]}")
        
        # Verify shape compatibility
        shape_mismatches = []
        for k in matched_keys:
            if vision_state_dict[k].shape != target_state[k].shape:
                shape_mismatches.append(k)
        
        if shape_mismatches:
            raise ValueError(f"Shape mismatch in {len(shape_mismatches)} keys: {shape_mismatches[:5]}")
        
        # Strict load - all matched keys must load successfully
        min_match_threshold = float(args.min_match_ratio)
        if match_ratio < min_match_threshold:
            raise ValueError(
                f"Stage-1 -> Stage-2 vision weight transfer failed: "
                f"{match_ratio*100:.1f}% < required {min_match_threshold*100:.1f}%"
            )
        
        # Load weights (strict=True since we've verified compatibility)
        target_vision_model.load_state_dict(vision_state_dict, strict=True)
        logger.info("✅ Stage 1 vision weights loaded successfully (100% match, strict=True)")
        
    elif args.stage1_checkpoint and stage1_has_lora:
        # === Legacy path: LoRA checkpoint (not recommended, but supported) ===
        logger.warning("Using legacy LoRA loading path - consider using merged checkpoint instead")
        raw_state = stage1_checkpoint_data.get('state_dict', {})
        
        # Build canonical -> tensor mapping from Stage-1
        source_by_canon = {}
        for k, v in raw_state.items():
            canon = _canonicalize_state_key(k)
            if canon not in source_by_canon:
                source_by_canon[canon] = v

        # Build canonical -> actual key mapping for Stage-2 target
        target_state = model.vision_encoder.model.state_dict()
        target_keys = set(target_state.keys())
        target_by_canon = {}
        for k in target_keys:
            canon = _canonicalize_state_key(k)
            if canon not in target_by_canon:
                target_by_canon[canon] = k
            else:
                existing = target_by_canon[canon]
                if existing.startswith("base_model.model.") is False and k.startswith("base_model.model."):
                    target_by_canon[canon] = k

        source_canons = set(source_by_canon.keys())
        target_canons = set(target_by_canon.keys())

        matched_canons = target_canons & source_canons

        logger.info("Weight transfer analysis (Stage1 -> Stage2 vision):")
        logger.info(f"  Target keys: {len(target_keys)} (canonical={len(target_canons)})")
        logger.info(f"  Source keys: {len(raw_state)} (canonical={len(source_canons)})")
        logger.info(f"  Matched: {len(matched_canons)}")

        # Load matched weights (skip shape mismatches from quantized checkpoint)
        filtered_state = {}
        skipped_shape_mismatch = 0
        for canon in matched_canons:
            tgt_key = target_by_canon[canon]
            src_tensor = source_by_canon[canon]
            tgt_shape = target_state[tgt_key].shape
            
            if src_tensor.shape != tgt_shape:
                skipped_shape_mismatch += 1
                continue
            
            filtered_state[tgt_key] = src_tensor

        if skipped_shape_mismatch > 0:
            logger.info(f"  Skipped {skipped_shape_mismatch} keys due to shape mismatch (quantized)")

        lora_loaded = sum(1 for k in filtered_state if 'lora_' in k)
        logger.info(f"  LoRA adapters loaded: {lora_loaded}")
        
        if lora_loaded == 0:
            raise ValueError("No LoRA adapter keys loaded from Stage 1 checkpoint")

        model.vision_encoder.model.load_state_dict(filtered_state, strict=False)
        logger.info("✅ Stage 1 LoRA weights loaded (legacy path)")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR / "stage2_generative",
        filename="reportgen-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3  # Save more for generation
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        mode='min',
        verbose=True
    )
    
    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        gradient_clip_val=1.0,
        accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS_STAGE2,
        log_every_n_steps=LOG_INTERVAL
    )
    
    logger.info("Starting Training...")
    trainer.fit(model, datamodule=dm)
    
    logger.info(f"Training complete. Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_STAGE2)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_STAGE2)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_STAGE2)
    parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Path to Stage 1 checkpoint")
    parser.add_argument("--unfreeze_vision", action="store_true", help="Unfreeze SigLIP")
    parser.add_argument("--min_match_ratio", type=float, default=0.95, help="Minimum matched/expected key ratio for transfer verification")
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow training to proceed even if transfer verification fails")
    parser.add_argument("--verify_param_norm", action="store_true", help="Also check parameter norm change to detect no-op loads")
    parser.add_argument("--strict_load", action="store_true", help="Do not allow override even if --allow_partial_load is set")
    args = parser.parse_args()
    
    main(args)
