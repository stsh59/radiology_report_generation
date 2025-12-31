"""
Stage 2 Training Script: Generative Fine-tuning.
OPTIMIZED with stage-specific hyperparameters, early stopping, and increased epochs.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pathlib import Path
import argparse
import torch

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
    
    # Model (uses config defaults for LR, warmup, Perceiver)
    logger.info("Initializing Model (ReportGen)...")
    model = ReportGenLightning(
        siglip_model_name=SIGLIP_MODEL_NAME,
        biogpt_model_name=BIOGPT_MODEL_NAME,
        learning_rate=args.lr,
        warmup_steps=WARMUP_STEPS_STAGE2,
        freeze_vision=not args.unfreeze_vision
    )
    
    # Load Stage 1 checkpoint if provided
    if args.stage1_checkpoint:
        logger.info(f"Loading Stage 1 weights from {args.stage1_checkpoint}")
        stage1_state = torch.load(args.stage1_checkpoint, map_location=DEVICE)
        if 'state_dict' in stage1_state:
            stage1_state = stage1_state['state_dict']

        # Detect whether the Stage-1 checkpoint contains LoRA adapter weights
        stage1_has_lora = any(("lora_" in k) or ("lora_A" in k) or ("lora_B" in k) for k in stage1_state.keys())
        if stage1_has_lora:
            logger.info("Stage 1 checkpoint appears to contain LoRA adapter weights. Wrapping Stage 2 vision encoder with LoRA before loading...")
            from models.peft_config import get_lora_config, apply_lora
            lora_config = get_lora_config(model_type="vision")
            model.vision_encoder.model = apply_lora(model.vision_encoder.model, lora_config)

        # Build canonical -> tensor mapping from Stage-1
        source_by_canon = {}
        for k, v in stage1_state.items():
            canon = _canonicalize_state_key(k)
            # Prefer first occurrence; duplicates are usually identical wrappers
            if canon not in source_by_canon:
                source_by_canon[canon] = v

        # Build canonical -> actual key mapping for Stage-2 target
        target_state = model.vision_encoder.model.state_dict()
        target_keys = set(target_state.keys())
        target_by_canon = {}
        for k in target_keys:
            canon = _canonicalize_state_key(k)
            # Prefer base_model.model.* keys if present (PEFT models)
            if canon not in target_by_canon:
                target_by_canon[canon] = k
            else:
                existing = target_by_canon[canon]
                if existing.startswith("base_model.model.") is False and k.startswith("base_model.model."):
                    target_by_canon[canon] = k

        source_canons = set(source_by_canon.keys())
        target_canons = set(target_by_canon.keys())

        matched_canons = target_canons & source_canons
        missing_canons = target_canons - source_canons
        extra_canons = source_canons - target_canons

        loaded_ratio = len(matched_canons) / len(target_canons) if target_canons else 0.0

        # LoRA verification (if target has LoRA keys, require some to match)
        target_lora_canons = {c for c in target_canons if ("lora_" in c) or ("lora_A" in c) or ("lora_B" in c)}
        matched_lora_canons = matched_canons & target_lora_canons

        logger.info("Weight transfer analysis (Stage1 -> Stage2 vision):")
        logger.info(f"  Target keys: {len(target_keys)} (canonical={len(target_canons)})")
        logger.info(f"  Source keys: {len(stage1_state)} (canonical={len(source_canons)})")
        logger.info(f"  Matched: {len(matched_canons)} ({loaded_ratio*100:.1f}%)")
        if target_lora_canons:
            logger.info(f"  LoRA keys (target): {len(target_lora_canons)}, matched LoRA: {len(matched_lora_canons)}")

        if missing_canons:
            missing_preview = sorted(list(missing_canons))[:20]
            logger.warning(f"  Missing in source: {len(missing_canons)} keys (showing up to 20): {missing_preview}")
        if extra_canons:
            extra_preview = sorted(list(extra_canons))[:20]
            logger.info(f"  Extra in source (ignored): {len(extra_canons)} keys (showing up to 20): {extra_preview}")

        # Verification gates (fail-fast by default)
        min_match_threshold = float(args.min_match_ratio)
        failures = []
        if loaded_ratio < min_match_threshold:
            failures.append(f"matched/expected {loaded_ratio*100:.1f}% < required {min_match_threshold*100:.1f}%")
        if stage1_has_lora and target_lora_canons and len(matched_lora_canons) == 0:
            failures.append("no LoRA adapter keys matched (stage1 has LoRA, but none transferred to stage2 vision)")

        # Optional parameter norm check (helps catch 'loaded nothing' cases)
        norm_before = None
        norm_after = None
        if args.verify_param_norm:
            norm_before = sum(p.norm().item() for p in model.vision_encoder.model.parameters())

        # Load matched weights into target actual keys
        filtered_state = {}
        for canon in matched_canons:
            tgt_key = target_by_canon[canon]
            filtered_state[tgt_key] = source_by_canon[canon]

        incompatible = model.vision_encoder.model.load_state_dict(filtered_state, strict=False)

        if args.verify_param_norm:
            norm_after = sum(p.norm().item() for p in model.vision_encoder.model.parameters())
            if norm_before is not None and norm_after is not None:
                if abs(norm_after - norm_before) < 0.01:
                    failures.append("parameter norms unchanged after load (possible no-op load)")
                else:
                    logger.info(f"  Norm change: {norm_before:.2f} -> {norm_after:.2f} (Δ={norm_after-norm_before:.2f})")

        # Log load_state_dict incompatibilities (bounded)
        if getattr(incompatible, "missing_keys", None):
            mk = list(incompatible.missing_keys)
            if mk:
                logger.warning(f"  load_state_dict missing_keys: {len(mk)} (showing up to 20): {mk[:20]}")
        if getattr(incompatible, "unexpected_keys", None):
            uk = list(incompatible.unexpected_keys)
            if uk:
                logger.warning(f"  load_state_dict unexpected_keys: {len(uk)} (showing up to 20): {uk[:20]}")

        if failures:
            error_msg = "Stage-1 -> Stage-2 weight transfer verification failed: " + "; ".join(failures)
            if args.allow_partial_load and not args.strict_load:
                logger.warning(error_msg + " - Proceeding due to --allow_partial_load.")
            else:
                raise ValueError(error_msg + " - Aborting training (use --allow_partial_load to override).")

        logger.info("✅ Stage 1 vision weights (and adapters, if present) loaded successfully.")

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
