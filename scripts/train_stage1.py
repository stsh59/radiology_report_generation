"""
Stage 1 Training Script: Contrastive Pretraining.
OPTIMIZED with stage-specific hyperparameters and early stopping.
"""
import sys
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pathlib import Path
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import *
from data.datamodule import MultiViewDataModule
from models.contrastive import MedicalSigLIPLightning
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main(args):
    pl.seed_everything(RANDOM_SEED, workers=True)
    
    # Explicitly create required directories
    ensure_dirs()
    
    # DataModule
    logger.info("Initializing DataModule (Stage 1)...")
    
    # Init Tokenizer for SigLIP (limit 64)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dm = MultiViewDataModule(
        train_csv=str(MIMIC_TRAIN_CSV),
        val_csv=str(MIMIC_VAL_CSV),
        test_csv=str(MIMIC_TEST_CSV),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=4,
        max_views=3,
        max_length=64,  # SigLIP limitation
        image_root=str(IMAGES_DIR_MIMIC),
        use_span_sampling=True  # Random span sampling for better contrastive alignment
    )
    
    # Model
    logger.info("Initializing Model (MedicalSigLIP)...")
    model = MedicalSigLIPLightning(
        model_name=SIGLIP_MODEL_NAME,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS_STAGE1,
        temperature=TEMPERATURE_STAGE1,
        use_qlora=args.use_qlora
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR / "stage1_contrastive",
        filename="siglip-contrastive-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Early Stopping - prevents overfitting
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
        accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS_STAGE1,
        log_every_n_steps=LOG_INTERVAL
    )
    
    logger.info("Starting Training...")
    trainer.fit(model, datamodule=dm)
    
    logger.info(f"Training complete. Best model path: {checkpoint_callback.best_model_path}")
    
    # === Merge LoRA and save VISION-ONLY weights for Stage 2 ===
    # Stage 2 only uses SigLIP's vision_model (BioGPT handles text).
    # Saving only vision_model avoids quantization issues with text_model.
    logger.info("Merging LoRA adapters and extracting vision-only weights for Stage 2...")
    try:
        merged_model = model.model.merge_and_unload()
        
        # Extract ONLY vision_model weights (Stage 2 doesn't use text_model)
        vision_state = merged_model.vision_model.state_dict()
        
        # Verify no text_model keys leaked in
        assert all(not k.startswith("text_model.") for k in vision_state.keys()), \
            "Vision checkpoint contains unexpected text_model keys"
        
        logger.info(f"Extracted {len(vision_state)} vision-only parameters")
        
        merged_checkpoint_path = CHECKPOINT_DIR / "stage1_contrastive" / "merged_for_stage2.pt"
        torch.save({
            'vision_model_state_dict': vision_state,  # Vision-only (clean transfer)
            'hparams': dict(model.hparams),
            'merged': True,
            'vision_only': True,  # Flag for Stage 2 detection
        }, merged_checkpoint_path)
        logger.info(f"Saved vision-only checkpoint to {merged_checkpoint_path}")
        logger.info("Use this checkpoint for Stage 2 to preserve visual adaptations.")
    except Exception as e:
        logger.warning(f"Could not merge LoRA: {e}")
        logger.warning("Stage 2 may not receive full Stage 1 learning.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_STAGE1)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_STAGE1)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_STAGE1)
    parser.add_argument("--use_qlora", action="store_true", 
                        help="Use 4-bit quantization (not recommended - breaks Stage 2 transfer)")
    args = parser.parse_args()
    
    main(args)
