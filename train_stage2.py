import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import SiglipProcessor, AutoTokenizer

from data.stage2_datamodule import Stage2DataModule
from models.stage2_generator import Stage2GeneratorModel


def main():
    model_name = "google/siglip-base-patch16-224"
    lm_name = "microsoft/biogpt-large"
    lora_path = "stage1_image_encoder"

    processor = SiglipProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_csv = "mimic_pa_lateral_train.csv"
    val_csv = "mimic_pa_lateral_val.csv"
    test_csv = "mimic_pa_lateral_test.csv"

    datamodule = Stage2DataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=8,
    )

    model = Stage2GeneratorModel(
        model_name=model_name,
        lora_path=lora_path,
        lm_name=lm_name,
        learning_rate=5e-5,
        num_latents=32,
        latent_dim=768,
        num_cross_attn_layers=2,
        num_cross_attn_heads=8,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/lm_loss",
        mode="min",
        save_top_k=1,
        filename="stage2-best",
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val/lm_loss",
        mode="min",
        patience=4,
        min_delta=1e-3,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    if checkpoint_callback.best_model_path:
        trainer.validate(datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()

