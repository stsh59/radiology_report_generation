import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import SiglipProcessor

from data.stage1_datamodule import Stage1DataModule
from models.stage1_contrastive import Stage1ContrastiveModel


def main():
    model_name = "google/siglip-base-patch16-224"
    processor = SiglipProcessor.from_pretrained(model_name)

    train_csv = "mimic_pa_lateral_train.csv"
    val_csv = "mimic_pa_lateral_val.csv"
    test_csv = "mimic_pa_lateral_test.csv"

    datamodule = Stage1DataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        processor=processor,
        batch_size=64,
    )

    model = Stage1ContrastiveModel(
        model_name=model_name,
        learning_rate=1e-4,
        temperature=0.07,
        lambda_img_img=0.1,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        mode="min",
        save_top_k=1,
        filename="stage1-best",
        save_last=False,
    )
    early_stop = EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=3,
        min_delta=1e-3,
    )

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    if checkpoint_callback.best_model_path:
        best_model = Stage1ContrastiveModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        output_dir = "stage1_image_encoder"
        os.makedirs(output_dir, exist_ok=True)
        best_model.save_image_encoder(output_dir)
        trainer.validate(best_model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()

