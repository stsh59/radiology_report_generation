from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import SiglipProcessor


class Stage1Dataset(Dataset):
    def __init__(self, csv_path: str, processor: SiglipProcessor) -> None:
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_pa = self._load_image(row["image_pa"])
        image_lat = self._load_image(row["image_lateral"])

        pa_tensor = self.image_processor(images=image_pa, return_tensors="pt")["pixel_values"].squeeze(0)
        lat_tensor = self.image_processor(images=image_lat, return_tensors="pt")["pixel_values"].squeeze(0)

        text = row["report_text"]
        text_tokens = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
            return_tensors=None,
        )

        return {
            "pixel_values_pa": pa_tensor,
            "pixel_values_lateral": lat_tensor,
            "input_ids": torch.tensor(text_tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(text_tokens["attention_mask"], dtype=torch.long),
        }


def _collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    pixel_values_pa = torch.stack([item["pixel_values_pa"] for item in batch], dim=0)
    pixel_values_lat = torch.stack([item["pixel_values_lateral"] for item in batch], dim=0)

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    max_len = max(x.shape[0] for x in input_ids)
    padded_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        length = ids.shape[0]
        padded_ids[i, :length] = ids
        padded_mask[i, :length] = mask

    return {
        "pixel_values_pa": pixel_values_pa,
        "pixel_values_lateral": pixel_values_lat,
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
    }


class Stage1DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        processor: SiglipProcessor,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Ensure a valid pad token id exists
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        self.pad_token_id = processor.tokenizer.pad_token_id

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = Stage1Dataset(self.train_csv, self.processor)
            self.val_dataset = Stage1Dataset(self.val_csv, self.processor)
        if stage == "test" or stage is None:
            self.test_dataset = Stage1Dataset(self.test_csv, self.processor)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: _collate_fn(batch, self.pad_token_id),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: _collate_fn(batch, self.pad_token_id),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: _collate_fn(batch, self.pad_token_id),
        )

