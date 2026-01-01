import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from torchvision import transforms
from ast import literal_eval
from utils.logger import setup_logger
from utils.config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

logger = setup_logger(__name__)

class MultiViewDataset(Dataset):
    """
    Robust Dataset for Multi-View Chest X-ray Report Generation.
    Source: MIMIC-CXR derived CSVs.
    
    Args:
        use_span_sampling: If True, sample random spans for short max_length (Stage-1).
                          Only applies to training. Val/test use first N tokens.
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_views: int = 3,
        max_length: int = 256,
        transform: Optional[transforms.Compose] = None,
        image_root: str = "mimic-cxr-dataset",
        use_span_sampling: bool = False,
        is_train: Optional[bool] = None
    ):
        self.tokenizer = tokenizer
        self.max_views = max_views
        self.max_length = max_length
        self.image_root = Path(image_root)
        self.use_span_sampling = use_span_sampling
        # Rate-limit warnings to avoid log spam
        self._image_load_warning_count = 0
        self._max_image_load_warnings = 20
        
        # Transforms: Medical-appropriate augmentation for training
        # Use explicit is_train if provided; fallback to filename heuristic for backward compatibility
        if is_train is not None:
            self.is_train = is_train
        else:
            self.is_train = 'train' in str(csv_path).lower()
        
        if transform is not None:
            self.transform = transform
        elif self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Larger for random crop
                transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),  # Anatomically valid for X-rays
                transforms.RandomRotation(degrees=10),   # Small rotations
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
            ])
        else:
            # Validation/Test: No augmentation, just resize
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
            ])
        
        # Load Data
        logger.info(f"Loading dataset from {csv_path}")
        self.data = pd.read_csv(csv_path)
        
        # Robust parsing of list columns
        list_cols = ['image_paths', 'view_positions']
        for col in list_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(self._safe_literal_eval)
        
        # Filter rows with no images or empty reports if necessary
        # (Assuming CSV is largely clean, but safety check is good)
        self.data = self.data[self.data['image_paths'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        self.data = self.data.reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.data)} samples.")

    def _safe_literal_eval(self, x):
        if isinstance(x, list): return x
        try:
            return literal_eval(str(x))
        except (ValueError, SyntaxError):
            return []

    def __len__(self):
        return len(self.data)

    def _get_images(self, image_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load, transform, and pad images.
        Returns:
            pixel_values: (Max_Views, 3, H, W)
            view_mask: (Max_Views) -> 1=Valid, 0=Pad
        """
        images = []
        view_mask = []

        for path in image_paths[:self.max_views]:
            full_path = self.image_root / path
            
            # Smart path checking
            if not full_path.exists():
                # Try relative to cwd
                if (Path.cwd() / path).exists():
                    full_path = Path.cwd() / path
            
            try:
                if not full_path.exists():
                    raise FileNotFoundError(str(full_path))
                img = Image.open(full_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                view_mask.append(1)
            except Exception as e:
                # Use placeholder (black image) but mark this view as invalid to avoid silent corruption
                if self._image_load_warning_count < self._max_image_load_warnings:
                    logger.warning(f"Failed to load image {full_path}: {e}")
                    self._image_load_warning_count += 1
                images.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))
                view_mask.append(0)

        # Padding to max_views
        while len(images) < self.max_views:
            images.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))
            view_mask.append(0)
            
        pixel_values = torch.stack(images)
        view_mask_tensor = torch.tensor(view_mask, dtype=torch.long)
        
        return pixel_values, view_mask_tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Process Images
        image_paths = row['image_paths']
        pixel_values, view_mask = self._get_images(image_paths)

        # Strict policy: if no views loaded successfully, this sample is not usable.
        # Use KeyError so DataLoader will surface the issue clearly.
        if int(view_mask.sum().item()) == 0:
            raise KeyError(f"No valid images could be loaded for index={idx}")
        
        # 1b. Process view positions (per-view categorical IDs)
        # Mapping: PAD=0, PA=1, AP=2, LATERAL=3, OTHER=4
        raw_view_positions = row.get('view_positions', [])
        if not isinstance(raw_view_positions, list):
            raw_view_positions = []

        # Validate alignment between paths and positions (log only; do not crash by default)
        if isinstance(image_paths, list) and raw_view_positions and (len(raw_view_positions) != len(image_paths)):
            if self._image_load_warning_count < self._max_image_load_warnings:
                logger.warning(
                    f"view_positions length ({len(raw_view_positions)}) != image_paths length ({len(image_paths)}) "
                    f"for index={idx}. Using per-slot masking/PAD mapping."
                )
                self._image_load_warning_count += 1
        
        view_pos_ids = []
        for i in range(self.max_views):
            # If the view failed to load (view_mask=0), treat position as PAD for safety.
            if int(view_mask[i].item()) == 0:
                view_pos_ids.append(0)  # PAD
                continue
            
            v = raw_view_positions[i] if i < len(raw_view_positions) else "OTHER"
            v = str(v).strip().upper()
            if v == "PA":
                view_pos_ids.append(1)
            elif v == "AP":
                view_pos_ids.append(2)
            elif v == "LATERAL":
                view_pos_ids.append(3)
            else:
                view_pos_ids.append(4)  # OTHER
        
        view_positions = torch.tensor(view_pos_ids, dtype=torch.long)
        
        # 2. Process Text
        # Prefer 'report_text', fallback to concatenation if needed
        report = row.get('report_text', "")
        
        if pd.isna(report) or str(report).strip() == "":
            # Fallback logic if columns existed (findings/impression)
            findings = str(row.get('findings', '')) if pd.notna(row.get('findings')) else ''
            impression = str(row.get('impression', '')) if pd.notna(row.get('impression')) else ''
            report = f"Findings: {findings} Impression: {impression}".strip()

        # Tokenize - with optional span sampling for Stage-1
        report_str = str(report)
        
        if self.use_span_sampling:
            # Span sampling operates on the FULL tokenized report first, then slices.
            # - Train: random span (improves contrastive alignment)
            # - Val/Test: deterministic span (start at 0) for reproducibility
            # Use add_special_tokens=False to get raw tokens, then add BOS/EOS manually if needed
            # Truncate to a large limit first to avoid warning, then slice to max_length
            full_encoded = self.tokenizer(
                report_str,
                truncation=True,
                max_length=4096,  # Large limit to capture full report without warning
                return_tensors="pt",
                add_special_tokens=False  # Handle special tokens after slicing
            )
            full_ids = full_encoded['input_ids'].squeeze(0)
            
            # Reserve space for BOS/EOS tokens (SigLIP uses them)
            effective_max = self.max_length - 2  # Reserve 2 for special tokens
            
            if len(full_ids) > effective_max:
                max_start = len(full_ids) - effective_max
                if self.is_train:
                    import random
                    start_idx = random.randint(0, max_start)
                else:
                    start_idx = 0
                sliced_ids = full_ids[start_idx:start_idx + effective_max]
            else:
                sliced_ids = full_ids
            
            # Add BOS and EOS tokens
            bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
            eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # Build final sequence: [BOS] + tokens + [EOS] + [PAD...]
            tokens_list = []
            if bos_id is not None:
                tokens_list.append(bos_id)
            tokens_list.extend(sliced_ids.tolist())
            if eos_id is not None:
                tokens_list.append(eos_id)
            
            # Pad to max_length
            if len(tokens_list) < self.max_length:
                pad_len = self.max_length - len(tokens_list)
                tokens_list.extend([pad_id] * pad_len)
            
            input_ids = torch.tensor(tokens_list[:self.max_length], dtype=torch.long)
        else:
            # Standard tokenization: first max_length tokens (deterministic)
            text_encoded = self.tokenizer(
                report_str,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = text_encoded['input_ids'].squeeze(0)
        
        # Create attention mask
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_id).long()
        
        # For Causal LM (BioGPT), labels are input_ids
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
             labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,  # (3, 3, 224, 224)
            'view_mask': view_mask,        # (3,)
            'view_positions': view_positions,  # (3,) categorical IDs
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'study_id': str(row.get('study_id', idx))
        }

