import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import random
import numpy as np

from data.dataset import MultiViewDataset
from utils.logger import setup_logger
from utils.config import RANDOM_SEED

logger = setup_logger(__name__)


def _worker_init_fn(worker_id: int):
    """Seed Python random and numpy per worker for reproducibility."""
    seed = RANDOM_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


class MultiViewDataModule(pl.LightningDataModule):
    """
    DataModule for Multi-View dataset (MIMIC-CXR).
    """
    def __init__(
        self,
        train_csv: str = "multiview_train.csv",
        val_csv: str = "multiview_val.csv",
        test_csv: str = "multiview_test.csv",
        tokenizer=None,
        batch_size: int = 4,
        num_workers: int = 4,
        max_views: int = 3,
        max_length: int = 256,
        image_root: str = "mimic-cxr-dataset",
        use_span_sampling: bool = False  # For Stage-1 random span sampling
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_views = max_views
        self.max_length = max_length
        self.image_root = image_root
        self.use_span_sampling = use_span_sampling
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiViewDataset(
                self.train_csv,
                self.tokenizer,
                max_views=self.max_views,
                max_length=self.max_length,
                image_root=self.image_root,
                use_span_sampling=self.use_span_sampling,  # Enable for train only
                is_train=True  # Explicit flag for train transforms
            )
            self.val_dataset = MultiViewDataset(
                self.val_csv,
                self.tokenizer,
                max_views=self.max_views,
                max_length=self.max_length,
                image_root=self.image_root,
                is_train=False  # Explicit flag for val transforms
            )
            
        if stage == 'test':
            self.test_dataset = MultiViewDataset(
                self.test_csv,
                self.tokenizer,
                max_views=self.max_views,
                max_length=self.max_length,
                image_root=self.image_root,
                is_train=False  # Explicit flag for test transforms
            )
            
    def _get_pathology_labels(self):
        """
        Detect pathological vs normal reports using multi-keyword heuristic.
        Returns: list of 0 (normal) or 1 (pathological)
        
        Classification rules (classify BEFORE span sampling on FULL report):
        - "normal" if contains: "no acute", "unremarkable", "normal", "clear", "no evidence"
        - "pathological" if contains: "opacity", "effusion", "consolidation", "mass", "nodule", "pneumonia"
        - Default: normal if keywords match, else pathological
        """
        normal_keywords = ['no acute', 'unremarkable', 'normal', 'clear', 'no evidence', 'stable']
        pathology_keywords = ['opacity', 'effusion', 'consolidation', 'mass', 'nodule', 
                             'pneumonia', 'edema', 'cardiomegaly', 'enlarged', 'abnormal']
        
        labels = []
        for idx, row in self.train_dataset.data.iterrows():
            report = str(row.get('report_text', '')).lower()
            
            # Check for pathology keywords first (more specific)
            has_pathology = any(kw in report for kw in pathology_keywords)
            has_normal = any(kw in report for kw in normal_keywords)
            
            # If has pathology keywords, classify as pathological
            if has_pathology and not has_normal:
                labels.append(1)  # Pathological
            elif has_normal and not has_pathology:
                labels.append(0)  # Normal
            elif has_pathology:  # Both present, prioritize pathology
                labels.append(1)
            else:
                labels.append(0)  # Default to normal
        
        return labels
            
    def train_dataloader(self):
        # Use WeightedRandomSampler for class balancing
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            labels = self._get_pathology_labels()
            
            # Calculate class weights (inverse frequency)
            from collections import Counter
            import torch
            counts = Counter(labels)
            total = len(labels)
            
            # Weight: total / (num_classes * count_per_class)
            weight_normal = total / (2 * counts.get(0, 1))
            weight_pathology = total / (2 * counts.get(1, 1))

            # Oversampling cap: limit relative sampling weight to avoid extreme duplication (default 5×)
            oversample_cap = 5.0
            if weight_pathology > 0 and weight_normal > 0:
                ratio = weight_pathology / weight_normal
                if ratio > oversample_cap:
                    weight_pathology = oversample_cap * weight_normal
                elif ratio < (1.0 / oversample_cap):
                    weight_normal = oversample_cap * weight_pathology
            
            sample_weights = [weight_pathology if l == 1 else weight_normal for l in labels]
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            # Log class distribution
            from utils.logger import setup_logger
            logger = setup_logger(__name__)
            logger.info(f"Class distribution: Normal={counts.get(0,0)}, Pathology={counts.get(1,0)}")
            logger.info(f"Sample weights (capped): Normal={weight_normal:.2f}, Pathology={weight_pathology:.2f}, cap={oversample_cap:.1f}x")
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,  # Use weighted sampler instead of shuffle
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=_worker_init_fn
            )
        
        # Fallback if dataset not ready
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn
        )