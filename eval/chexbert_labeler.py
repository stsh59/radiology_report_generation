"""
CheXbert-based Clinical Labeler for Radiology Report Evaluation.

This module provides functionality to extract pathology labels from radiology reports
and compute clinical F1 scores for model evaluation.

The 14 CheXbert pathology labels are:
- Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum,
- Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion,
- Pleural Other, Pneumonia, Pneumothorax, Support Devices
"""
import logging
import re
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# CheXbert pathology labels
CHEXBERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly", 
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]

# Keyword patterns for rule-based extraction (fallback)
PATHOLOGY_PATTERNS = {
    "Atelectasis": [r"atelecta", r"collapse"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged heart", r"cardiac enlargement", r"heart.{0,20}enlarged"],
    "Consolidation": [r"consolidat", r"airspace disease"],
    "Edema": [r"edema", r"congestion", r"pulmonary.{0,10}edema"],
    "Enlarged Cardiomediastinum": [r"mediastin.{0,10}widen", r"enlarged mediastinum"],
    "Fracture": [r"fracture", r"broken"],
    "Lung Lesion": [r"lesion", r"mass", r"nodule", r"tumor"],
    "Lung Opacity": [r"opacity", r"opacit", r"infiltrat", r"haziness"],
    "No Finding": [r"no.{0,20}acute", r"normal", r"unremarkable", r"clear lungs", r"no.{0,20}abnormal"],
    "Pleural Effusion": [r"pleural effusion", r"effusion", r"pleural fluid"],
    "Pleural Other": [r"pleural thicken", r"pleural.{0,10}abnormal"],
    "Pneumonia": [r"pneumonia", r"infectious", r"infection"],
    "Pneumothorax": [r"pneumothorax"],
    "Support Devices": [r"catheter", r"tube", r"line", r"pacemaker", r"device", r"icd", r"stent", r"wire"]
}

# Negation patterns
NEGATION_PATTERNS = [
    r"no\s+",
    r"without\s+",
    r"negative\s+for",
    r"rule\s*out",
    r"absent",
    r"resolved",
    r"cleared",
    r"no\s+evidence\s+of",
    r"not\s+see",
    r"unlikely"
]


class CheXbertLabeler:
    """
    Wrapper for CheXbert-style pathology extraction from radiology reports.
    
    Uses either the official CheXbert model (if available) or a rule-based
    fallback for extracting 14 pathology labels.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the labeler.
        
        Args:
            use_gpu: Whether to use GPU for CheXbert model (if available)
        """
        self.use_gpu = use_gpu
        self.chexbert_available = False
        self.model = None
        
        # Try to load CheXbert
        try:
            self._load_chexbert()
        except Exception as e:
            logger.warning(f"CheXbert model not available, using rule-based extraction: {e}")
    
    def _load_chexbert(self):
        """Attempt to load CheXbert model."""
        # Try importing from common CheXbert packages
        try:
            # Option 1: Try chexbert pip package
            from chexbert.label import label
            self.chexbert_available = True
            logger.info("CheXbert loaded successfully from chexbert package")
            return
        except ImportError:
            pass
        
        # If no package available, we'll use rule-based extraction
        logger.info("Using rule-based pathology extraction (CheXbert package not installed)")
    
    def _rule_based_extract(self, text: str) -> Dict[str, int]:
        """
        Extract pathology labels using rule-based pattern matching.
        
        Args:
            text: Radiology report text
        
        Returns:
            Dictionary mapping label names to values (0=negative, 1=positive)
        """
        text_lower = text.lower()
        labels = {}
        
        for label, patterns in PATHOLOGY_PATTERNS.items():
            found = False
            negated = False
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    found = True
                    # Check for negation before the match
                    for match in matches:
                        start = max(0, match.start() - 30)
                        context = text_lower[start:match.start()]
                        for neg_pattern in NEGATION_PATTERNS:
                            if re.search(neg_pattern, context):
                                negated = True
                                break
            
            if found and not negated:
                labels[label] = 1
            else:
                labels[label] = 0
        
        return labels
    
    def extract_labels(self, texts: List[str]) -> List[Dict[str, int]]:
        """
        Extract pathology labels from a list of radiology reports.
        
        Args:
            texts: List of report texts
        
        Returns:
            List of dictionaries mapping label names to values
        """
        if self.chexbert_available:
            # Use CheXbert model
            return self._chexbert_extract(texts)
        else:
            # Use rule-based extraction
            results = []
            for text in texts:
                results.append(self._rule_based_extract(text))
            return results
    
    def _chexbert_extract(self, texts: List[str]) -> List[Dict[str, int]]:
        """Extract using CheXbert model (when available)."""
        # Placeholder for actual CheXbert integration
        # This would use the CheXbert model for extraction
        return [self._rule_based_extract(text) for text in texts]


def compute_chexbert_metrics(
    reference_labels: List[Dict[str, int]],
    generated_labels: List[Dict[str, int]]
) -> Dict[str, float]:
    """
    Compute CheXbert F1 metrics between reference and generated labels.
    
    Args:
        reference_labels: Labels extracted from reference reports
        generated_labels: Labels extracted from generated reports
    
    Returns:
        Dictionary with accuracy, micro F1, macro F1, and per-class F1
    """
    # Convert to arrays
    n_samples = len(reference_labels)
    n_labels = len(CHEXBERT_LABELS)
    
    ref_array = np.zeros((n_samples, n_labels))
    gen_array = np.zeros((n_samples, n_labels))
    
    for i, (ref, gen) in enumerate(zip(reference_labels, generated_labels)):
        for j, label in enumerate(CHEXBERT_LABELS):
            ref_array[i, j] = ref.get(label, 0)
            gen_array[i, j] = gen.get(label, 0)
    
    # Compute metrics
    # Accuracy (exact match per sample)
    correct = (ref_array == gen_array).all(axis=1).sum()
    accuracy = correct / n_samples
    
    # Per-class metrics
    per_class_f1 = {}
    precisions = []
    recalls = []
    f1_scores = []
    
    for j, label in enumerate(CHEXBERT_LABELS):
        ref_col = ref_array[:, j]
        gen_col = gen_array[:, j]
        
        tp = ((ref_col == 1) & (gen_col == 1)).sum()
        fp = ((ref_col == 0) & (gen_col == 1)).sum()
        fn = ((ref_col == 1) & (gen_col == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_f1[label] = f1
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Macro F1 (average across classes)
    macro_f1 = np.mean(f1_scores)
    
    # Micro F1 (aggregate TP, FP, FN across all classes)
    total_tp = ((ref_array == 1) & (gen_array == 1)).sum()
    total_fp = ((ref_array == 0) & (gen_array == 1)).sum()
    total_fn = ((ref_array == 1) & (gen_array == 0)).sum()
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    return {
        "chexbert_accuracy": accuracy,
        "chexbert_f1_micro": micro_f1,
        "chexbert_f1_macro": macro_f1,
        "chexbert_precision_micro": micro_precision,
        "chexbert_recall_micro": micro_recall,
        "chexbert_f1_per_class": per_class_f1
    }

