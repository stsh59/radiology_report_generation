"""
Evaluate models using PyTorch Lightning.
"""
import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from transformers import BioGptTokenizer

from data.datamodule import MultiViewDataModule
from models.generative import ReportGenLightning
from eval.metrics import MedicalReportMetrics, format_metrics_report
from utils.config import (
    OUTPUT_DIR, MIMIC_TEST_CSV, IMAGES_DIR_MIMIC, BIOGPT_MODEL_NAME,
    GENERATION_NUM_BEAMS, MAX_GENERATION_LENGTH
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


@torch.no_grad()
def generate_reports(model, datamodule, split='test', limit_batches=None):
    """
    Generate reports using Lightning model.
    """
    model.eval()
    
    if split == 'test':
        dataloader = datamodule.test_dataloader()
    elif split == 'val':
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")
    
    results = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Generating reports")):
        if limit_batches and i >= limit_batches:
            break
            
        pixel_values = batch['pixel_values'].to(model.device)
        view_mask = batch['view_mask'].to(model.device)
        view_positions = batch.get('view_positions')
        if view_positions is not None:
            view_positions = view_positions.to(model.device)
        input_ids = batch['input_ids'] # Ground truth for reference
        study_ids = batch['study_id']
        
        # Decode reference texts
        references = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Generate using config defaults for consistency
        generated_texts = model.generate(
            pixel_values=pixel_values,
            view_mask=view_mask,
            view_positions=view_positions,
            max_length=MAX_GENERATION_LENGTH,
            num_beams=GENERATION_NUM_BEAMS
        )
        
        for j in range(len(generated_texts)):
            results.append({
                'study_id': study_ids[j],
                'generated': generated_texts[j],
                'reference': references[j]
            })
    
    return results


def evaluate_model(checkpoint_path: str, split: str = 'test', output_dir: Path = None, limit_batches: int = None):
    """
    Evaluate a Lightning model checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading Lightning checkpoint from {checkpoint_path}")
    
    # Load model (make sure to set use_qlora=False if on CPU, or let auto-detect handle it)
    # Since we are just evaluating, strict loading might be tricky if environment changed.
    # ReportGenLightning handles auto-detect of CUDA for QLoRA in init, 
    # but load_from_checkpoint calls init.
    model = ReportGenLightning.load_from_checkpoint(checkpoint_path, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Setup Data
    tokenizer = BioGptTokenizer.from_pretrained(BIOGPT_MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    datamodule = MultiViewDataModule(
        test_csv=str(MIMIC_TEST_CSV),
        tokenizer=tokenizer,
        batch_size=8,
        num_workers=4,
        max_views=3,
        max_length=256,
        image_root=str(IMAGES_DIR_MIMIC)
    )
    datamodule.setup(stage='test')
    
    logger.info(f"Generating reports on {split} set...")
    logger.info(f"Generation params: max_length={MAX_GENERATION_LENGTH}, num_beams={GENERATION_NUM_BEAMS}")
    results = generate_reports(model, datamodule, split=split, limit_batches=limit_batches)
    
    if output_dir is None:
        if checkpoint_path.parent.name == "checkpoints": # Handle basic structure
             output_dir = OUTPUT_DIR / "evaluation" / checkpoint_path.stem
        else:
             output_dir = OUTPUT_DIR / "evaluation" / checkpoint_path.parent.name
             
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_csv = output_dir / f"generated_reports_{split}.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved generated reports to {results_csv}")
    
    logger.info("Computing evaluation metrics...")
    metric_calculator = MedicalReportMetrics()
    
    references = [r['reference'] for r in results]
    generated = [r['generated'] for r in results]
    
    metrics = metric_calculator.compute_all_metrics(references, generated)
    
    logger.info("\n" + format_metrics_report(metrics))
    
    metrics_file = output_dir / f"metrics_{split}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    return metrics, results
