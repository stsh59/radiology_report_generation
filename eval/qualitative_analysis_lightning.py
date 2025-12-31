"""
Qualitative analysis using PyTorch Lightning models.
"""
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import random
from transformers import BioGptTokenizer

from data.datamodule import MultiViewDataModule
from models.generative import ReportGenLightning
from utils.config import (
    OUTPUT_DIR, IMAGES_DIR_MIMIC, MIMIC_TEST_CSV, BIOGPT_MODEL_NAME,
    GENERATION_NUM_BEAMS, MAX_GENERATION_LENGTH, IMAGE_MEAN, IMAGE_STD
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


@torch.no_grad()
def visualize_samples(checkpoint_path: str, num_samples: int = 5):
    """
    Visualize predictions using Lightning model.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading Lightning checkpoint from {checkpoint_path}")
    
    model = ReportGenLightning.load_from_checkpoint(checkpoint_path, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Setup Data
    tokenizer = BioGptTokenizer.from_pretrained(BIOGPT_MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Use batch_size=1 for simple iteration, but dataset gives items directly
    datamodule = MultiViewDataModule(
        test_csv=str(MIMIC_TEST_CSV),
        tokenizer=tokenizer,
        batch_size=1,
        num_workers=0,
        max_views=3,
        max_length=256,
        image_root=str(IMAGES_DIR_MIMIC)
    )
    datamodule.setup(stage='test')
    dataset = datamodule.test_dataset
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 6 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(indices):
        batch = dataset[sample_idx]
        
        # Unpack
        pixel_values = batch['pixel_values'] # (3, 3, 224, 224)
        view_mask = batch['view_mask']       # (3,)
        view_positions = batch.get('view_positions')
        input_ids = batch['input_ids']
        study_id = batch['study_id']
        
        # Prepare for Model
        pixel_values_batch = pixel_values.unsqueeze(0).to(device)
        view_mask_batch = view_mask.unsqueeze(0).to(device)
        view_positions_batch = None
        if view_positions is not None:
            view_positions_batch = view_positions.unsqueeze(0).to(device)
        
        # Generate using config defaults for consistency
        generated_text = model.generate(
            pixel_values=pixel_values_batch,
            view_mask=view_mask_batch,
            view_positions=view_positions_batch,
            max_length=MAX_GENERATION_LENGTH,
            num_beams=GENERATION_NUM_BEAMS
        )[0]
        
        # Reference
        reference_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Prepare Image for Display
        # 1. Denormalize using actual config values
        mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGE_STD).view(3, 1, 1)
        imgs = pixel_values * std + mean
        # 2. Make Grid of views
        grid = torchvision.utils.make_grid(imgs, nrow=3, padding=2)
        # 3. CHW -> HWC
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        ax = axes[idx]
        ax.imshow(grid_np)
        ax.axis('off')
        
        # Add text info
        ax.set_title(f"Study UID: {study_id}", fontsize=12, fontweight='bold', pad=20)
        
        # Text box
        text_str = (
            f"GENERATED:\n{generated_text}\n\n"
            f"REFERENCE:\n{reference_text}"
        )
        
        # Place text to the right or below? Below is easier with this layout.
        # Actually, let's just put it in the title or axis text
        # Better: use text inside axis
        ax.text(
            0, -0.1,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            wrap=True
        )
    
    plt.tight_layout()
    
    if checkpoint_path.parent.name == "checkpoints":
         output_dir = OUTPUT_DIR / "qualitative" / checkpoint_path.stem
    else:
         output_dir = OUTPUT_DIR / "qualitative" / checkpoint_path.parent.name
         
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "qualitative_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    
    # plt.show() # Skip show in headless env
