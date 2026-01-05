"""
Stage 1: Contrastive Learning Module (MedicalSigLIP).
Implements QLoRA fine-tuning of SigLIP for multi-view image-text alignment.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
from models.peft_config import get_qlora_config, apply_qlora, count_trainable_parameters

class ContrastiveLoss(nn.Module):
    """Contrastive loss (InfoNCE) for image-text alignment."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Sim Matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # Labels
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_i2t = self.cross_entropy(logits, labels)
        loss_t2i = self.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2

class MedicalSigLIPLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        temperature: float = 0.07,
        use_qlora: bool = False  # Changed from True - fp16 weights transfer cleanly to Stage 2
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Model
        if use_qlora and torch.cuda.is_available():
            from models.peft_config import get_qlora_config, apply_qlora
            lora_config, bnb_config = get_qlora_config()
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": 0} 
            )
            self.model = apply_qlora(self.model, lora_config)
        else:
            # Standard LoRA without quantization (fp16 base weights)
            from models.peft_config import get_lora_config, apply_lora
            self.model = AutoModel.from_pretrained(model_name)
            lora_config = get_lora_config(model_type="vision")  # Uses config: q_proj, k_proj, v_proj, out_proj
            self.model = apply_lora(self.model, lora_config)
        
        # Processor for validation text encoding (if needed outside tokens)
        # But we pass pre-tokenized IDs usually.
        
        self.criterion = ContrastiveLoss(temperature=temperature)
        count_trainable_parameters(self.model)

    def forward(self, pixel_values, input_ids, attention_mask=None, view_mask=None):
        # 1. Image Encoding (Multi-View)
        # pixel_values: (B, V, C, H, W)
        b, v, c, h, w = pixel_values.shape
        pixel_values_flat = pixel_values.view(b * v, c, h, w)
        
        # Get pooled features for contrastive
        image_features = self.model.get_image_features(pixel_values=pixel_values_flat) # (B*V, D)
        
        # Masked averaging across views (only valid views, not padded black images)
        image_features = image_features.view(b, v, -1)  # (B, V, D)
        if view_mask is not None:
            # view_mask: (B, V) where 1=valid, 0=padding
            mask = view_mask.unsqueeze(-1).float()  # (B, V, 1)
            image_features = (image_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            image_features = image_features.mean(dim=1)  # (B, D)
        
        # 2. Text Encoding
        # SigLIP's text model expects input_ids usually.
        # Check if model has text_model or uses separate processor calls.
        # AutoModel usually returns (text_embeds, image_embeds) depending on input.
        # But get_text_features is cleaner.
        # Pass attention_mask when available to avoid pad tokens affecting embeddings
        if attention_mask is not None:
            text_features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            text_features = self.model.get_text_features(input_ids=input_ids)
        
        return image_features, text_features

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            view_mask = batch.get('view_mask')
        else:
             # Fallback
             pixel_values, texts, meta = batch
             attention_mask = None
             view_mask = None

        image_embeds, text_embeds = self(pixel_values, input_ids, attention_mask=attention_mask, view_mask=view_mask)
        loss = self.criterion(image_embeds, text_embeds)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            view_mask = batch.get('view_mask')
        else:
            pixel_values, texts, meta = batch
            attention_mask = None
            view_mask = None
        
        image_embeds, text_embeds = self(pixel_values, input_ids, attention_mask=attention_mask, view_mask=view_mask)
        loss = self.criterion(image_embeds, text_embeds)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
