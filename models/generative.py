"""
Stage 2: Generative Model Module (ReportGenLightning).
Implements End-to-End Report Generation:
SigLIP (Frozen/QLoRA) -> Perceiver Resampler -> BioGPT (QLoRA)
"""
import torch
import pytorch_lightning as pl
from typing import List, Optional
from transformers import get_linear_schedule_with_warmup

from models.components import SigLIPEncoder, PerceiverResampler
from models.peft_config import get_lora_config, get_qlora_config, apply_lora, apply_qlora
from transformers import BioGptForCausalLM, BioGptTokenizer

class BioGPTGenerator(torch.nn.Module):
    def __init__(self, model_name="microsoft/biogpt", freeze_encoder=True):
        super().__init__()
        self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = BioGptForCausalLM.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask, image_embeds, labels=None, label_smoothing=0.0):
        # image_embeds: (B, Num_Queries, Dim)
        # input_ids: (B, Seq_Len)
        
        # Get text embeddings
        inputs_embeds = self.model.biogpt.embed_tokens(input_ids)  # (B, Seq, Dim)
        
        # Combine masks
        B, Img_Len, Dim = image_embeds.shape
        B, Txt_Len = input_ids.shape
        
        image_attention_mask = torch.ones((B, Img_Len), device=image_embeds.device).long()
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        combined_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
        
        # Forward WITHOUT labels to get logits only
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=None  # Don't compute internal loss
        )
        
        # Manual loss computation with label smoothing
        if labels is not None:
            # Create combined labels (image tokens = -100)
            image_labels = torch.full((B, Img_Len), -100, device=labels.device, dtype=labels.dtype)
            combined_labels = torch.cat([image_labels, labels], dim=1)
            
            # Shift for causal LM: predict next token
            # logits: (B, Seq, Vocab), labels: (B, Seq)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = combined_labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Attach loss to outputs for compatibility
            outputs.loss = loss
        
        return outputs

    def generate(self, image_embeds, max_length=None, **kwargs):
        # Import config for generation parameters
        from utils.config import (
            MAX_GENERATION_LENGTH, GENERATION_NUM_BEAMS, GENERATION_DO_SAMPLE,
            GENERATION_REPETITION_PENALTY, GENERATION_NO_REPEAT_NGRAM_SIZE, 
            GENERATION_LENGTH_PENALTY, GENERATION_EARLY_STOPPING
        )
        
        max_length = max_length or MAX_GENERATION_LENGTH
        
        # Create attention mask for images
        B, Img_Len, Dim = image_embeds.shape
        image_attention_mask = torch.ones((B, Img_Len), device=image_embeds.device).long()
        
        # PURE BEAM SEARCH (Deterministic for medical applications)
        outputs = self.model.generate(
            inputs_embeds=image_embeds,
            attention_mask=image_attention_mask,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=kwargs.get('num_beams', GENERATION_NUM_BEAMS),
            do_sample=kwargs.get('do_sample', GENERATION_DO_SAMPLE),  # False
            repetition_penalty=kwargs.get('repetition_penalty', GENERATION_REPETITION_PENALTY),
            no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', GENERATION_NO_REPEAT_NGRAM_SIZE),
            length_penalty=kwargs.get('length_penalty', GENERATION_LENGTH_PENALTY),
            early_stopping=kwargs.get('early_stopping', GENERATION_EARLY_STOPPING)
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

class ReportGenLightning(pl.LightningModule):
    def __init__(
        self,
        siglip_model_name: str = "google/siglip-base-patch16-224",
        biogpt_model_name: str = "microsoft/biogpt",
        num_queries: int = None,  # Will use config default
        learning_rate: float = None,  # Will use config default
        warmup_steps: int = None,  # Will use config default
        freeze_vision: bool = True,
        use_qlora: bool = True,
        vision_lora_enabled: bool = False  # Track if vision encoder uses LoRA
    ):
        super().__init__()
        
        # Import config for defaults
        from utils.config import (
            NUM_PROJECTION_QUERIES, PROJECTION_LAYERS, PROJECTION_DROPOUT,
            LEARNING_RATE_STAGE2, WARMUP_STEPS_STAGE2, LABEL_SMOOTHING
        )
        
        # Apply defaults from config
        num_queries = num_queries or NUM_PROJECTION_QUERIES
        learning_rate = learning_rate or LEARNING_RATE_STAGE2
        warmup_steps = warmup_steps or WARMUP_STEPS_STAGE2
        
        self.save_hyperparameters()
        
        # 1. Vision Encoder (SigLIP)
        self.vision_encoder = SigLIPEncoder(siglip_model_name)
        
        # Apply LoRA to vision encoder if enabled (for checkpoint compatibility)
        if vision_lora_enabled:
            lora_config_vision = get_lora_config(model_type="vision")
            self.vision_encoder.model = apply_lora(self.vision_encoder.model, lora_config_vision)
        
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()
            
        # 2. Perceiver Resampler (OPTIMIZED from config)
        self.perceiver = PerceiverResampler(
            input_dim=768,  # SigLIP dim
            output_dim=1024, # BioGPT dim
            num_queries=num_queries,
            num_layers=PROJECTION_LAYERS,
            dropout=PROJECTION_DROPOUT
        )
        
        # 3. BioGPT Generator
        self.biogpt = BioGPTGenerator(biogpt_model_name, freeze_encoder=False)
        
        # Apply LoRA to BioGPT with HIGHER RANK for LLM
        lora_config = get_lora_config(model_type="llm")  # Uses r=64, alpha=128
        self.biogpt.model = apply_lora(self.biogpt.model, lora_config)
        
        self.tokenizer = self.biogpt.tokenizer
        self.label_smoothing = LABEL_SMOOTHING  # 0.1

    def forward(self, pixel_values, input_ids, attention_mask, view_mask=None, labels=None, view_positions=None):
        # 1. Vision Features
        # (B, V, C, H, W) -> (B, V*Seq, Dim)
        with torch.set_grad_enabled(not self.hparams.freeze_vision):
            image_embeds = self.vision_encoder.get_patch_embeddings(pixel_values)
        
        # 2. Perceiver Resampler
        # Projects to fixed number of tokens (B, Num_Queries, Dim)
        # view_mask: (B, V) -> Used to create key_padding_mask inside
        # Note: image_embeds has V*Seq length. view_mask needs expansion inside perceiver if provided.
        projected_embeds = self.perceiver(image_embeds, view_mask=view_mask, view_positions=view_positions)
        
        # 3. Generate / Loss (with label smoothing)
        outputs = self.biogpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=projected_embeds,
            labels=labels,
            label_smoothing=self.label_smoothing
        )
        return outputs

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            view_mask = batch.get('view_mask')
            view_positions = batch.get('view_positions')
        
        outputs = self(pixel_values, input_ids, attention_mask, view_mask, labels, view_positions)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            view_mask = batch.get('view_mask')
            view_positions = batch.get('view_positions')
            
        outputs = self(pixel_values, input_ids, attention_mask, view_mask, labels, view_positions)
        loss = outputs.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Compute generation metrics once per epoch (diagnostic only; not used for optimization/checkpointing)
        if batch_idx == 0:
            try:
                # Generate reports
                generated = self.generate(pixel_values, view_mask=view_mask, view_positions=view_positions)
                
                # Decode reference from labels
                reference_ids = labels.clone()
                reference_ids[reference_ids == -100] = self.tokenizer.pad_token_id
                references = self.tokenizer.batch_decode(reference_ids, skip_special_tokens=True)
                
                # Compute BLEU on tokenizer-consistent tokens (BioGPT tokenizer), not whitespace
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smoothing = SmoothingFunction().method1
                
                bleu_scores = []
                for gen, ref in zip(generated, references):
                    gen_toks = self.tokenizer.tokenize(gen)
                    ref_toks = self.tokenizer.tokenize(ref)
                    if len(gen_toks) > 0 and len(ref_toks) > 0:
                        score = sentence_bleu([ref_toks], gen_toks, smoothing_function=smoothing)
                        bleu_scores.append(score)
                
                if bleu_scores:
                    avg_bleu = sum(bleu_scores) / len(bleu_scores)
                    self.log('val_bleu_tok', avg_bleu, on_step=False, on_epoch=True, prog_bar=False)

                # Compute ROUGE-L (f-measure), diagnostic only
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rougeL_scores = []
                for gen, ref in zip(generated, references):
                    scores = scorer.score(ref, gen)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                if rougeL_scores:
                    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
                    self.log('val_rougeL', avg_rougeL, on_step=False, on_epoch=True, prog_bar=False)
            except Exception as e:
                # Don't fail training if metrics fail; log for observability
                if not hasattr(self, '_metric_error_logged'):
                    self._metric_error_logged = True
                    import logging
                    logging.getLogger(__name__).warning(f"Validation metrics failed (rate-limited): {e}")
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.learning_rate
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def generate(self, pixel_values, view_mask=None, view_positions=None, **kwargs):
        # Vision
        image_embeds = self.vision_encoder.get_patch_embeddings(pixel_values)
        # Perceiver
        projected_embeds = self.perceiver(image_embeds, view_mask=view_mask, view_positions=view_positions)
        # BioGPT Generate
        return self.biogpt.generate(projected_embeds, **kwargs)
