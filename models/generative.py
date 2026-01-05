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
        """
        Generate medical reports from image embeddings.
        
        Architecture: [image_embeds][BOS] -> generated_tokens
        BOS is appended AFTER images to match training pattern [images][text].
        """
        from utils.config import (
            MAX_GENERATION_LENGTH, GENERATION_NUM_BEAMS,
            GENERATION_REPETITION_PENALTY, GENERATION_NO_REPEAT_NGRAM_SIZE
        )
        
        max_length = max_length or MAX_GENERATION_LENGTH
        B, Img_Len, Dim = image_embeds.shape
        
        # === BOS Token: Appended AFTER images ===
        # The model is trained on [images][text], so BOS marks the start of text generation
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.eos_token_id  # BioGPT fallback
        
        bos_tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=image_embeds.device)
        bos_embeds = self.model.biogpt.embed_tokens(bos_tokens)  # (B, 1, Dim)
        
        # Concatenate: [image_embeds][bos_embed] - BOS after images
        combined_embeds = torch.cat([image_embeds, bos_embeds], dim=1)
        input_length = Img_Len + 1  # Track for output slicing
        
        combined_attention_mask = torch.ones((B, input_length), device=image_embeds.device).long()
        
        # === Optimized Generation Parameters (based on report length analysis) ===
        outputs = self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_length,
            min_new_tokens=20,           # Safeguard: at least ~80 chars, prevents empty outputs
            early_stopping=False,        # Explore all beams for best medical accuracy
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=kwargs.get('num_beams', GENERATION_NUM_BEAMS),
            do_sample=False,             # Deterministic for medical applications
            repetition_penalty=kwargs.get('repetition_penalty', GENERATION_REPETITION_PENALTY),
            no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', GENERATION_NO_REPEAT_NGRAM_SIZE),
            length_penalty=1.0,          # Neutral: let model decide natural length
        )
        
        # === Output Slicing: Remove placeholder positions ===
        # First `input_length` positions are filled with pad_token_id (placeholders for embeddings)
        # Only positions after that contain actual generated token IDs
        generated_ids = outputs[:, input_length:]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

class ReportGenLightning(pl.LightningModule):
    def __init__(
        self,
        siglip_model_name: str = "google/siglip-base-patch16-224",
        biogpt_model_name: str = "microsoft/biogpt",
        num_queries: int = None,  # Will use config default
        learning_rate: float = None,  # Will use config default
        warmup_steps: int = None,  # Will use config default
        freeze_vision: bool = None,  # None = auto-detect based on stage1_checkpoint_provided
        use_qlora: bool = True,
        vision_lora_enabled: bool = False,  # Track if vision encoder uses LoRA
        stage1_checkpoint_provided: bool = False  # For auto-detect freeze logic
    ):
        super().__init__()
        
        # Import config for defaults
        from utils.config import (
            NUM_PROJECTION_QUERIES, PROJECTION_LAYERS, PROJECTION_DROPOUT,
            LEARNING_RATE_STAGE2, WARMUP_STEPS_STAGE2, LABEL_SMOOTHING
        )
        from utils.logger import setup_logger
        logger = setup_logger(__name__)
        
        # Apply defaults from config
        num_queries = num_queries or NUM_PROJECTION_QUERIES
        learning_rate = learning_rate or LEARNING_RATE_STAGE2
        warmup_steps = warmup_steps or WARMUP_STEPS_STAGE2
        
        # === Smart freeze logic ===
        # Auto-detect: unfreeze if no Stage 1 checkpoint provided
        if freeze_vision is None:
            freeze_vision = stage1_checkpoint_provided
            logger.info(f"Auto freeze_vision={freeze_vision} (stage1_provided={stage1_checkpoint_provided})")
        
        self.save_hyperparameters()
        
        # 1. Vision Encoder (SigLIP)
        self.vision_encoder = SigLIPEncoder(siglip_model_name)
        
        # Apply LoRA to vision encoder if enabled (for checkpoint compatibility)
        if vision_lora_enabled:
            lora_config_vision = get_lora_config(model_type="vision")
            self.vision_encoder.model = apply_lora(self.vision_encoder.model, lora_config_vision)
            logger.info("Vision LoRA adapters applied")
        
        # === Freeze logic with LoRA-aware eval() handling ===
        if freeze_vision:
            # Freeze base but keep LoRA trainable
            for name, param in self.vision_encoder.named_parameters():
                if 'lora_' in name.lower():
                    param.requires_grad = True  # LoRA stays trainable
                else:
                    param.requires_grad = False
            
            # CRITICAL: Do NOT call eval() if LoRA enabled!
            # eval() disables dropout in LoRA layers, hurting training.
            if not vision_lora_enabled:
                self.vision_encoder.eval()
                logger.info("Vision encoder: fully frozen (eval mode)")
            else:
                # Keep in train mode for LoRA dropout to work
                logger.info("Vision encoder: base frozen, LoRA trainable (train mode for dropout)")
        else:
            # Full training mode
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            self.vision_encoder.train()
            logger.info("Vision encoder: fully trainable")
            
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
        
        # === Enable gradient checkpointing for BioGPT (VRAM optimization) ===
        # Only for BioGPT - NOT for vision encoder (it's frozen, checkpointing adds overhead)
        if hasattr(self.biogpt.model, 'gradient_checkpointing_enable'):
            self.biogpt.model.gradient_checkpointing_enable()
            logger.info("BioGPT gradient checkpointing enabled (~30-40% VRAM reduction)")
        
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
        
        # === Perceiver gradient verification (every 100 steps) ===
        if batch_idx % 100 == 0 and batch_idx > 0:
            grad_norm = 0.0
            grad_count = 0
            for name, param in self.perceiver.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
                    grad_count += 1
            
            if grad_count > 0:
                grad_norm = grad_norm ** 0.5
                self.log('perceiver_grad_norm', grad_norm, on_step=True, prog_bar=False)
                if grad_norm < 1e-7:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Step {self.trainer.global_step}: Perceiver gradients near zero!"
                    )
        
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
        # === Weight decay exclusion for LayerNorm and bias (stability improvement) ===
        # Patterns to catch: BioGPT's "layer_norm", Perceiver's ".norm.", and all biases
        no_decay = ["bias", "LayerNorm", "layer_norm", ".norm."]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
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
