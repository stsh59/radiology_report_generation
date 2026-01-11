from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from models.perceiver_resampler import PerceiverResampler


def _lcs_length(x_tokens: List[str], y_tokens: List[str]) -> int:
    m, n = len(x_tokens), len(y_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x_tokens[i - 1] == y_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            scores.append(0.0)
            continue
        prec = lcs / max(len(pred_tokens), 1)
        rec = lcs / max(len(ref_tokens), 1)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((2 * prec * rec) / (prec + rec))
    return float(sum(scores) / max(len(scores), 1))


def _compute_bleu4(preds: List[str], refs: List[str]) -> float:
    # Simple BLEU-4 with brevity penalty using n-gram precision over whitespace tokens
    from collections import Counter
    import math

    def ngram_counts(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def modified_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_counts = ngram_counts(pred_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)
        overlap = sum(min(count, ref_counts[ng]) for ng, count in pred_counts.items())
        total = sum(pred_counts.values())
        return overlap / total if total > 0 else 0.0

    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        precisions = [modified_precision(pred_tokens, ref_tokens, n) for n in range(1, 5)]
        if any(p == 0 for p in precisions):
            geo_mean = 0.0
        else:
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens)) if len(pred_tokens) < len(ref_tokens) else 1.0
        scores.append(bp * geo_mean)
    return float(sum(scores) / max(len(scores), 1))


class Stage2GeneratorModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        lora_path: str = "stage1_image_encoder",
        lm_name: str = "microsoft/biogpt-large",
        learning_rate: float = 5e-5,
        num_latents: int = 32,
        latent_dim: int = 768,
        num_cross_attn_layers: int = 2,
        num_cross_attn_heads: int = 8,
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        base_vision = SiglipVisionModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        vision_peft = get_peft_model(base_vision, lora_cfg)
        vision_peft.load_adapter(lora_path, adapter_name="default", is_trainable=False)
        vision_peft.set_adapter("default")
        self.image_encoder: SiglipVisionModel = vision_peft
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.perceiver = PerceiverResampler(
            dim=latent_dim,
            num_latents=num_latents,
            num_layers=num_cross_attn_layers,
            num_heads=num_cross_attn_heads,
        )

        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.projection = nn.Linear(latent_dim, self.lm.config.hidden_size, bias=True)
        for p in self.lm.parameters():
            p.requires_grad = False

        # Ensure projection and perceiver are the only trainable modules
        for p in self.projection.parameters():
            p.requires_grad = True
        for p in self.perceiver.parameters():
            p.requires_grad = True

        self.learning_rate = learning_rate
        self.max_new_tokens = max_new_tokens
        self.val_preds: List[str] = []
        self.val_refs: List[str] = []

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.learning_rate)

    def _encode_images(self, pixel_values_pa: torch.Tensor, pixel_values_lat: torch.Tensor) -> torch.Tensor:
        pa_tokens = self.image_encoder(pixel_values=pixel_values_pa).last_hidden_state
        lat_tokens = self.image_encoder(pixel_values=pixel_values_lat).last_hidden_state
        tokens = torch.cat([pa_tokens, lat_tokens], dim=1)
        latents = self.perceiver(tokens)
        return latents  # (b, num_latents, dim)

    def _build_inputs(
        self, vision_latents: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, _ = vision_latents.shape
        vision_mask = torch.ones((b, l), device=vision_latents.device, dtype=attention_mask.dtype)
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        vision_proj = self.projection(vision_latents)
        inputs_embeds = torch.cat([vision_proj, text_embeds], dim=1)
        attn_mask = torch.cat([vision_mask, attention_mask], dim=1)
        return inputs_embeds, attn_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_values_pa = batch["pixel_values_pa"]
        pixel_values_lat = batch["pixel_values_lateral"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        vision_latents = self._encode_images(pixel_values_pa, pixel_values_lat)
        inputs_embeds, attn_mask = self._build_inputs(vision_latents, input_ids, attention_mask)

        vision_labels = torch.full(
            (vision_latents.size(0), vision_latents.size(1)), -100, device=labels.device, dtype=labels.dtype
        )
        full_labels = torch.cat([vision_labels, labels], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=full_labels,
        )
        return outputs.loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("train/lm_loss", loss, prog_bar=True, batch_size=batch["input_ids"].size(0))
        return loss

    def _greedy_generate(self, vision_latents: torch.Tensor, tokenizer: AutoTokenizer) -> List[str]:
        device = vision_latents.device
        b = vision_latents.size(0)
        generated = torch.full((b, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        outputs_text: List[List[int]] = [[] for _ in range(b)]

        for _ in range(self.max_new_tokens):
            text_embeds = self.lm.get_input_embeddings()(generated)
            vision_proj = self.projection(vision_latents)
            inputs_embeds = torch.cat([vision_proj, text_embeds], dim=1)
            attn_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
            logits = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask).logits
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
            for i, token in enumerate(next_token.tolist()):
                if not finished[i]:
                    outputs_text[i].append(token)
                    if token == tokenizer.eos_token_id:
                        finished[i] = True
            if finished.all():
                break

        decoded = []
        for tokens in outputs_text:
            decoded.append(tokenizer.decode(tokens, skip_special_tokens=True))
        return decoded

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pixel_values_pa = batch["pixel_values_pa"]
        pixel_values_lat = batch["pixel_values_lateral"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        vision_latents = self._encode_images(pixel_values_pa, pixel_values_lat)
        inputs_embeds, attn_mask = self._build_inputs(vision_latents, input_ids, attention_mask)

        vision_labels = torch.full(
            (vision_latents.size(0), vision_latents.size(1)), -100, device=labels.device, dtype=labels.dtype
        )
        full_labels = torch.cat([vision_labels, labels], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=full_labels,
        )
        loss = outputs.loss
        self.log("val/lm_loss", loss, prog_bar=True, batch_size=input_ids.size(0))

        # Greedy decoding for metrics (validation only)
        tokenizer: AutoTokenizer = self.trainer.datamodule.tokenizer  # type: ignore
        with torch.no_grad():
            preds = self._greedy_generate(vision_latents, tokenizer)
        refs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        self.val_preds.extend(preds)
        self.val_refs.extend(refs)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_preds = []
        self.val_refs = []

    def on_validation_epoch_end(self) -> None:
        bleu4 = _compute_bleu4(self.val_preds, self.val_refs)
        rouge_l = _compute_rouge_l(self.val_preds, self.val_refs)
        self.log("val/bleu4", bleu4, prog_bar=False, sync_dist=True)
        self.log("val/rougeL", rouge_l, prog_bar=False, sync_dist=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("test/lm_loss", loss, batch_size=batch["input_ids"].size(0))
        return loss

