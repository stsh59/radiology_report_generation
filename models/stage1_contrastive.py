from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
import pytorch_lightning as pl
from transformers import SiglipModel, SiglipVisionModel, SiglipTextModel, SiglipProcessor


class Stage1ContrastiveModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        learning_rate: float = 1e-4,
        temperature: float = 0.07,
        lambda_img_img: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        base_model = SiglipModel.from_pretrained(model_name)
        self.processor = SiglipProcessor.from_pretrained(model_name)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )

        self.vision_model: SiglipVisionModel = get_peft_model(base_model.vision_model, lora_cfg)
        self.text_model: SiglipTextModel = get_peft_model(base_model.text_model, lora_cfg)
        self.visual_projection = base_model.visual_projection
        self.text_projection = base_model.text_projection

        for name, param in self.vision_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        for name, param in self.text_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        for param in self.visual_projection.parameters():
            param.requires_grad = False
        for param in self.text_projection.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.lambda_img_img = lambda_img_img
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        return optimizer

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_values_pa = batch["pixel_values_pa"]
        pixel_values_lat = batch["pixel_values_lateral"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        v_pa_tokens = self.vision_model(pixel_values=pixel_values_pa).last_hidden_state  # (b, seq, dim)
        v_lat_tokens = self.vision_model(pixel_values=pixel_values_lat).last_hidden_state
        v_pa_pooled = v_pa_tokens.mean(dim=1)
        v_lat_pooled = v_lat_tokens.mean(dim=1)
        v_pa = self.visual_projection(v_pa_pooled)
        v_lat = self.visual_projection(v_lat_pooled)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_outputs.last_hidden_state  # (b, seq, dim)
        mask = attention_mask.unsqueeze(-1)
        text_pooled = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        t = self.text_projection(text_pooled)

        v_pa = F.normalize(v_pa, dim=-1)
        v_lat = F.normalize(v_lat, dim=-1)
        t = F.normalize(t, dim=-1)
        return v_pa, v_lat, t

    def _compute_losses(self, v_pa: torch.Tensor, v_lat: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        tau = self.temperature
        b = t.shape[0]

        logits_pa_t = (v_pa @ t.T) / tau
        logits_lat_t = (v_lat @ t.T) / tau
        target = torch.arange(b, device=self.device)

        loss_img_to_text = 0.5 * (
            F.cross_entropy(logits_pa_t, target) + F.cross_entropy(logits_lat_t, target)
        )

        images = torch.cat([v_pa, v_lat], dim=0)  # (2b, d)
        logits_t_img = (t @ images.T) / tau  # (b, 2b)
        positives = torch.stack([target, target + b], dim=1)  # (b, 2)
        logits_max = torch.logsumexp(logits_t_img, dim=1)
        pos_logits = torch.logsumexp(
            torch.gather(logits_t_img, 1, positives),
            dim=1,
        )
        loss_text_to_img = -(pos_logits - logits_max).mean()

        cosine_sim = F.cosine_similarity(v_pa, v_lat, dim=-1)
        loss_img_img = (1 - cosine_sim).mean()

        total_loss = loss_img_to_text + loss_text_to_img + self.lambda_img_img * loss_img_img

        return {
            "loss_img_to_text": loss_img_to_text,
            "loss_text_to_img": loss_text_to_img,
            "loss_img_img": loss_img_img,
            "loss": total_loss,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        v_pa, v_lat, t = self.forward(batch)
        losses = self._compute_losses(v_pa, v_lat, t)
        batch_size = batch["input_ids"].size(0)
        self.log("train/loss", losses["loss"], prog_bar=True, batch_size=batch_size)
        self.log("train/loss_img_to_text", losses["loss_img_to_text"], batch_size=batch_size)
        self.log("train/loss_text_to_img", losses["loss_text_to_img"], batch_size=batch_size)
        self.log("train/loss_img_img", losses["loss_img_img"], batch_size=batch_size)
        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        v_pa, v_lat, t = self.forward(batch)
        losses = self._compute_losses(v_pa, v_lat, t)
        batch_size = batch["input_ids"].size(0)
        self.log("val/total_loss", losses["loss"], prog_bar=True, batch_size=batch_size)
        self.log("val/loss_img_to_text", losses["loss_img_to_text"], batch_size=batch_size)
        self.log("val/loss_text_to_img", losses["loss_text_to_img"], batch_size=batch_size)
        self.log("val/loss_img_img", losses["loss_img_img"], batch_size=batch_size)
        return losses["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        v_pa, v_lat, t = self.forward(batch)
        losses = self._compute_losses(v_pa, v_lat, t)
        batch_size = batch["input_ids"].size(0)
        self.log("test/total_loss", losses["loss"], batch_size=batch_size)
        return losses["loss"]

    def save_image_encoder(self, output_dir: str) -> None:
        self.vision_model.save_pretrained(output_dir)

