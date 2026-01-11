import math
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

        v_pa_tokens = self.vision_model(pixel_values=pixel_values_pa).last_hidden_state  # (b, li, dv)
        v_lat_tokens = self.vision_model(pixel_values=pixel_values_lat).last_hidden_state  # (b, li, dv)
        v_pa_tokens = self.visual_projection(v_pa_tokens)  # (b, li, d)
        v_lat_tokens = self.visual_projection(v_lat_tokens)  # (b, li, d)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_outputs.last_hidden_state  # (b, lt, dt)
        t_tokens = self.text_projection(text_tokens)  # (b, lt, d)

        # L2 normalize per token
        v_pa_tokens = F.normalize(v_pa_tokens, dim=-1)
        v_lat_tokens = F.normalize(v_lat_tokens, dim=-1)
        t_tokens = F.normalize(t_tokens, dim=-1)
        # Mask out padded text tokens
        t_mask = batch["attention_mask"].unsqueeze(-1)
        t_tokens = t_tokens * t_mask
        return v_pa_tokens, v_lat_tokens, t_tokens

    def _compute_losses(
        self, v_pa_tokens: torch.Tensor, v_lat_tokens: torch.Tensor, t_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        tau = self.temperature
        b, li, _ = v_pa_tokens.shape
        target = torch.arange(b, device=self.device)

        # Image -> Text (PA, Lat) using token-level logsumexp aggregation
        sim_pa_t = torch.einsum("bid,Bjd->bBij", v_pa_tokens, t_tokens) / tau  # (b, b, li, lt)
        sim_pa_t = sim_pa_t.clamp(min=-50, max=50)
        sim_lat_t = torch.einsum("bid,Bjd->bBij", v_lat_tokens, t_tokens) / tau
        sim_lat_t = sim_lat_t.clamp(min=-50, max=50)
        logits_pa_t = torch.logsumexp(sim_pa_t, dim=(2, 3))  # (b, b)
        logits_lat_t = torch.logsumexp(sim_lat_t, dim=(2, 3))  # (b, b)
        loss_img_to_text = 0.5 * (
            F.cross_entropy(logits_pa_t, target) + F.cross_entropy(logits_lat_t, target)
        )

        # Text -> Image (multi-positive: PA + Lateral)
        images_tokens = torch.cat([v_pa_tokens, v_lat_tokens], dim=0)  # (2b, li, d)
        sim_t_img = torch.einsum("bid,Bjd->bBij", t_tokens, images_tokens) / tau  # (b, 2b, lt, li)
        sim_t_img = sim_t_img.clamp(min=-50, max=50)
        logits_t_img = torch.logsumexp(sim_t_img, dim=(2, 3))  # (b, 2b)
        positives = torch.stack([target, target + b], dim=1)  # (b, 2)
        logits_max = torch.logsumexp(logits_t_img, dim=1)
        pos_logits = torch.logsumexp(torch.gather(logits_t_img, 1, positives), dim=1)
        loss_text_to_img = -(pos_logits - logits_max).mean()

        # Image–Image consistency (token-level cosine aggregated via log-mean-exp)
        sim_pa_lat = torch.einsum("bid,bjd->bij", v_pa_tokens, v_lat_tokens)  # (b, li, li)
        log_avg_sim = torch.logsumexp(sim_pa_lat, dim=(1, 2)) - math.log(li * li)
        loss_img_img = (1 - log_avg_sim).mean()

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

