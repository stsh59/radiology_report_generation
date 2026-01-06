"""
Shared model components for the Multi-View Report Generation Pipeline.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from utils.config import NUM_PROJECTION_QUERIES, PROJECTION_LAYERS, PROJECTION_HEADS, PROJECTION_DROPOUT

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler (Transformer Decoder with Learnable Queries).
    Maps variable-length visual sequences to a fixed number of context tokens.
    OPTIMIZED: Deeper architecture, more queries, view position embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = 768,    # SigLIP embedding dim
        output_dim: int = 1024,  # BioGPT hidden dim
        num_queries: int = NUM_PROJECTION_QUERIES,   # Default from config (64)
        num_layers: int = PROJECTION_LAYERS,         # Default from config (4)
        nhead: int = PROJECTION_HEADS,
        dropout: float = PROJECTION_DROPOUT,         # Default from config (0.15)
        max_views: int = 3
    ):
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        self.max_views = max_views
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, output_dim) * 0.02)
        
        # Project input to output dimension first (to match query dim)
        self.input_proj = nn.Linear(input_dim, output_dim)

        # View position embeddings (categorical, added to visual tokens; not concatenated)
        # Mapping: PAD=0, PA=1, AP=2, LATERAL=3, OTHER=4
        self.view_embedding = nn.Embedding(num_embeddings=5, embedding_dim=output_dim, padding_idx=0)
        
        # Transformer Decoder (Cross-Attention)
        # Note: 'batch_first=True' is standard in newer PyTorch versions
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection / Norm
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
            
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Small init for view embeddings (PAD stays 0 due to padding_idx)
        nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.view_embedding.weight[0].zero_()
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        view_mask: torch.Tensor = None,
        view_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            image_embeds: [batch_size, seq_len, input_dim]
                          For multi-view, seq_len = num_views * patches_per_image
            view_mask: Optional [batch_size, seq_len] mask (True/1 for valid tokens, False/0 for padding)
                       If using nn.Transformer with key_padding_mask, True usually means IGNORE.
                       We will standardize on 0=padding, 1=valid for general logic, 
                       but convert to proper format for PyTorch transformer.
            view_positions: Optional [batch_size, num_views] or [batch_size, seq_len] categorical IDs
                            (PAD, PA, AP, LATERAL, OTHER). If per-view, expanded to per-patch.
        
        Returns:
            projected: [batch_size, num_queries, output_dim]
        """
        batch_size = image_embeds.shape[0]
        
        # Project visual features
        memory = self.input_proj(image_embeds)  # [batch, seq_len, output_dim]

        # Add view embeddings (not concatenation) if provided
        if view_positions is not None:
            if view_positions.dtype != torch.long:
                view_positions = view_positions.long()
            if view_positions.device != memory.device:
                view_positions = view_positions.to(memory.device)

            if view_positions.shape[1] != image_embeds.shape[1]:
                tokens_per_view = image_embeds.shape[1] // view_positions.shape[1]
                extended_positions = view_positions.repeat_interleave(tokens_per_view, dim=1)
            else:
                extended_positions = view_positions

            extended_positions = extended_positions.clamp(min=0, max=4)
            memory = memory + self.view_embedding(extended_positions)
        
        # Expand queries
        tgt = self.queries.expand(batch_size, -1, -1)  # [batch, num_queries, output_dim]
        
        # Create Key Padding Mask if needed
        # PyTorch Transformer checks `key_padding_mask` where True values are ignored.
        # Our view_mask (if derived from pixel_values) usually has 0 for padding.
        key_padding_mask = None
        if view_mask is not None:
            # Assume strict mask: (B, Num_Views) -> we need (B, Seq_Len)
            # If view_mask is per-view, we expand it to per-patch
            if view_mask.shape[1] != image_embeds.shape[1]:
                 # Logic: tokens_per_view = seq_len // num_views
                 tokens_per_view = image_embeds.shape[1] // view_mask.shape[1]
                 extended_mask = view_mask.repeat_interleave(tokens_per_view, dim=1)
            else:
                 extended_mask = view_mask
            
            # Convert to bool: True = Ignore (Padding), False = Keep
            # If our mask is 1=Valid, 0=Pad:
            key_padding_mask = (extended_mask == 0)

        # Apply Transformer with residual connection for gradient flow
        transformer_out = self.transformer(
            tgt, 
            memory, 
            memory_key_padding_mask=key_padding_mask
        )
        
        # Residual connection: helps gradients flow through the bridge
        # This prevents the "Perceiver gradients near zero" issue
        output = tgt + transformer_out
        
        output = self.norm(self.out_proj(output))
        
        return output

class SigLIPEncoder(nn.Module):
    """
    Wrapper for SigLIP Vision Model.
    """
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def _get_vision_model(self):
        """
        Resolve the underlying vision model, even if `self.model` is wrapped (e.g., PEFT).
        Avoid caching a stale handle so LoRA-wrapped models are actually used at runtime.
        """
        base = self.model
        if hasattr(base, "get_base_model"):
            try:
                base = base.get_base_model()
            except Exception:
                base = self.model

        if hasattr(base, "vision_model"):
            return base.vision_model
        return base
            
    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Accepts (B, C, H, W) or (B, V, C, H, W)
        Returns (B, Seq_Len, Dim)
        """
        vision_model = self._get_vision_model()
        if pixel_values.dim() == 5:
            b, v, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * v, c, h, w)
            outputs = vision_model(pixel_values=pixel_values)
            emb = outputs.last_hidden_state # (B*V, Patch_Seq, Dim)
            
            # Reshape
            emb = emb.view(b, v * emb.shape[1], emb.shape[2]) # (B, V*Patch_Seq, Dim)
            return emb
        
        # Standard Case
        outputs = vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state
