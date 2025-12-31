"""
PEFT (LoRA and QLoRA) configuration utilities.
"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

from utils.config import (
    LORA_R_VISION, LORA_ALPHA_VISION, LORA_R_LLM, LORA_ALPHA_LLM,
    LORA_DROPOUT, LORA_TARGET_MODULES_VISION, LORA_TARGET_MODULES_LLM
)


def get_lora_config(
    r: int = None,
    lora_alpha: int = None,
    lora_dropout: float = LORA_DROPOUT,
    target_modules: list = None,
    model_type: str = "vision"  # "vision" or "llm"
) -> LoraConfig:
    """
    Get LoRA configuration.
    
    Args:
        r: LoRA rank (defaults based on model_type)
        lora_alpha: LoRA alpha parameter (defaults based on model_type)
        lora_dropout: Dropout rate
        target_modules: Modules to apply LoRA to
        model_type: "vision" for SigLIP, "llm" for BioGPT
    
    Returns:
        LoraConfig object
    """
    if model_type == "llm":
        r = r or LORA_R_LLM
        lora_alpha = lora_alpha or LORA_ALPHA_LLM
        target_modules = target_modules or LORA_TARGET_MODULES_LLM
    else:
        r = r or LORA_R_VISION
        lora_alpha = lora_alpha or LORA_ALPHA_VISION
        target_modules = target_modules or LORA_TARGET_MODULES_VISION
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None
    )


def get_qlora_config(
    r: int = None,
    lora_alpha: int = None,
    lora_dropout: float = LORA_DROPOUT,
    target_modules: list = None,
    model_type: str = "vision"
) -> tuple:
    """
    Get QLoRA configuration (LoRA + 4-bit quantization).
    
    Args:
        r: LoRA rank (defaults based on model_type)
        lora_alpha: LoRA alpha parameter (defaults based on model_type)
        lora_dropout: Dropout rate
        target_modules: Modules to apply LoRA to
        model_type: "vision" or "llm"
    
    Returns:
        Tuple of (LoraConfig, BitsAndBytesConfig)
    """
    lora_config = get_lora_config(r, lora_alpha, lora_dropout, target_modules, model_type)
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=["vision_model.head", "head"]
    )
    
    return lora_config, bnb_config


def apply_lora(model: torch.nn.Module, lora_config: LoraConfig):
    """
    Apply LoRA to a model.
    
    Args:
        model: Model to apply LoRA to
        lora_config: LoRA configuration
    
    Returns:
        PEFT model
    """
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def apply_qlora(model: torch.nn.Module, lora_config: LoraConfig):
    """
    Apply QLoRA to a model (LoRA + quantization).
    
    Args:
        model: Model to apply QLoRA to (should be loaded with quantization)
        lora_config: LoRA configuration
    
    Returns:
        PEFT model
    """
    # Custom preparation for SigLIP to avoid NotImplementedError: get_input_embeddings
    # 1. Use prepare_model_for_kbit_training but disable GC to avoid the crash
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    # 2. Manually enable GC
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    # 3. Manually register hooks for input embeddings (SigLIP specific)
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
        
    # Vision embeddings
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings"):
         if hasattr(model.vision_model.embeddings, "patch_embedding"):
             model.vision_model.embeddings.patch_embedding.register_forward_hook(make_inputs_require_grad)
    
    # Text embeddings
    if hasattr(model, "text_model") and hasattr(model.text_model, "embeddings"):
        model.text_model.embeddings.register_forward_hook(make_inputs_require_grad)
    
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model


def count_trainable_parameters(model: torch.nn.Module) -> dict:
    """
    Count trainable parameters in a model.
    
    Args:
        model: Model to count parameters for
    
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = 0
    all_params = 0
    
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        'trainable_params': trainable_params,
        'all_params': all_params,
        'trainable_percentage': 100 * trainable_params / all_params if all_params > 0 else 0
    }

