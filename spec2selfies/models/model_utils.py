import os
from typing import Any, Dict, Optional, List
from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn
import transformers
from spec2selfies.models.MolToSpec import MolToSpec

from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
)

from transformers import RobertaTokenizerFast

EMBEDDER_MODEL_NAMES = [
    "SELFormer"
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "None"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


def get_device():
    """
    Function that checks
    for GPU availability and returns
    the appropriate device.
    :return: torch.device
    """
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.max(dim=1).values
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def stack_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((B, S * D))  # stack along seq length
    assert pooled_outputs.shape == (B, S * D)
    return pooled_outputs


def load_embedder_and_tokenizer(
        name: str,
        with_formula: Optional[bool] = False,
        ATOM_LIST: Optional[List[str]] = None,
        **kwargs
):
   
    if name == 'SELFormer':
        model_name = "./data/saved_models/SELFormer_allModel_unfrozen_sid/checkpoint-34500"
        
        config = RobertaConfig.from_pretrained(model_name, num_labels=1801)
        config.with_formula = with_formula
        config.ATOM_LIST = ATOM_LIST
        config.tokenizer_name = "./data/RobertaFastTokenizer"
        model = MolToSpec.from_pretrained(model_name, config=config)
        tokenizer = RobertaTokenizerFast.from_pretrained("./data/RobertaFastTokenizer", do_lower_case=False)

    else :
        print (f"This embedder {name} cannot be loaded")
        

    return model, tokenizer


def load_encoder_decoder(model_name: str, lora: bool = False) -> transformers.PreTrainedModel:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

    if lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)

    return model


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer
