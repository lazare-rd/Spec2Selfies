from typing import Callable, Dict

import torch
import transformers
import logging

from vec2text.models import InversionModel


def tokenize_function(
    tokenizer: transformers.PreTrainedTokenizer,
    text_column_name: str,
    max_seq_length: int = None,
    padding_strategy: bool = None,
    embedder_tokenizer: transformers.PreTrainedTokenizer = None,
) -> Callable[[Dict], Dict]:
    
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
            
        texts = examples[text_column_name]
        seq_len = max_seq_length if max_seq_length is not None else tokenizer.model_max_length
        pad = padding_strategy if padding_strategy is not None else False
        
        output = tokenizer(
            texts,
            padding=pad,
            truncation=True,
            max_length=seq_len,
        )
        
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        if embedder_tokenizer is not None :
            embedder_output = embedder_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            
            embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

            return {**output, **embedder_output}

        return output

    return tokenize_function_inner


def embed_dataset_batch(model: InversionModel, batch: Dict) -> Dict:
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        batch["frozen_embeddings"] = model.call_embedding_model(**emb_input_ids)
    return batch


def get_tokenizer_mapping(
    lm: str, inverter: str, inverter_vocab_size: int
) -> torch.Tensor:
    """Computes the mapping from token outputs in `lm`'s vocabulary to those in `inverter's
    vocabulary. Makes some assumptions about spacing.
    """
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm)
    inverter_tokenizer = transformers.AutoTokenizer.from_pretrained(inverter)

    lm_vocab = lm_tokenizer.vocab
    mapping = torch.zeros(len(lm_vocab), dtype=torch.long)
    for k, idx in lm_tokenizer.vocab.items():
        # We replace space tokens with nothing and allow the call to
        # inverter_tokenizer.decode to determine this. We also
        # filter out 2 and 3 as first tokens which are extremely common
        # when the T5 tokenizer processes unicode. (These are hacks
        # specific to the LLAMA-T5 lm-inverter pairing, and it would
        # be better to find an automated wa to do this later.)
        mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[0]
        if mapping[idx] in [2, 3]:
            mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[1]

    preservation = len(set(mapping.tolist())) / len(lm_vocab)
    print(
        f"Mapped tokenizer {lm} to {inverter}. Preserved {preservation*100:.1f}% of unique tokens."
    )
    return mapping
