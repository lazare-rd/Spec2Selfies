import copy
from typing import Dict, Optional, Tuple
from icecream import ic

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from spec2selfies.models.model_utils import (
    FREEZE_STRATEGIES,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
)

class CorrectorEncoderModel(transformers.PreTrainedModel):
    """Embeds text and concats with a provided embedding.

    TODO improve comment here.
    """

    config_class = InversionConfig
    encoder_decoder: transformers.PreTrainedModel

    def __init__(
        self,
        config: InversionConfig,
    ):
        super().__init__(config=config)

        # Backtrack compatibility stuff for old checkpoint
        self.embedder_dim = getattr(config, "embedder_dim", 1801)
        self.bottleneck_dim = getattr(config, "bottleneck_dim", 900)
        self.formula_dim = getattr(config, "formula_dim", 11)
        self.with_formula = getattr(config, "with_formula", False)
        self.with_sep = getattr(config, "with_sep", True)
        #

        self.num_repeat_tokens = config.num_repeat_tokens
        ignore_hypothesis_embedding = config.corrector_ignore_hypothesis_embedding
        self.use_ff_dropout = False
        self.dropout = 0
        if hasattr(config, "use_ff_dropout"):
            self.use_ff_dropout = config.use_ff_dropout
            self.dropout = config.dropout
        self.store_embedder = config.store_embedder
        self.embedder_no_grad = config.embedder_no_grad

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        self.tokenizer = tokenizer
        
        encoder_decoder =  load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )
        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()

        # Do not load embedding model if it will no be used
        self.embedder = None
        self.embedder_tokenizer = None
        
        self.num_repeat_tokens = config.num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.name_or_path = config.model_name_or_path
        self.transform_strategy = config.embedding_transform_strategy
        (
            self.embedding_transform1,
            self.embedding_transform2,
            self.embedding_transform3,
            self.formula_transform1,
            self.formula_transform2,
            self.formula_transform3,
    
        ) = self.get_transform_fn()
            
        if config.torch_dtype:
            print (config.torch_dtype)
            self.embedding_transform1 = self.embedding_transform1.to(dtype=config.torch_dtype)
            self.embedding_transform2 = self.embedding_transform2.to(dtype=config.torch_dtype)
            self.embedding_transform3 = self.embedding_transform3.to(dtype=config.torch_dtype)
            self.formula_transform1 = self.formula_transform1.to(dtype=config.torch_dtype)
            self.formula_transform2 = self.formula_transform2.to(dtype=config.torch_dtype)
            self.formula_transform3 = self.formula_transform3.to(dtype=config.torch_dtype)
        
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding
        # TODO argparse; default to 0?
        self.training_embedding_noise_level = 0
        # self.training_embedding_noise_level = 1e-5  # adding for openai...
        self.use_ln = True
        if self.use_ln:
            self.layernorm = nn.LayerNorm(self.encoder_hidden_dim)
        # print(f"Corrector encoder noise level {self.training_embedding_noise_level}")
    def get_transform_fn(self):
        if self.transform_strategy == "repeat":
            embedding_transform = (
                nn.Sequential(
                    nn.Linear(self.embedder_dim, self.bottleneck_dim),
                    nn.Dropout(self.dropout),
                    nn.GELU(),  
                    nn.Linear(self.bottleneck_dim, self.encoder_hidden_dim * self.num_repeat_tokens),
                )
                for _ in range(3)
            )

            formula_transform = (
                nn.Sequential(
                    nn.Linear(self.formula_dim, self.bottleneck_dim),
                    nn.Dropout(self.dropout),
                    nn.GELU(),  
                    nn.Linear(self.bottleneck_dim, self.encoder_hidden_dim * self.num_repeat_tokens),
                )
                for _ in range(3)
            )
            
            return *embedding_transform, *formula_transform
        
        else :
            raise ValueError(f"This transform strategy '{self.transform_strategy}' is not implemented")



    def get_encoder_embedding(
        self,
        embedding: torch.Tensor,
        hypothesis_spectra: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        formula: Optional[torch.Tensor] = None,
        hypothesis_formula: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, D = embedding.shape

        ic(embedding.shape)
        ic(hypothesis_spectra.shape)
        assert embedding.shape == (batch_size, self.embedder_dim)
        assert hypothesis_spectra.shape == (batch_size, self.embedder_dim)

        if embedding.dtype != self.dtype:
            embedding = embedding.to(self.dtype)

        if (self.training) and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * torch.randn(
                embedding.shape, device=embedding.device
            )
            hypothesis_spectra += self.training_embedding_noise_level * torch.randn(
                hypothesis_spectra.shape, device=hypothesis_spectra.device
            )

        use_formula = formula is not None and hypothesis_formula is not None
        if use_formula:
            if (
                formula.dtype != self.dtype or
                hypothesis_formula.dtype != self.dtype
            ):
                formula = formula.to(self.dtype)
                hypothesis_formula = hypothesis_formula.to(self.dtype)
                
            diff_formula = formula - hypothesis_formula

            formula = self.formula_transform1(formula)
            formula = formula.reshape(
                (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
            )
            #
            diff_formula = self.formula_transform2(diff_formula)
            diff_formula = diff_formula.reshape(
                (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
            )
            #
            hypothesis_formula = self.formula_transform3(hypothesis_formula)
            hypothesis_formula = hypothesis_formula.reshape(
                (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
            )

        if self.ignore_hypothesis_embedding:
            # For "No Feedback" ablation
            hypothesis_spectra = embedding

        diff_embedding = embedding - hypothesis_spectra

        embedding = self.embedding_transform1(embedding)
        embedding = embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        #
        diff_embedding = self.embedding_transform2(diff_embedding)
        diff_embedding = diff_embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        #
        hypothesis_spectra = self.embedding_transform3(hypothesis_spectra)
        hypothesis_spectra = hypothesis_spectra.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        inputs_embeds = self.encoder_decoder.get_encoder().embed_tokens(hypothesis_input_ids)
        ones = torch.ones(
                (batch_size, 1), dtype=torch.long, device=hypothesis_input_ids.device
            )
        # This section allows us to use sep_tokens in between 'pieces' of inputs if we want.
        if self.with_sep:
            sep_token = ones * self.encoder_decoder.config.eos_token_id
            sep_token = self.encoder_decoder.get_encoder().embed_tokens(sep_token)
            
        # Base components
        main_chunks = [embedding, hypothesis_spectra, diff_embedding, inputs_embeds]
        main_repeat = 3 * self.num_repeat_tokens
        
        if self.with_sep:
            # Insert sep tokens between chunks
            def interleave_with_sep(chunks):
                result = []
                for chunk in chunks:
                    result.append(chunk)
                    result.append(sep_token)
                return result
        
            main_chunks = interleave_with_sep(main_chunks)[:-1] # We do not keep the last sep_token
            main_repeat += 4  # for the 3 added sep tokens
            main_chunks.insert(0, sep_token)
            
        # Final assembly
        if use_formula:
            formula_chunks = [formula, hypothesis_formula, diff_formula]
            formula_repeat = 3 * self.num_repeat_tokens
            
            if self.with_sep:
                formula_chunks = interleave_with_sep(formula_chunks)
                formula_repeat += 3
            
            inputs_embeds = torch.cat(formula_chunks + main_chunks, dim=1)
            attention_mask = torch.cat(
                (ones.repeat(1, formula_repeat), ones.repeat(1, main_repeat), hypothesis_attention_mask),
                dim=1,
            )
        else:
            inputs_embeds = torch.cat(main_chunks, dim=1)
            attention_mask = torch.cat(
                (ones.repeat(1, main_repeat), hypothesis_attention_mask),
                dim=1,
            )

        if self.use_ln:
            inputs_embeds = self.layernorm(inputs_embeds)
        
        return (inputs_embeds, attention_mask)

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        return_dict_in_generate: bool = False,
    ) -> torch.Tensor:
        if "max_length" not in generation_kwargs:
            generation_kwargs = copy.copy(
                generation_kwargs
            )  # make a copy so we can edit
            generation_kwargs["max_length"] = inputs.get(
                "input_ids", inputs["embedder_input_ids"]
            ).shape[1]

        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=inputs["spectre"],
            hypothesis_input_ids=inputs["hypothesis_input_ids"],
            hypothesis_attention_mask=inputs["hypothesis_attention_mask"],
            hypothesis_spectra=inputs["hypothesis_spectra"],
            formula=inputs.get("formula"),
            hypothesis_formula=inputs.get("hypothesis_formula"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            ic (inputs_embeds.shape)
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        embedding: torch.Tensor,
        hypothesis_spectra,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        formula: Optional[torch.Tensor] = None,
        hypothesis_formula: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            formula=formula,
            hypothesis_spectra=hypothesis_spectra,
            hypothesis_formula=hypothesis_formula,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
