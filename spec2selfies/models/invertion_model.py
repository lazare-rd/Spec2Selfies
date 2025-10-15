import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
)

from transformers import RobertaTokenizerFast

import spec2selfies.models.MolToSpec as mts

from spec2selfies.models.config import InversionConfig
from spec2selfies.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)

from icecream import ic

class InversionModel(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    config_class = InversionConfig
    embedder: mts.MolToSpec
    embedder_tokenizer: transformers.RobertaTokenizerFast  # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.AutoTokenizer  # encoder_decoder's tokenizer
    embedding_transform: nn.Module  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int  # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    embedded_tokens: torch.Tensor  # used for decoding
    embedder_model_api: Optional[str]
    with_formula: bool

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n
        store_embedder = config.store_embedder
        self.embedder_no_grad = config.embedder_no_grad
        self.with_formula = config.with_formula

        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

        # Do not load embedding model if it will no be used
        self.embedder = None
        self.embedder_tokenizer = None
        if store_embedder:
            embedder, embedder_tokenizer = load_embedder_and_tokenizer(
                name=config.embedder_model_name,
                #torch_dtype=config.embedder_torch_dtype,
                with_formula=config.with_formula,
                ATOM_LIST=config.ATOM_LIST,
            )
            
            self.embedder = embedder
            self.embedder_tokenizer = embedder_tokenizer

            if self.embedder_no_grad:
                for param in self.embedder.parameters():
                    param.requires_grad = False
                self.embedder.eval()

        self.encoder_decoder = encoder_decoder
        self.embedder_is_decoder = False
        
        self.num_repeat_tokens = config.num_repeat_tokens        
        self.transform_strategy = "repeat"  # "none" # "repeat"
        self.embedder_dim = config.embedder_dim                           # size of a spectra
        self.bottleneck_dim = config.bottleneck_dim       
        self.formula_dim = config.formula_dim
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        
        self.embedding_transform, self.formula_transform = self.get_transform_fn()
        ic (next(self.embedding_transform.parameters()).device)
        ic (next(self.formula_transform.parameters()).device)
       
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)
        ######################################################
        self.tokenizer = tokenizer
        self.freeze(freeze_strategy=config.freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level")

    def get_transform_fn(self):
        if self.transform_strategy == "repeat":
            embedding_transform = nn.Sequential(
                nn.Linear(self.embedder_dim, self.bottleneck_dim),
                nn.Dropout(self.encoder_decoder.config.dropout),
                nn.GELU(),  
                nn.Linear(self.bottleneck_dim, self.encoder_hidden_dim * self.num_repeat_tokens),
            )
            formula_transform = nn.Sequential(
                nn.Linear(self.formula_dim, self.bottleneck_dim),
                nn.Dropout(self.encoder_decoder.config.dropout),
                nn.GELU(),  
                nn.Linear(self.bottleneck_dim, self.encoder_hidden_dim * self.num_repeat_tokens),
            )
            return embedding_transform, formula_transform

        else :
            raise ValueError(f"This transform strategy '{self.transform_strategy}' is not implemented")
        
    def _freeze_encoder(self):
        for name, param in self.encoder_decoder.named_parameters():
            if ("encoder" in name):  
                param.requires_grad = False

    def _freeze_decoder(self):
        for name, param in self.encoder_decoder.named_parameters():
            # in this case, freeze embeddings too
            if ("decoder" in name or "shared" in name):  
                param.requires_grad = False

    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
        elif freeze_strategy == "None":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device


    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedder = self.embedder
        if self.embedder_no_grad:
            embedder.eval()

        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return torch.zeros(
                (batch_size, self.embedder_dim),
                dtype=torch.float32,
                device=self.embedder_device,
            )
        
        elif self.with_formula:
            embeddings, _, formulas = embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = (embeddings, formulas)
            
        else:
            embeddings, _, _ = embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = (embeddings, None)

        if self.training and self.noise_level > 0:
            embeddings += self.noise_level * torch.randn(
                embeddings.shape, device=embeddings.device
            )
            embeddings = (embeddings, None)
            
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        embeddings: Optional[torch.Tensor] = None,
        formulas: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("** embed_and_project")
        assert not ((embedder_input_ids is None) and (embeddings is None))
        if embeddings is not None:
            assert len(embeddings.shape) == 2 # batch by d
            if self.with_formula:
                assert formulas is not None
                assert len(formulas.shape) == 2
        
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings, formulas = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
        else:
            embeddings, formulas = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
        
        if embeddings.dtype != self.dtype:
            embeddings = embeddings.to(self.dtype)
            
        repeated_embeddings = self.embedding_transform(embeddings)               
        # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
        embeddings = repeated_embeddings.reshape(
            (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
        )
        
        if self.with_formula:
            if formulas.dtype != self.dtype:
                formulas = formulas.to(self.dtype)
            repeated_formulas = self.formula_transform(formulas)
            formulas = repeated_formulas.reshape(
                (*repeated_formulas.shape[:-1], self.num_repeat_tokens, -1)
            )
            embeddings = torch.cat((embeddings, formulas), dim=1)
    
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            embeddings=inputs.get("spectre"),
            formulas=inputs.get("formula")
        )
        return self.encoder_decoder.generate(
            #required: input embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # optional: input IDs (for starting generation).
            # typically not set unless generating prefixes for
            # reranking.
            **generation_kwargs,
            )

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        spectre: Optional[torch.Tensor] = None,
        formula: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            embeddings=spectre,
            formulas=formula,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
