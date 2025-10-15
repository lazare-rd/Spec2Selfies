import copy
from typing import List, Dict
from icecream import ic

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from spec2selfies.models.config import InversionConfig

import spec2selfies
from spec2selfies.models.model_utils import device
from spec2selfies.models.MolToSpec import MolToSpec
from transformers import BertPreTrainedModel, RobertaConfig, RobertaTokenizerFast

import selfies as sf
from collections import Counter
from rdkit import Chem

ic.disable()

def load_embedder_and_tokenizer():
    model_name = "./data/saved_models/SELFormer_allModel_unfrozen_sid/checkpoint-34500"

    model_class = MolToSpec
    embedder_tokenizer = RobertaTokenizerFast.from_pretrained("./data/RobertaFastTokenizer", do_lower_case=False)
    config = RobertaConfig.from_pretrained(model_name, num_labels=1801)
    embedder = model_class.from_pretrained(model_name, config=config)

    return embedder, embedder_tokenizer


def load_pretrained_corrector(embedder: str) -> spec2selfies.trainers.CorrectorTrainer:
    """
    Gets the CorrectorTrainer object for the given embedder.
    """

    if embedder == "text-embedding-ada-002":
        inversion_model = spec2selfies.models.InversionModel.from_pretrained(
            "jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer"
        )
        model = spec2selfies.models.CorrectorEncoderModel.from_pretrained(
            "jxm/spec2selfies__openai_ada002__msmarco__msl128__corrector"
        )
    elif embedder == "gtr-base":
        inversion_model = spec2selfies.models.InversionModel.from_pretrained(
            "jxm/gtr__nq__32"
        )
        model = spec2selfies.models.CorrectorEncoderModel.from_pretrained(
            "jxm/gtr__nq__32__correct"
        )
    elif embedder == "SELFormer":
        config = InversionConfig.from_pretrained(
            "./data/saved_models/Inverter__30epochs__encoder_decoder_frozen/checkpoint-15411"
        )

        # backward compatibiliy stuff...
        config.store_embedder = False
        config.with_sep = True
        config.with_formula = False
        config.ATOM_LIST = []
        config.embedder_dim = 1801
        config.bottleneck_dim = 1801
        config.formula_dim = 11
        
        inversion_model = spec2selfies.models.InversionModel.from_pretrained(
            "./data/saved_models/Inverter__30epochs__encoder_decoder_frozen/checkpoint-15411",
            config=config
        )
        
        if not hasattr(config, "use_ff_dropout"):
            config.use_ff_dropout=True
            config.dropout = 0.1
            
        model = spec2selfies.models.CorrectorEncoderModel.from_pretrained(
            "./data/saved_models/Corrector__30epoch__lora"
        )
        
    elif embedder == "SELFormer_WithFormula":
        config = InversionConfig.from_pretrained("./data/saved_models/Inverter_withFormulas")
        config.store_embedder = False
        config.with_sep = True
        if not hasattr(config, "use_ff_dropout"):
            config.use_ff_dropout=True
            config.dropout = 0.1

        inversion_model = spec2selfies.models.InversionModel.from_pretrained(
            "./data/saved_models/Inverter_withFormulas",
            config=config,
        )
        model = spec2selfies.models.CorrectorEncoderModel.from_pretrained(
            "./data/saved_models/Corrector_with_Formulas_80M"
        )
    elif embedder == "SELFormer_PerfectModel":
        config = InversionConfig.from_pretrained(
            "./data/saved_models/Inverter_PerfectModel"
        )
        config.store_embedder = False
        config.with_sep = True
        config.use_perfect_model = True
        config.with_formula = False
        config.ATOM_LIST = []
        
        inversion_model = spec2selfies.models.InversionModel.from_pretrained(
            "./data/saved_models/Inverter_PerfectModel",
            config=config,
        )

        if not hasattr(config, "use_ff_dropout"):
            config.use_ff_dropout=True
            config.dropout = 0.1
            
        model = spec2selfies.models.CorrectorEncoderModel.from_pretrained(
            "./data/saved_models/Corrector_PerfectModel_80M"
        )
    elif embedder == "SELFormer_PerfectModel_WithFormula":
        config = InversionConfig.from_pretrained(
            "./data/saved_models/Inverter_PerfectModel_withFormula"
        )
        config.store_embedder = False
        config.with_sep = True
        config.use_perfect_model = True
        if not hasattr(config, "use_ff_dropout"):
            config.use_ff_dropout=True
            config.dropout = 0.1

        inversion_model = spec2selfies.models.InversionModel.from_pretrained(
            "./data/saved_models/Inverter_PerfectModel_withFormula",
            config=config,
        )
        model = spec2selfies.models.CorrectorEncoderModel.from_pretrained(
            "./data/saved_models/Corrector_PerfectModel_WithFormula_80M"
        )
    else:
        raise NotImplementedError(f"embedder `{embedder}` not implemented")

    return load_corrector(inversion_model, model)


def load_corrector(
    inversion_model: spec2selfies.models.InversionModel,
    corrector_model: spec2selfies.models.CorrectorEncoderModel,
) -> spec2selfies.trainers.CorrectorTrainer:
    """Load in the inversion and corrector models

    Args:
        inversion_model (spec2selfies.models.InversionModel): _description_
        corrector_model (spec2selfies.models.CorrectorEncoderModel): _description_

    Returns:
        spec2selfies.trainers.CorrectorTrainer: Corrector model to invert an embedding back to text
    """

    inversion_trainer = spec2selfies.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )

    # backwards compatibility stuff
    corrector_model.config.dispatch_batches = None
    corrector = spec2selfies.trainers.CorrectorTrainer(
        model=corrector_model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=spec2selfies.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    return corrector


def invert_embeddings(
    embeddings: torch.Tensor,
    corrector: spec2selfies.trainers.CorrectorTrainer,
    gen_kwargs: dict = None,
    num_steps: int = None,
    sequence_beam_width: int = 0,
    sim_fn: str = None,
    with_formula: bool = False,
    selfies: str = None,
    alpha: float = None,
) -> List[str]:
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()

    if gen_kwargs is None:
        gen_kwargs = copy.copy(corrector.gen_kwargs)
    
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 128
    gen_kwargs["pad_token_id"] = corrector.tokenizer.pad_token_id
    gen_kwargs["eos_token_id"] = corrector.tokenizer.eos_token_id
    gen_kwargs["bos_token_id"] = corrector.tokenizer.bos_token_id

    if sim_fn is not None:
        corrector.sim_fn_str = sim_fn

    if with_formula and selfies is not None:
        formula = corrector.embedder.get_formula_vec(selfies)
    else:
        formula = None

    if num_steps is None:
        assert (
            sequence_beam_width == 0
        ), "can't set a nonzero beam width without multiple steps"

        regenerated = corrector.inversion_trainer.generate(
            inputs={
                "spectre": embeddings,
                "formula": formula,
            },
            generation_kwargs=gen_kwargs,
        )
    else:
        corrector.return_best_hypothesis = sequence_beam_width > 0
        regenerated = corrector.generate(
            inputs={
                "spectre": embeddings,
                "formula": formula,
            },
            generation_kwargs=gen_kwargs,
            num_recursive_steps=num_steps,
            sequence_beam_width=sequence_beam_width,
            alpha=alpha,
        )

    output_strings = corrector.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings


def invert_embeddings_eval(
    inputs: Dict[str, torch.Tensor],
    corrector: spec2selfies.trainers.CorrectorTrainer,
    gen_kwargs: dict = None,
    num_steps: int = None,
    sequence_beam_width: int = 0,
    sim_fn: str = None,
    alpha: float = None,
) -> List[str]:
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()
    with_formula = corrector.model.with_formula

    if gen_kwargs is None:
        gen_kwargs = copy.copy(corrector.gen_kwargs)
    
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 200
    gen_kwargs["pad_token_id"] = corrector.tokenizer.pad_token_id
    gen_kwargs["eos_token_id"] = corrector.tokenizer.eos_token_id
    gen_kwargs["bos_token_id"] = corrector.tokenizer.bos_token_id 

    corrector.return_best_hypothesis = sequence_beam_width > 0
    if sim_fn is not None:
        corrector.sim_fn_str = sim_fn

    (
        text_ids,
        hypothesis_spectras,
        hypothesis_embeddings,
        hypothesis_formulas,
        scores,
        initial_score
    ) = corrector.generate_with_hypotheses(
        inputs=inputs,
        generation_kwargs=gen_kwargs,
        num_recursive_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
        _eval=True,
        alpha=alpha,
    )
    
    # Performing all padding operations on CPU
    text_ids = [t.cpu() for t in text_ids]
    hypothesis_spectras = [t.cpu() for t in hypothesis_spectras]
    hypothesis_embeddings = [t.cpu() for t in hypothesis_embeddings]
    if with_formula:
        hypothesis_formulas = [t.cpu() for t in hypothesis_formulas]
    scores = [t.cpu() for t in scores]
    initial_score = initial_score.cpu()
    
    # Padding the ids to max_length to be able to log them into a tensor buffer
    steps = len(text_ids)
    batch, beam, _ = text_ids[1].shape

    bm_text_ids =[text_ids.pop(0).squeeze(1)]
    bm_spectra = hypothesis_spectras.pop(0).squeeze(1)
    if with_formula:
        bm_formula = hypothesis_formulas.pop(0).squeeze(1)
    
    text_ids.insert(0, torch.zeros(batch, beam, gen_kwargs["max_length"]))
    bm_text_ids.insert(0, torch.zeros(batch, 514))
        
    for i, t in enumerate(text_ids):
        text_ids[i] = t.reshape(batch*beam, -1)

    all_ids = [ids for step in text_ids for ids in step]
    bm_text_ids = [ids for step in bm_text_ids for ids in step]
    
    padded_ids = pad_sequence(all_ids, batch_first=True, padding_value=corrector.tokenizer.pad_token_id)
    padded_bm_text_ids = pad_sequence(bm_text_ids, batch_first=True, padding_value=corrector.tokenizer.pad_token_id)
    # We reshape at steps+1 to take into account the dummy max_length tensor that we will dump afterwards
    padded_ids = padded_ids.reshape(steps, batch, beam, -1).permute(1, 0, 2, 3)
    padded_ids = padded_ids[:, 1:]
    padded_bm_text_ids = padded_bm_text_ids.reshape(2, batch, -1).permute(1, 0, 2)
    padded_bm_text_ids = padded_bm_text_ids[:, 1:].squeeze(1)
    
    # We turn lists of tensors into one big tensor
    hypothesis_embeddings = torch.cat(hypothesis_embeddings, dim=1).reshape(batch, steps-1, beam, -1)
    hypothesis_spectras = torch.cat(hypothesis_spectras, dim=1).reshape(batch, steps-1, beam, -1)
    if with_formula:
        hypothesis_formulas = torch.cat(hypothesis_formulas, dim=1).reshape(batch, steps-1, beam, -1)
    scores = torch.cat(scores, dim=1).reshape(batch, steps-1, beam)
    
    outputs = {
        "text_ids" : padded_ids,
        "spectras" : hypothesis_spectras,
        "embeddings" : hypothesis_embeddings,
        "bm_text_ids": padded_bm_text_ids,
        "bm_spectra": bm_spectra,
        "scores" : scores,
        "initial_score": initial_score,
    }

    if with_formula:
        outputs = {
            **outputs,
            **{
            "bm_formula": bm_formula,
            "formulas": hypothesis_formulas,
            }
        }

    return outputs


def invert_embeddings_and_return_hypotheses(
    embeddings: torch.Tensor,
    corrector: spec2selfies.trainers.CorrectorTrainer,
    gen_kwargs: dict = None,
    num_steps: int = None,
    sequence_beam_width: int = 0,
    sim_fn: str = None,
) -> List[str]:
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()

    if gen_kwargs is None:
        gen_kwargs = copy.copy(corrector.gen_kwargs)
    
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 128
    gen_kwargs["pad_token_id"] = corrector.tokenizer.pad_token_id
    gen_kwargs["eos_token_id"] = corrector.tokenizer.eos_token_id
    gen_kwargs["bos_token_id"] = corrector.tokenizer.bos_token_id 

    corrector.return_best_hypothesis = sequence_beam_width > 0
    if sim_fn is not None:
        corrector.sim_fn_str = sim_fn

    regenerated, hypotheses, _, _, _ = corrector.generate_with_hypotheses(
        inputs={
            "spectre": embeddings,
        },
        generation_kwargs=gen_kwargs,
        num_recursive_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )
    output_strings = []
    for hypothesis in regenerated:
        output_strings.append(
            corrector.tokenizer.batch_decode(hypothesis, skip_special_tokens=True)
        )

    return output_strings, hypotheses


def invert_strings(
    strings: List[str],
    corrector: spec2selfies.trainers.CorrectorTrainer,
    num_steps: int = None,
    sequence_beam_width: int = 0,
) -> List[str]:

    embedder, embedder_tokenizer = load_embedder_and_tokenizer()
    
    inputs = embedder_tokenizer(
        strings,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    inputs = inputs.to(device)
    embedder.to(device)
    with torch.no_grad():
        frozen_embeddings = embedder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )[0]

    return invert_embeddings_and_return_hypotheses(
        embeddings=frozen_embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )
