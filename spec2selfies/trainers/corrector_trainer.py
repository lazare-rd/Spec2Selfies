import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import random as rd
from icecream import ic

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy

from spec2selfies.models.corrector_model import CorrectorEncoderModel
from spec2selfies.models.model_utils import freeze_params, load_embedder_and_tokenizer
from spec2selfies.run_args import TrainingArguments
from spec2selfies.utils import dataset_map_multi_worker

from .base import BaseTrainer
from .invertion_trainer import InversionTrainer

logger = logging.getLogger(__name__)
#ic.disable()

class SISMeasure(nn.Module):

    def forward(self, model_spectra, target_spectra, conv=True, norm=True):
        
        target_spectra_conv = self.gaussian_filter1d_torch(target_spectra, 5)                     
        model_spectra_conv = y.numpy() if not conv else self.gaussian_filter1d_torch(model_spectra, 5)  
    
        if norm:
            target_spectra_conv = nn.functional.normalize(target_spectra_conv, p=1)  
            model_spectra_conv = nn.functional.normalize(model_spectra_conv, p=1)
        else: # back to torch (not optimal)
            target_spectra_conv = torch.from_numpy(target_spectra_conv)
            model_spectra_conv = torch.from_numpy(model_spectra_conv)

        loss = torch.ones_like(target_spectra)

        loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) \
                + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
        
        loss = torch.sum(loss, dim=-1)
        
        loss = 1/(1+loss)
        
        return loss

    def gaussian_filter1d_torch(self, x, sigma):
        if sigma == 0:
            return x

        radius = int(4 * sigma)
        kernel_size = 2 * radius + 1
        
        coords = torch.arange(kernel_size, device=x.device) - radius
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        kernel = kernel.view(1, 1, -1)

        # When there is beam dim
        if len(x.shape) > 2: 
            B, beam, signal = x.shape
            
            x = x.reshape(B * beam, 1, signal)
            
            x = F.pad(x, (radius, radius), mode='reflect')
            x = F.conv1d(x, kernel, padding=radius, groups=1)
            x = x[:, :, radius:-radius]
            
            x = x.reshape(B, beam, signal)

        # When there's not
        else :
            x = x.unsqueeze(1)
            
            x = F.pad(x, (radius, radius), mode='reflect')
            x = F.conv1d(x, kernel, padding=radius, groups=1)
            x = x[:, :, radius:-radius]

            x = x.squeeze(1)
        
        return x

 

class CorrectorTrainer(BaseTrainer):
    """Trains an encoder model to generate embeddings that recursively correct of an
    InversionTrainer.
    """

    train_dataset: datasets.Dataset
    eval_dataset: datasets.Dataset
    # TODO: don't assume that the encoder has to have the same tokenizer as the encoder_decoder
    # or embedder model.

    _hypothesis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    # If set, only take hypothesis if it improves our distance to ground-truth.
    return_best_hypothesis: bool = False

    # Initialize from this hypothesis, if set
    initial_hypothesis_str: Optional[str] = None

    def __init__(
        self,
        model: CorrectorEncoderModel,
        inversion_trainer: InversionTrainer,
        args: Optional[TrainingArguments],
        **kwargs,
    ):
        # Freeze other model params
        super().__init__(
            model=model,
            args=args,
            **kwargs,
        )

        self.tokenizer = self.model.tokenizer
        self.inversion_trainer = inversion_trainer
        if inversion_trainer is not None:
            freeze_params(inversion_trainer.model)
            # We're training this corrector model to correct outputs from
            # a model trained & loaded via the inversion trainer.
            self.inversion_trainer.model.use_frozen_embeddings_as_input = True
            self.tokenizer = self.inversion_trainer.model.tokenizer

            # Need to train with same device as the inversion model to avoid weird errors.
            assert self.args.fp16 == self.inversion_trainer.args.fp16
            assert self.args.bf16 == self.inversion_trainer.args.bf16

        self.with_formula = self.model.with_formula
        model, tokenizer = load_embedder_and_tokenizer(
            'SELFormer',
            with_formula=self.with_formula,
            ATOM_LIST=['I', 'C', 'S', 'N', 'Br', 'H', 'O', 'Si', 'Cl', 'P', 'F']
        )

        device = next(self.model.parameters()).device
        self.embedder = model.to(device)
        self.embedder_tokenizer = tokenizer

        for param in self.embedder.parameters():
            param.requires_grad = False
        self.embedder.eval()
        
        self.initial_hypothesis_str = None

        # Number of steps of self-correction
        
        self.num_gen_recursive_steps = 1
        self.sequence_beam_width = 1

        # If set, return closest (in embedding space) hypothesis we see during generation
        self.return_best_hypothesis = True
        self.sim_fn_str = "sis"


    def evaluation_loop(
        self, dataloader: torch.utils.data.DataLoader, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """

        output = super().evaluation_loop(dataloader=dataloader, *args, **kwargs)
        """
        random_inputs = []
        preds = []
        for _ in range (3):
            random_inputs.append(
                self.eval_dataset[rd.randint(0, len(self.eval_dataset))]
            )
        
        with torch.no_grad():
            for i in range(len(random_inputs)):
                inputs = self.data_collator([random_inputs[i]])
                print (inputs)
                for key in inputs.keys():
                    inputs[key] = inputs[key].to("cuda")
                preds.append(
                    self.generate(
                        inputs=inputs,
                        generation_kwargs=self.gen_kwargs
                    ).squeeze(0)
                )
        preds_decoded = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )

        true_input_ids = [input['input_ids'] for input in random_inputs]

        true_decoded = self.tokenizer.batch_decode(
             true_input_ids, skip_special_tokens=True
        )

        for i in range (len(random_inputs)):
            print (f"[TRUE] : {true_decoded[i]}")
            print (f"[PRED] : {preds_decoded[i]}")
            print ('\n')
        """

        return output

    def _precompute_hypothesis_and_embedding(
        self,
        ds_inputs: Dict[str, torch.Tensor],
        collator=None,
    ) -> Dict[str, torch.Tensor]:
        assert not self.model.training
        
        inputs = collator.tokenizer.pad(
            {k: v for k, v in ds_inputs.items() if k != "labels"},
            padding=collator.padding,
            max_length=collator.max_length,
            pad_to_multiple_of=collator.pad_to_multiple_of,
            return_tensors=collator.return_tensors,
        ).to(self.args.device)

        (
            spectre,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_spectra,
        ) = self._get_hypothesis_uncached(inputs=inputs)
        ds_inputs["spectre"] = spectre.cpu()
        ds_inputs["hypothesis_spectra"] = hypothesis_spectra.cpu()

        # cut padding so we can batch by length later
        ds_inputs["hypothesis_input_ids"] = []
        ds_inputs["hypothesis_attention_mask"] = []
        for input_ids, attention_mask in zip(
            hypothesis_input_ids.cpu(), hypothesis_attention_mask.cpu()
        ):
            num_tokens = attention_mask.sum()
            ds_inputs["hypothesis_input_ids"].append(input_ids[: num_tokens + 1])
            ds_inputs["hypothesis_attention_mask"].append(
                attention_mask[: num_tokens + 1]
            )
        print("input_ids[0]:", self.tokenizer.decode(ds_inputs["input_ids"][0]))
        print(
            "hypothesis_input_ids[0]:",
            self.tokenizer.decode(ds_inputs["hypothesis_input_ids"][0]),
        )
        return ds_inputs

    def _preprocess_dataset_hypotheses(
        self, dataset: datasets.Dataset, filter_correct_examples: bool = False
    ) -> Tuple[datasets.Dataset, str]:
        #
        # In each model directory, we store a copy of the dataset with hypotheses
        # generated by the model that's checkpointed in this directory. This
        # won't scale well, but hopefully we don't do this with too many models,
        # and precomputing 5M hypotheses on A100 takes ~8 hours, so they're worth
        # storing.
        #_preprocess_dataset_hypotheses
        # Note that the dataset fingerprint changes with calls to select()
        # so we won't overwrite the big dataset files when we use tiny subsets
        # during testing.
        cache_dir = os.environ["VEC2TEXT_CACHE"]
        assert os.path.exists(cache_dir)
        ####
        cache_path = os.path.join(cache_dir, f"{dataset._fingerprint}_hypotheses.cache")
        if not os.path.exists(cache_path):
            print(f"\t[{dataset.builder_name}] Saving hypotheses to path {cache_path}")

            dataset = dataset_map_multi_worker(
                dataset=dataset,
                map_fn=functools.partial(
                    self._precompute_hypothesis_and_embedding,
                    collator=self.data_collator,
                ),
                batched=True,
                batch_size=(self.args.train_batch_size * 2),
                desc="Precomputing hypotheses for data",
            )

            if filter_correct_examples:
                old_length = len(dataset)

                def embedding_is_not_correct(ex):
                    return (
                        ~torch.isclose(
                            ex["spectre"].to(self.args.device),
                            ex["hypothesis_spectra"].to(self.args.device),
                        ).all(dim=1)
                    ).tolist()

                dataset = dataset.filter(
                    embedding_is_not_correct,
                    batched=True,
                    batch_size=1024,
                )
                print(f"filtered {old_length} datapoints to {len(dataset)}")
            dataset.save_to_disk(cache_path)
        else:
            logging.info("Loading hypotheses from path %s", cache_path)
            print(
                f"\t[{dataset.builder_name}] Loading hypotheses from path {cache_path}"
            )
            dataset = datasets.load_from_disk(cache_path)
        dataset.set_format("pt")
        return dataset, cache_path

    def precompute_hypotheses(self) -> None:
        """Generates and embeds hypotheses using `self.inversion_trainer`.

        Returns path to precomputed-and-saved train dataset, which is sometimes
        useful for outside processes.
        """
        logger.info("Precomputing frozen embedding & hypotheses before training")

        self.train_dataset, train_cache_path = self._preprocess_dataset_hypotheses(
            dataset=self.train_dataset, filter_correct_examples=True
        )
        for k, v in self.eval_dataset.items():
            self.eval_dataset[k], _ = self._preprocess_dataset_hypotheses(
                dataset=v, filter_correct_examples=False
            )

    def _inner_training_loop(self, *args, **kwargs):
        # Don't let tokenizers run in parallel mode.
        # os.environ["TOKENIZERS_PARALLELISM"] = "False"

        if self.inversion_trainer is None:
            return super()._inner_training_loop(*args, **kwargs)
        else:
            self.model.eval()
            self.model.to(self.args.device)
            self.inversion_trainer.model.to(next(self.model.parameters()).device)
            self.precompute_hypotheses()
            self.model.train()
            self.inversion_trainer.model.cpu()
            return super()._inner_training_loop(*args, **kwargs)


    # For now is broken because of all the changes in generate_with_beam
    # Copy on generate with hypothesis to fix
    def generate(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,
        alpha: float = None,
    ) -> torch.Tensor:
        """Generates text using self-correction.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (torch.Tensor): ids of generated text
        """
        try:
            spectre = inputs["spectre"]
            formula = inputs["formula"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_spectra = inputs["hypothesis_spectra"]
            hypothesis_formula = inputs["hypothesis_formula"]
        except KeyError:
            (
                spectre,
                formula,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_spectra,
                hypothesis_formula,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # Add beam dimension:
        #       (batch, ...) -> (batch, beam, ...)
        inputs["spectre"] = spectre
        inputs["formula"] = formula
        inputs["hypothesis_input_ids"] = hypothesis_input_ids
        inputs["hypothesis_attention_mask"] = hypothesis_attention_mask
        inputs["hypothesis_spectra"] = hypothesis_spectra
        inputs["hypothesis_formula"] = hypothesis_formula
        # print("generating with sequence_beam_width:", (sequence_beam_width or self.sequence_beam_width))

        num_recursive_steps = num_recursive_steps or self.num_gen_recursive_steps
        sequence_beam_width = sequence_beam_width or self.sequence_beam_width
        num_recursive_steps_so_far = 0

        total_best_scores_seen = None  # Track best scores for early stopping
        while num_recursive_steps >= 1:
            (
                gen_text_ids,
                hypothesis_spectra,
                _,
                hypothesis_formula,
                best_scores,
                _
            ) = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=num_recursive_steps,
                num_recursive_steps_so_far=num_recursive_steps_so_far,
                sequence_beam_width=sequence_beam_width,
                _eval = True,
                alpha=alpha
            )

            ic (gen_text_ids.shape)
            batch_size, sequence_beam_width, _ = gen_text_ids.shape
        
            gen_text_ids = gen_text_ids.reshape(
                (batch_size * sequence_beam_width, -1)
            )
            hypothesis_spectra = hypothesis_spectra.reshape(
                (batch_size * sequence_beam_width, -1)
            )
            
            if hypothesis_formula is not None:
                hypothesis_formula = hypothesis_formula.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
            
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_spectra"] = hypothesis_spectra
            inputs["hypothesis_formula"] = hypothesis_formula
            # step counters
            num_recursive_steps -= 1
            num_recursive_steps_so_far += 1
            # early stopping
            if best_scores is not None:
                if (total_best_scores_seen is not None) and torch.isclose(
                    best_scores, total_best_scores_seen, atol=1e-3
                ):
                    print(
                        "scores stopped increasing! stopping early after",
                        num_recursive_steps_so_far,
                        "steps",
                    )
                    break
                best_scores = total_best_scores_seen

        return gen_text_ids

    def generate_with_hypotheses(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,
        _eval: bool = False,
        alpha: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates text using self-correction. Works exactly like generate(), but returns all the intermediate hypotheses steps.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (List[torch.Tensor]): ids of generated text, for each hypothesis sequence
            hypothesis_spectras (List[torch.Tensor]): embeddings of each hypothesis sequence
        """
        try:
            spectre = inputs["spectre"]
            formula = inputs["formula"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_spectra = inputs["hypothesis_spectra"]
            hypothesis_formula = inputs["hypothesis_formula"]
        except KeyError:
            (
                spectre,
                formula,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_spectra,
                hypothesis_formula,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # Add beam dimension:
        #       (batch, ...) -> (batch, beam, ...)
        inputs["spectre"] = spectre
        inputs["formula"] = formula
        inputs["hypothesis_input_ids"] = hypothesis_input_ids
        inputs["hypothesis_attention_mask"] = hypothesis_attention_mask
        inputs["hypothesis_spectra"] = hypothesis_spectra
        inputs["hypothesis_formula"] = hypothesis_formula

        num_recursive_steps = num_recursive_steps or self.num_gen_recursive_steps
        sequence_beam_width = sequence_beam_width or self.sequence_beam_width
        num_recursive_steps_so_far = 0

        total_best_scores_seen = None  # Track best scores for early stopping
        sim_fn_spectra = self.get_sim_fn()

        ic (inputs["hypothesis_spectra"].shape)
        ic (inputs["spectre"].shape)
        initial_score = sim_fn_spectra(
            inputs["hypothesis_spectra"],
            inputs["spectre"]
        )
        ground_truth_embedding = inputs["hypothesis_spectra"]
        
        hypothesis_embeddings = [] # We do not yet have an hypothesis embedding
        beam_scores = []
        hypothesis_spectras = [ground_truth_embedding]  # Track hypothesis spectras
        hypothesis_formulas = [inputs["hypothesis_formula"]] # Track hypothesis formula
        hypothesis_ids = [inputs["hypothesis_input_ids"]]  # Track hypothesis ids

        while num_recursive_steps >= 1:
            (
                gen_text_ids,
                hypothesis_spectra,
                hypothesis_embedding,
                hypothesis_formula,
                best_scores,
                best_scores_idx
            ) = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=num_recursive_steps,
                num_recursive_steps_so_far=num_recursive_steps_so_far,
                sequence_beam_width=sequence_beam_width,
                _eval = _eval,
                alpha = alpha,
            )
            batch_size, sequence_beam_width, _ = gen_text_ids.shape
            
            hypothesis_ids.append(gen_text_ids)
            hypothesis_spectras.append(hypothesis_spectra)
            hypothesis_embeddings.append(hypothesis_embedding)
            hypothesis_formulas.append(hypothesis_formula)
            beam_scores.append(best_scores)
            
            gen_text_ids = gen_text_ids.reshape(
                (batch_size * sequence_beam_width, -1)
            )
            hypothesis_spectra = hypothesis_spectra.reshape(
                (batch_size * sequence_beam_width, -1)
            )
            hypothesis_embedding = hypothesis_embedding.reshape(
                (batch_size * sequence_beam_width, -1)
            )
            if hypothesis_formula is not None:
                hypothesis_formula = hypothesis_formula.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
            ic (gen_text_ids.shape)
            ic (hypothesis_spectra.shape)
            if best_scores is not None:
                ic (best_scores.shape)
            
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_spectra"] = hypothesis_spectra
            inputs["hypothesis_formula"] = hypothesis_formula
            # step counters
            num_recursive_steps -= 1
            num_recursive_steps_so_far += 1

        return (
            hypothesis_ids,
            hypothesis_spectras,
            hypothesis_embeddings,
            hypothesis_formulas,
            beam_scores,
            initial_score,
        )

    def _generate_with_beam(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int,
        num_recursive_steps_so_far: int,
        sequence_beam_width: int,
        alpha: float = None,
        _eval: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates text using self-correction.
        
        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
        generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            num_recursive_steps (int): Number of remaining steps of recursion, used to know when to stop
            num_recusive_steps_so_far (int): Number of steps of recursion performed so far. This is how we
                can check if it's the initial hypothesis or not.
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (torch.Tensor): ids of generated text
        """
        sim_fn_spectra = self.get_sim_fn()
        sim_fn_formula = self.get_sim_fn('diff')
        if alpha is None and self.with_formula:
            alpha = self.model.config.alpha_spec_form
        
        assert num_recursive_steps >= 1
        spectre = inputs["spectre"]
        
        if not generation_kwargs["do_sample"]:
            num_return_sequences = max(
                sequence_beam_width, generation_kwargs.get("num_beams", 1)
            )
            generation_kwargs["num_beams"] = num_return_sequences
            generation_kwargs["num_return_sequences"] = num_return_sequences
            
        if (num_recursive_steps_so_far == 0) and (
                self.initial_hypothesis_str is not None
        ):
            '''
            ###################################################
            
            Not using the hypothesis provided by the base model at step 0 (might be broken)
            
            ###################################################
            '''
            # Support setting a string as the initial hypothesis (for ablations)
            logger.info(f"Using initial hypothesis: {self.initial_hypothesis_str}")
            # If set, uses this string as the hypothesis for step 0 of self-correction
            batch_size = spectre.shape[0]
            gen_text_ids = (
                self.embedder_tokenizer(
                    [self.initial_hypothesis_str],
                    return_tensors="pt",
                    max_length=inputs["hypothesis_input_ids"].shape[1],
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                .repeat((batch_size, 1))
                .to(self.args.device)
            )
            # gen_text_ids = (
            #     torch.randint(
            #         low=1,
            #         high=self.embedder_tokenizer.vocab_size,
            #         size=(1, inputs["hypothesis_input_ids"].shape[1]),
            #         dtype=torch.long,
            #     )
            #     .repeat((batch_size, 1))
            #     .to(self.args.device)
            # )
            bos_token_id = self.model.encoder_decoder.config.decoder_start_token_id
            bos_token_ids = (
                torch.ones(
                    (batch_size, 1), dtype=torch.long, device=gen_text_ids.device
                )
                * bos_token_id
            )
            gen_text_ids = torch.cat((bos_token_ids, gen_text_ids[:, :-1]), dim=1)
        
        else:
            '''
            ###################################################
            
            Using the base model (default behaviour)
            
            ###################################################
            '''
            ic (inputs)
            outputs = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                return_dict_in_generate=True,
            )
            gen_text_ids = outputs.sequences
            # get scores for sequences to compute sequence-level likelihood.
            # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
            if "beam_indices" in outputs:
                with torch.no_grad():
                    transition_scores = (
                        self.model.encoder_decoder.compute_transition_scores(
                            outputs.sequences,
                            outputs.scores,
                            outputs.beam_indices,
                            normalize_logits=True,
                        )
                    )
            else:
                with torch.no_grad():
                    transition_scores = (
                        self.model.encoder_decoder.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )
                    )
            length_penalty = self.model.encoder_decoder.generation_config.length_penalty
            output_length = (transition_scores < 0).sum(1)
            del outputs.scores
            gen_text_scores = transition_scores.sum(axis=1) / (
                output_length**length_penalty
            )  # log probs

        '''
        ###################################################
        
        Re-embedding the inputs to perform the beam logic
        
        ###################################################
        '''
        (
            hypothesis_spectra,
            hypothesis_embedding,
            hypothesis_formula
        ) = self.embed_generated_hypothesis(input_ids=gen_text_ids, return_embedding=True)

        if num_recursive_steps_so_far == 0:
            batch_size = spectre.shape[0]
        else:
            # after the first step, we've already copied frozen embeddings across the beam
            batch_size = int(spectre.shape[0] / sequence_beam_width)

        best_scores = None
        best_scores_idx = None
        
        if gen_text_ids.shape[0] > batch_size:
            '''
            ###################################################
            
            Beam logic
            
            ###################################################
            '''
            
            if sequence_beam_width == 1:
                '''
                ###################################################
                
                Regular beam search
                
                ###################################################
                '''
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                distances_per_beam = sim_fn(
                    hypothesis_spectra.reshape((batch_size, beam_width, -1)),
                    inputs["spectre"][:, None, :],
                )

                if hypothesis_formula is not None:
                    
                    distances_per_beam_formula = sim_fn_formula(
                        hypothesis_formula.reshape((batch_size, beam_width, -1)),
                        inputs["formula"][:, None, :]
                    )
                    
                    distances_per_beam = (
                        alpha * distances_per_beam + (1-alpha) * distances_per_beam_formula
                    )
                
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))
                    
                best_idx_in_beam = scores.argmax(1)
                
                hypothesis_spectra = hypothesis_spectra.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                
                gen_text_ids = gen_text_ids.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]

                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]

                if hypothesis_formula is not None:
                    hypothesis_formula = hypothesis_formula.reshape(
                        (batch_size, beam_width, -1)
                    )[torch.arange(batch_size), best_idx_in_beam]

            elif num_recursive_steps == 1 and not _eval:
                '''
                ###################################################
                
                Last step for seq-level beam search
                --> We keep only the best hypothesis accross seq beam
                
                ###################################################
                '''
                # Base case for sequence-level beam search.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                spectre_per_beam = (
                    inputs["spectre"][:, None, :]
                    .repeat((1, num_return_sequences, 1))
                    .reshape((batch_size, beam_width, -1))
                )

                distances_per_beam = sim_fn_spectra(
                    hypothesis_spectra.reshape((batch_size, beam_width, -1)),
                    spectre_per_beam,
                )

                if hypothesis_formula is not None:
                    formula_per_beam = (
                        inputs["formula"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape((batch_size, beam_width, -1))
                    )
                    distances_per_beam_formula = sim_fn_formula(
                        hypothesis_formula.reshape((batch_size, beam_width, -1)),
                        formula_per_beam,
                    )
                
                    distances_per_beam = (
                        alpha * distances_per_beam + (1-alpha) * distances_per_beam_formula
                    )
                    
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))
                    
                best_idx_in_beam = scores.argmax(dim=1)
                
                hypothesis_spectra = hypothesis_spectra.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam
                ]
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                
                if hypothesis_formula is not None:
                    hypothesis_formula = hypothesis_formula.reshape(
                        (batch_size, beam_width, -1)
                    )[torch.arange(batch_size), best_idx_in_beam]

            else:
                '''
                ###################################################
                
                Base case for seq-level beam search
                
                ###################################################
                '''
                # Now get top things in the beam like normal.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                # Yes indeed, but it's always the case since beam_width == seq_beam_width * num_beam ??
                """
                assert (
                    beam_width % sequence_beam_width == 0
                ), "inner beam width must divide sequence beam width"
                """
                
                if num_recursive_steps_so_far == 0:
                    # This is the first return for sequence-level beam search.
                    # First we have to copy the frozen embedding
                    spectre_per_beam = (
                        inputs["spectre"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape((batch_size, num_return_sequences, -1))
                    )
                    inputs["spectre"] = (
                        inputs["spectre"][:, None, :]
                        .repeat((1, sequence_beam_width, 1))
                        .reshape((batch_size * sequence_beam_width, -1))
                    )
                    if hypothesis_formula is not None:
                        formula_per_beam = (
                            inputs["formula"][:, None, :]
                            .repeat((1, num_return_sequences, 1))
                            .reshape((batch_size, num_return_sequences, -1))
                        )
                        inputs["formula"] = (
                            inputs["formula"][:, None, :]
                            .repeat((1, sequence_beam_width, 1))
                            .reshape((batch_size * sequence_beam_width, -1))
                        )
                else:
                    spectre_per_beam = (
                        inputs["spectre"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape(
                            (batch_size, sequence_beam_width * num_return_sequences, -1)
                        )
                    )

                    if hypothesis_formula is not None:
                        formula_per_beam = (
                            inputs["formula"][:, None, :]
                            .repeat((1, num_return_sequences, 1))
                            .reshape(
                                (batch_size, sequence_beam_width * num_return_sequences, -1)
                            )
                        )

                distances_per_beam = sim_fn_spectra(
                    hypothesis_spectra.reshape((batch_size, beam_width, -1)),
                    spectre_per_beam,
                )

                if hypothesis_formula is not None:
                    
                    distances_per_beam_formula = sim_fn_formula(
                        hypothesis_formula.reshape((batch_size, beam_width, -1)),
                        formula_per_beam,
                    )
                    
                    distances_per_beam = (
                        alpha * distances_per_beam + (1-alpha) * distances_per_beam_formula
                    )
                
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))

                best_idx_in_beam_total = scores.topk(dim=1, k=beam_width).indices
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )
                hypothesis_spectra = hypothesis_spectra.reshape(
                    (batch_size, beam_width, -1)
                )
                gen_text_ids = gen_text_ids.reshape(
                    (batch_size, beam_width, -1)
                )
                best_idx_in_beam = []
                for batch_idx in range(len(best_idx_in_beam_total)):
                    gen_text_set = set()  # track uniqueness
                    best_idx_in_beam.append([])
                    for j in best_idx_in_beam_total[batch_idx].tolist():
                        gen_text_i = tuple(gen_text_ids[batch_idx, j].tolist())
                        if gen_text_i not in gen_text_set:
                            gen_text_set.add(gen_text_i)
                            best_idx_in_beam[batch_idx].append(j)
                        if len(best_idx_in_beam[batch_idx]) == sequence_beam_width:
                            break
                best_idx_in_beam = torch.tensor(
                    best_idx_in_beam, device=best_idx_in_beam_total.device
                )
                # now take top unique things
                hypothesis_spectra = hypothesis_spectra.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size)[:, None], best_idx_in_beam]
                                
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size)[:, None], best_idx_in_beam
                ]
            
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size)[:, None], best_idx_in_beam]

                if hypothesis_formula is not None:
                    hypothesis_formula = hypothesis_formula.reshape(
                        (batch_size, beam_width, -1)
                    )[torch.arange(batch_size)[:, None], best_idx_in_beam]

            # print scores for any type of beam search
            best_scores = scores.topk(dim=1, k=sequence_beam_width).values
            best_scores_idx = scores.argmax(1)
        
        # make sure we reshape correctly
        # (can't do a shape check on gen_text_ids because of the dynamic length.)
        assert hypothesis_spectra.shape[-1] == inputs["spectre"].shape[-1]
        
        return gen_text_ids, hypothesis_spectra, hypothesis_embedding, hypothesis_formula, best_scores, best_scores_idx

    def get_spectre(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        return_embedding: bool,
        selfies: str,
    ):
        with torch.no_grad():
            spectre = self.embedder(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
                selfies=selfies,
                return_embedding = return_embedding
            )

        outputs = tuple(
            s.to(self.args.device) if s is not None else s
            for s in spectre
        )
        
        return outputs

    def get_sim_fn(self, input_fn_str: str = None) -> Optional[nn.Module]:
        fn_str = self.sim_fn_str
        if input_fn_str is not None:
            fn_str = input_fn_str
        if fn_str == "sis":
            sim_fn = SISMeasure()
        elif fn_str == "cos":
            sim_fn = torch.nn.CosineSimilarity(dim=2)
        elif fn_str == "diff":
            class RelativeL1Difference(nn.Module):
                def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    numerator = torch.sum(torch.abs(a - b), dim=-1)
                    denominator = torch.sum(torch.max(a, b), dim=-1)
                    return 1 - numerator / (denominator + 1e-8)  # avoid divide-by-zero
            sim_fn = RelativeL1Difference()
        else :
            print (f"This measure [{fn_str}] is not implemented")
            sim_fn = None

        return sim_fn

    def embed_generated_hypothesis(self, input_ids: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """Embeds a generated hypothesis. Has to remove EOS token and add BOS token
        at the beginning.
        """
        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        ic (inputs_str)
        emb_input_ids = self.embedder_tokenizer(
            inputs_str,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(input_ids.device)

        ic(emb_input_ids)
        return self.get_spectre(
            embedder_input_ids=emb_input_ids.input_ids,
            embedder_attention_mask=emb_input_ids.attention_mask,
            selfies = inputs_str,
            return_embedding = return_embedding,
        )

    def _get_hypothesis_uncached(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "spectre" in inputs:
            spectre = inputs.get("spectre")
            if "formula" in inputs:
                formula = inputs.get("formula")
        elif "embedder_input_ids" in inputs:
            spectre, _, formula = self.get_spectre(
                embedder_input_ids=inputs["embedder_input_ids"],
                embedder_attention_mask=inputs["embedder_attention_mask"],
            )
        else:
            assert (
                "input_ids" in inputs
            ), f"cannot generate hypothesis with input keys: {inputs.keys()}"
            spectre, _, formula = self.embed_generated_hypothesis(
                input_ids=inputs["input_ids"]
            )

        generation_kwargs = {
            "early_stopping": True,
            "num_beams": 10,
            "do_sample": False,
            "max_length": 514,
            "num_return_sequences": 1,
        }

        hypothesis_input_ids = self.inversion_trainer.model.generate(
            inputs={
                "spectre": spectre,
                "formula": formula,
            },
            generation_kwargs=generation_kwargs,
        )
        hypothesis_attention_mask = (
            hypothesis_input_ids != self.model.encoder_decoder.config.pad_token_id
        )

        ic (hypothesis_input_ids.shape)
        hypothesis_spectra, _, hypothesis_formula = self.embed_generated_hypothesis(
            input_ids=hypothesis_input_ids
        )

        return (
            spectre,
            formula,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_spectra,
            hypothesis_formula,
        )

    def compute_loss(
        self,
        model: CorrectorEncoderModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        batch_size, seq_length = inputs["input_ids"].shape
        try:
            spectre = inputs["spectre"]
            formula = inputs.get("formula")
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_spectra = inputs["hypothesis_spectra"]
            hypothesis_formula = inputs.get("hypothesis_formula")
            
        except KeyError:
            (
                spectre,
                formula,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_spectra,
                hypothesis_formula,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        labels = inputs["labels"]
        outputs = self.model(
            embedding=spectre,
            formula=formula,
            hypothesis_spectra=hypothesis_spectra,
            hypothesis_formula=hypothesis_formula,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
            labels=labels,
        )
        return outputs.loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`. Called during self.evalaute()
        """
        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        with torch.no_grad():
            loss = self.compute_loss(model=model, inputs=inputs)

        logits, labels = None, None
        return loss, logits, labels

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we stopped sharing params between the ff layers
        if {
            "embedding_transform.3.weight",
            "embedding_transform.3.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform_1.0.weight"] = state_dict.pop(
                "embedding_transform.0.weight"
            )
            state_dict["embedding_transform_1.0.bias"] = state_dict.pop(
                "embedding_transform.0.bias"
            )
            state_dict["embedding_transform_1.3.weight"] = state_dict.pop(
                "embedding_transform.3.weight"
            )
            state_dict["embedding_transform_1.3.bias"] = state_dict.pop(
                "embedding_transform.3.bias"
            )
            #
            state_dict["embedding_transform_2.0.weight"] = state_dict[
                "embedding_transform_1.0.weight"
            ]
            state_dict["embedding_transform_2.0.bias"] = state_dict[
                "embedding_transform_1.0.bias"
            ]
            state_dict["embedding_transform_2.3.weight"] = state_dict[
                "embedding_transform_1.3.weight"
            ]
            state_dict["embedding_transform_2.3.bias"] = state_dict[
                "embedding_transform_1.3.bias"
            ]
            #
            state_dict["embedding_transform_3.0.weight"] = state_dict[
                "embedding_transform_1.0.weight"
            ]
            state_dict["embedding_transform_3.0.bias"] = state_dict[
                "embedding_transform_1.0.bias"
            ]
            state_dict["embedding_transform_3.3.weight"] = state_dict[
                "embedding_transform_1.3.weight"
            ]
            state_dict["embedding_transform_3.3.bias"] = state_dict[
                "embedding_transform_1.3.bias"
            ]
        return state_dict
