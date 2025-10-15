import abc
import functools
import hashlib
import json
import logging
import os
import resource
import sys
from typing import Dict, Optional
from dataclasses import asdict

import datasets
from datasets import load_from_disk
import torch
import transformers
from transformers import logging as hf_logging
from transformers import EarlyStoppingCallback

import spec2selfies
from spec2selfies.collator import DataCollatorForCorrection, DataCollatorForInvertion
from spec2selfies.models import (
    CorrectorEncoderModel,
    InversionModel
)
from spec2selfies.trainers.invertion_trainer import InversionTrainer
from spec2selfies.trainers.corrector_trainer import CorrectorTrainer

from spec2selfies.models.config import InversionConfig
from spec2selfies.run_args import DataArguments, ModelArguments, TrainingArguments
from spec2selfies.aliases import load_experiment_and_trainer_from_alias

import wandb


from torch.serialization import add_safe_globals, get_unsafe_globals_in_checkpoint 
import numpy
import accelerate

# So that pytorch allow us to load our own checkpoints !!
add_safe_globals([
    numpy.ndarray,
    numpy.dtype,
    numpy.core.multiarray._reconstruct,
    accelerate.utils.dataclasses.DistributedType,
    transformers.trainer_utils.HubStrategy,
    transformers.training_args.OptimizerNames,
    transformers.trainer_pt_utils.AcceleratorConfig,
    accelerate.state.PartialState,
    transformers.trainer_utils.IntervalStrategy,
    spec2selfies.run_args.TrainingArguments,
    transformers.trainer_utils.SaveStrategy,
    transformers.trainer_utils.SchedulerType,
])

os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger = logging.getLogger(__name__)

# Noisy compilation from torch.compile
try:
    torch._logging.set_logs(dynamo=logging.INFO)
except AttributeError:
    # torch version too low
    pass


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


class Experiment(abc.ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        logger.info(
            "Save checkpoints according to metric_for_best_model %s:",
            training_args.metric_for_best_model,
        )

        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        # Set random seed, add hash to output path.
        transformers.set_seed(training_args.seed)

        if training_args.output_dir is None:
            training_args.output_dir = os.path.join("saves", self.kwargs_hash)
        print(f"Experiment output_dir = {training_args.output_dir}")
        # Set up output_dir and wandb.
        self._setup_logging()
        
    @property
    def config(self) -> InversionConfig:
        return InversionConfig(
            **asdict(self.data_args),
            **asdict(self.model_args),
            **asdict(self.training_args),
        )

    @property
    def dataset_kwargs(self) -> Dict[str, str]:
        return {
            "model_name": self.model_args.model_name_or_path,
            "embedder_name": self.model_args.embedder_model_name,
            "max_seq_length": str(self.model_args.max_seq_length),
            "use_less_data": str(self.data_args.use_less_data),
        }

    def _setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )
        hf_logging.set_verbosity_debug()

    def run(self):
        if self.training_args.do_eval:
            print ("we're here (in experiment.run())")
            self.evaluate()
        else:
            self.train()

    def train(self) -> Dict:
        # *** Training ***
        training_args = self.training_args
        logger.info("*** Training ***")

        # Log on each process a small summary of training.
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}"
        )

        logger.info(f"logging steps : {training_args.logging_steps}")
        logger.info(f"save steps : {training_args.save_steps}")
        logger.info(f"eval steps : {training_args.eval_steps}")
        logger.info(f"eval strategy : {training_args.eval_strategy}")
        logger.info(f"use lora : {self.model_args.use_lora}")
        #logger.info(f"Training/evaluation parameters {training_args}")
        # Checkpointing logic
        checkpoint = self._get_checkpoint()
        logging.info("Experiment::train() loaded checkpoint %s", checkpoint)
        trainer = self.load_trainer()

        # Save model_args and data_args before training. Trainer will save training_args.
        if training_args.local_rank <= 0:
            torch.save(
                self.data_args, os.path.join(training_args.output_dir, "data_args.bin")
            )
            torch.save(
                self.model_args,
                os.path.join(training_args.output_dir, "model_args.bin"),
            )

        # train.   :)
        print(f"train() called – resume-from_checkpoint = {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics

    def evaluate(self) -> Dict:
        # *** Evaluation ***
        logger.info("*** Evaluate ***")
        trainer = self.load_trainer()
        num_eval_samples = len(trainer.eval_dataset)
        metrics = trainer.evaluate()
        max_eval_samples = (
            self.data_args.max_eval_samples
            if self.data_args.max_eval_samples is not None
            else num_eval_samples
        )
        metrics["eval_samples"] = min(max_eval_samples, num_eval_samples)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

    def _get_checkpoint(self) -> Optional[str]:
        training_args = self.training_args
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
                training_args.output_dir
            )
            if (
                last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if checkpoint:
            logger.info("Loading from checkpoint %s", checkpoint)
        else:
            logger.info("No checkpoint found, training from scratch")

        return checkpoint

    # Creates a unique hash for each training
    @property
    def kwargs_hash(self) -> str:
        all_args = {
            **vars(self.model_args),
            **vars(self.data_args),
            **vars(self.training_args),
        }
        all_args.pop("local_rank")
        # print("all_args:", all_args)
        return md5_hash_kwargs(**all_args)

    @property
    def _world_size(self) -> int:
        try:
            return torch.distributed.get_world_size()
        except (RuntimeError, ValueError):
            return 1

    @property
    def _is_main_worker(self) -> bool:
        return (self.training_args.local_rank <= 0) and (
            int(os.environ.get("LOCAL_RANK", 0)) <= 0
        )

    
    @abc.abstractmethod
    def load_trainer(self) -> transformers.Trainer:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_model(self) -> transformers.PreTrainedModel:
        raise NotImplementedError()

    def load_tokenizer(self) -> transformers.PreTrainedTokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            padding="max_length",
            truncation="max_length",
            max_length=self.model_args.max_seq_length,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Disable super annoying warning:
        # https://github.com/huggingface/transformers/issues/22638
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer

    
    def get_collator(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> DataCollatorForInvertion:
        return DataCollatorForInvertion(
            tokenizer,
            model=None,
            return_tensors=None,
            label_pad_token_id=-100,
            padding="max_length",
            max_length=self.model_args.max_seq_length,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )
    

    def load_train_dataset(
            self,
            use_perfect_model: bool,
    ) -> datasets.Dataset:
        
        logger.info(f"Loading training dataset {self.data_args.train_dataset_path}")
        dataset = load_from_disk(self.data_args.train_dataset_path)
        spectra_to_drop = 'model_spectra'
        if use_perfect_model:
            print ("Using model spectra")
            spectra_to_drop = 'spectre'

        COLUMNS_TO_DROP = ['selfies', spectra_to_drop, "hypothesis_selfies"]
        COLUMNS_TO_DROP = [c for c in COLUMNS_TO_DROP if c in dataset.column_names]
        
        # this argument allows us to *train* on less data (for example 1% of our training set).
        if self.data_args.use_less_data and (self.data_args.use_less_data > 0):
            new_length = min(len(dataset), self.data_args.use_less_data)
            dataset = dataset.select(range(new_length))

        dataset = dataset.remove_columns(COLUMNS_TO_DROP)
        if use_perfect_model:
            dataset = dataset.rename_column('model_spectra', 'spectre')
        #dataset.rename_column('attention_mask', 'embedder_attention_mask')
        dataset.set_format("pt")

        return dataset

    def load_eval_dataset(
            self,
            use_perfect_model: bool,
    ) -> datasets.Dataset:
        
        logger.info(f"Loading eval dataset {self.data_args.eval_dataset_path}")
        dataset = load_from_disk(self.data_args.eval_dataset_path)
        spectra_to_drop = 'model_spectra'
        if use_perfect_model:
            print ("Using model spectra")
            spectra_to_drop = 'spectre'
            
        COLUMNS_TO_DROP = ['selfies', spectra_to_drop, "hypothesis_selfies"]
        COLUMNS_TO_DROP = [c for c in COLUMNS_TO_DROP if c in dataset.column_names]
        dataset = dataset.remove_columns(COLUMNS_TO_DROP)

        if use_perfect_model:
            dataset = dataset.rename_column('model_spectra', 'spectre')

        print (dataset)
        #dataset.rename_column('input_ids', 'embedder_input_ids')
        #dataset.rename_column('attention_mask', 'embedder_attention_mask')
        dataset.set_format("pt")

        return dataset

class InversionExperiment(Experiment):
    @property
    def trainer_cls(self):
        return InversionTrainer

    def load_model(self) -> transformers.PreTrainedModel:
        return InversionModel(
            config=self.config,
        )

    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset = self.load_train_dataset(self.config.use_perfect_model)
        eval_dataset = self.load_eval_dataset(self.config.use_perfect_model)
        
        if self.data_args.use_less_data and (self.data_args.use_less_data > 0):
            logger.info(f"Using less data ({self.data_args.use_less_data})")

            
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params"

        )

        if self.training_args.mock_embedder:
            # This mode allows us to get the embedders off the GPU during training
            # once we've computed all the embeddings we need. :)
            assert (
                model.config.use_frozen_embeddings_as_input
            ), "must use frozen embeddings if mock_embedder=True"
            print(
                "IMPORTANT: Mocking embedder for the rest of training (to save GPU memory)."
                " Do not trust embedding-based evaluation metrics."
            )
            model.embedder.cpu()
            del model.embedder
            model.embedder = MockEmbedder(embedder_dim=model.embedder_dim)

        return self.trainer_cls(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

class CorrectorExperiment(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-correct-1"

    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        if self.training_args.corrector_model_from_pretrained:
            (
                _,
                inversion_trainer,
            ) = spec2selfies.analyze_utils.load_experiment_and_trainer_from_pretrained(
                name=self.training_args.corrector_model_from_pretrained,
                # max_seq_length=self.model_args.max_seq_length,
                use_less_data=self.data_args.use_less_data,
            )
        elif self.model_args.use_precomputed_hypotheses:
            train_dataset = self.load_train_dataset(self.config.use_perfect_model)
            eval_dataset = self.load_eval_dataset(self.config.use_perfect_model)
            trainer = spec2selfies.trainers.corrector_trainer.CorrectorTrainer(
                model=model,
                inversion_trainer=None,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForCorrection(
                    tokenizer=model.tokenizer
                ),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            return trainer
        else:
            (
                _,
                inversion_trainer,
            ) = load_experiment_and_trainer_from_alias(
                alias=self.training_args.corrector_model_alias,
                store_embedder=self.model_args.store_embedder,
                max_seq_length=self.model_args.max_seq_length,
                use_less_data=self.data_args.use_less_data,
            )
            
        return spec2selfies.trainers.corrector_trainer.CorrectorTrainer(
            model=model,
            inversion_trainer=inversion_trainer,
            args=self.training_args,
            data_collator=DataCollatorForCorrection(
                tokenizer=inversion_trainer.model.tokenizer
            ),
        )

    def load_model(self) -> transformers.PreTrainedModel:
        return CorrectorEncoderModel(
            config=self.config,
        )




EXPERIMENT_CLS_MAP = {
    "inversion": InversionExperiment,
    "corrector": CorrectorExperiment,
}


def experiment_from_args(model_args, data_args, training_args) -> Experiment:
    if training_args.experiment in EXPERIMENT_CLS_MAP:
        experiment_cls = EXPERIMENT_CLS_MAP[training_args.experiment]  # type: ignore
    else:
        raise ValueError(f"Unknown experiment {training_args.experiment}")
    return experiment_cls(model_args, data_args, training_args)  # type: ignore
