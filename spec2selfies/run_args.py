import os
from dataclasses import dataclass, field
from typing import Optional
from typing import List

import torch
import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import IntervalStrategy

from spec2selfies.models.model_utils import (
    EMBEDDER_MODEL_NAMES,
    FREEZE_STRATEGIES,
    EMBEDDING_TRANSFORM_STRATEGIES
)

DATASET_NAMES = [
    "merged_scaffold_split"
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        ###
        ## huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="./data/saved_models/models--zjunlp--MolGen-large/snapshots/3c6e8fb91a783853a3782552a85efc4df6f96d0a",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    embedder_model_name: str = field(
        ###
        ## huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="SELFormer",
        metadata={
            "help": "Model to get embeddings from (locally)",
            "choices": EMBEDDER_MODEL_NAMES,
        },
    )
    embedder_gaussian_noise_level: float = field(
        default=0.0, metadata={"help": "noise level to add during training to embedder"}
    )
    embedder_torch_dtype: str = field(
        default="float32",
        metadata={
            "help": "torch dtype of embedder",
            "choices": ["float32", "float16", "bfloat16"],
        },
    )
    embedding_transform_strategy: str = field(
        default="repeat",
        metadata={
            "help": "Strategy for transforming from sentence embedding into sequence-level input for encoder-decoder",
            "choices": EMBEDDING_TRANSFORM_STRATEGIES,
        },
    )
    encoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 encoder"}
    )
    decoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 decoder"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    max_seq_length: int = field(
        default=1024, metadata={"help": "Maximum sequence length for tokenizer"}
    )
    torch_dtype: str = field(
        default='float16',
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    num_repeat_tokens: int = field(
        default=16,
        metadata={
            "help": "Number of times to repeat embedding along T5 input sequence length."
        },
    )

    embedder_dim: int = field(
        default=1801,
        metadata={
            "help": "Dim of the embedding"
        },
    )

    bottleneck_dim: int = field(
        default=900,
        metadata={
            "help": "Bottleneck dim for embedding transform MLP"
        },
    )

    ATOM_LIST: List[str] = field(
        default_factory=lambda: ['I', 'C', 'S', 'N', 'Br', 'H', 'O', 'Si', 'Cl', 'P', 'F'],
        metadata={
            "help": "List of atoms present in the dataset"
        }
    )

    formula_dim: int = field(
        init=False,
        metadata={
            "help": "Formula dim for embedding transform MLP, (equals to length of ATOM_LIST)"
        },
    )
    
    with_formula: bool = field(
        default=True,
        metadata={"help": "Whether to use mol formula as input along with the spectra"}
    )

    with_sep: bool = field(
        default=True,
        metadata={"help": "Wether to put sep_tokens between 'pieces' of inputs for corrector"}
    )

    use_perfect_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Wether to use a model computed spectra as input VS a true spectra."
                 "Using a computed spectra allows us to simulate a model that predicts"
                 "the spectras perfectly"
            )
        }
    )

    alpha_spec_form: float = field(
        default=0,
        metadata={
            "help": (
                "When using mol formula, ratio between formula error and spec error in seq beam search"
                "(1 --> only spec | 0 --> only formula)"
            )
        }
    )

    use_ff_dropout: bool = field(
        default=True,
        metadata={"help": "Wether to use dropout in the ff layers of BART"}
    )

    dropout: float = field(
        default=0.3,
        metadata={"help": "Dropout value"}
    )
    embedder_no_grad: bool = field(
        default=True, metadata={"help": "Whether to disable grads embedder"}
    )
    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LORA+int8 for fine-tuning"}
    )
    embedder_fake_with_zeros: bool = field(
        default=False,
        metadata={
            "help": "Whether to pass all zeros as embedding (and not use DPR at all)"
        },
    )
    use_frozen_embeddings_as_input: bool = field(
        default=True,
        metadata={
            "help": "Whether to pass a 'frozen_embedding' column and train on that instead of generating embeddings on-the-fly"
        },
    )

    use_precomputed_hypotheses: bool = field(
        default=True,
        metadata={
            "help": "Wether to use a dataset with precomputed hypotheses or compute them before training"
        },
    )

    store_embedder: bool = field(
        default=False,
        metadata={
            "help": "Wether to load the embedder when instanciating an invertion model. Will be overidden to True if use_frozen_embeddings_as_input is False"
        },
    )
    
    corrector_ignore_hypothesis_embedding: bool = field(
        default=False,
        metadata={
            "help": "If set, and training corrector encoder, will ignore the hypothesis embedding"
        },
    )
    embeddings_from_layer_n: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, uses embeddings from layer n - for example set to 0 to use word embeddings"
        },
    )
    freeze_strategy: str = field(
        default="none",
        metadata={
            "help": "which part of the model to freeze",
            "choices": FREEZE_STRATEGIES,
        },
    )

    def __post_init__(self):
        self.formula_dim = len(self.ATOM_LIST)
        
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass

class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./data/scaffold_split_merged_spectra/arrow/hypothesis_dataset",
        metadata={
            "help": "The path of the dataset",
        },
    )

    max_eval_samples: int = field(
        default=500,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    use_less_data: int = field(
        default=-1,
        metadata={
            "help": {"Use a small amount of the training/eval data (for testing)"}
        },
    )

    def __post_init__(self):
       if self.dataset_name is not None:
           self.train_dataset_path = self.dataset_name + "/train"
           self.eval_dataset_path = self.dataset_name + "/eval"
           self.test_dataset_path = self.dataset_name + "/test"
       else:
           raise ValueError("You must provide a --dataset_name")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for training saves. If not set, will output to data/saved_models/<random hash>."
        },
    )
    corrector_model_alias: Optional[str] = field(
        default="BART-base__SELFormer",
        metadata={"help": "Alias of corrector model to train (defined in aliases.py)"},
    )
    corrector_model_from_pretrained: Optional[str] = field(
        default=None,
        metadata={
            "help": "Alias of pre-trained corrector model to train (defined in aliases.py)"
        },
    )
    cheat_on_train_hypotheses: bool = field(
        default=False,
        metadata={
            "help": "When set, will interpolate true with pred train hypothesis for 'closer' training data"
        },
    )

    steps_per_epoch: int = field(
        default=500_000,
        metadata={"required": False, "help": "Size of pseudo-training set."},
    )
    num_train_epochs: float = field(
        default=30.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."},
    )
    use_wandb: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )

    fp16: bool = field(
        default=False,
        metadata={"help": ("Whether to use fp16 (mixed) precision instead of 32-bit.")},
    )
    # torch_compile: bool = True # for torch 2

    ##################### Experimental Settings ####################
    experiment: str = field(
        default="inversion",
        metadata={
            "required": False,
            "help": "Which experiment to run (defines model, loss func, dataset...) ",
            "choices": [
                "inversion",
                "corrector",
            ],
        },
    )
    exp_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this specific run of an experiment",
        },
    )
    exp_group_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this sweep / series of experiments",
        },
    )

    # Need to *not* remove unused columns so we keep query_attention_mask, etc.
    # which huggingface doesn't think we need.
    remove_unused_columns: bool = False

    # Do evaluation and logging on certain num steps.
    evaluation_strategy: str = field (
        default = "steps",
        metadata={"help": "Strategy for evaluation during training"}
    )
    
    logging_strategy: str = "steps"
    save_strategy: str = "steps"
    do_train: bool = True
    
    do_eval : bool = field(
        default = False,
        metadata={"help": "Perform the eval steps"}
    )

    save_total_limit: int = 2  # Maximum number of checkpoints to save.

    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = field(
        default=700,
        metadata={"help": "Number of steps per save"},
    )
    eval_steps: int = field(
        default=700,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )

    metric_for_best_model: str = field(
        default='eval_loss',
        metadata={
            "help": "Metric to look at to log best model"
        },
    )

    debug_model : bool = field(
        default=False,
        metadata={
            "help": "If you run in debug mode, eval_step and logging_step are set very low for testing purposes"
        },
    )
    mock_embedder: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, will delete the embedder and replace all embedder logits with"
                " zeros once training starts. You probably don't want to do this. But "
                " if you precomputed all the embeddings for train and val, this will"
                " work fine, except the embedding-based metrics (just cosine similarity"
                " I think) will be broken."
            )
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    include_inputs_for_metrics: bool = True

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        self._frozen = True
        self.report_to = (
            ["wandb"] if (self.use_wandb and (self.local_rank <= 0)) else []
        )
        self.dataloader_pin_memory = True
        num_workers = torch.cuda.device_count()
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")

        self.dataloader_drop_last = False

        # Scale logging steps proportional to batch size.
        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        if self.debug_model:
            self.logging_steps = 40
            self.eval_steps = 40
            self.save_steps = 10_000
        
        # defaults from SentenceTransformers
        # lr 2e-5
        self.adam_epsilon = 1e-6

        self.group_by_length = True
        self.length_column_name = "length"

        self.load_best_model_at_end = True
        self.greater_is_better = False
        
        # Avoid conflicts with evaluation_strategy
        if self.evaluation_strategy == "steps":
            self.eval_strategy = IntervalStrategy.STEPS
        elif self.evaluation_strategy == "epoch":
            self.eval_strategy = IntervalStrategy.EPOCH
        else :
            self.eval_strategy = IntervalStrategy.NO
        # self.ddp_backend = "gloo"

"""
-- Config to check if the training computes properlys --
"""
def create_mock_args(
        experiment: str
) -> tuple :
    model_args = ModelArguments(
        model_name_or_path=path_jeanzay_model("t5-small"),
        embedder_model_name="small_mock",
    )

    data_args = DataArguments(
        dataset_name="squad",
        max_eval_samples=1,
        use_less_data=1,
    )

    if experiment == "inversion_decoder_only":
        training_args = TrainingArguments(
            output_dir="./mock_output_inverter",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=1,
            eval_steps=1,
            exp_name="mock_test",
            experiment=experiment,
            use_wandb=False,
            report_to="none",
        )
    else :
        training_args = TrainingArguments(
            output_dir="./mock_output_corrector",
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=1,
            eval_steps=1,
            exp_name="mock_test",
            experiment=experiment,
            use_wandb=False,
            report_to="none",
            corrector_model_alias="t5-base___smallEmbedder_for_mock_1epoch"
        )
    

    return model_args, data_args, training_args

"""
-- Config to check at low cost if the model is learning properly --

High learning rate
Low size dataset
Low nbre of epoch 
"""

def create_sound_check_args (
        experiment: str
) -> tuple :

    model_args = ModelArguments(
        model_name_or_path=path_jeanzay_model("t5-small"),
        embedder_model_name="small_mock",
    )

    data_args = DataArguments(
        dataset_name="squad",
        max_eval_samples=20,
        use_less_data=100,
    )

    if experiment == "inversion_decoder_only":
        training_args = TrainingArguments(
            output_dir="./soundcheck_output_inverter",
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            num_train_epochs=30,
            learning_rate=1e-3,
            logging_steps=5,
            save_steps=30,
            eval_steps=10,
            exp_name="sound_check",
            experiment=experiment,
            use_wandb=False,
            report_to="none",
        )
    else :
        training_args = TrainingArguments(
            output_dir="./soundcheck_output_corrector",
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            num_train_epochs=30,
            learning_rate=1e-3,
            logging_steps=5,
            save_steps=30,
            eval_steps=10,
            exp_name="sound_check",
            experiment=experiment,
            use_wandb=False,
            report_to="none",
            corrector_model_alias="t5-base___smallEmbedder_for_mock_30epoch"
        )
        
    return model_args, data_args, training_args
    
