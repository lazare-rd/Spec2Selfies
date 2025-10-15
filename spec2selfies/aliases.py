import spec2selfies

ARGS_DICT = {}
# Dictionary mapping model names
CHECKPOINT_FOLDERS_DICT = {
    "BART-base__SELFormer" : "./data/saved_models/Inverter__30epochs__encoder_decoder_frozen/checkpoint-15411",
}


def load_experiment_and_trainer_from_alias(
        alias: str, store_embedder: bool, max_seq_length: int = None, use_less_data: int = None
):  # -> trainers.InversionTrainer:
    try:
        args_str = ARGS_DICT.get(alias)
        checkpoint_folder = CHECKPOINT_FOLDERS_DICT[alias]
    except KeyError:
        print(f"{alias} not found in aliases.py, using as checkpoint folder")
        args_str = None
        checkpoint_folder = alias
    print(f"loading alias {alias} from {checkpoint_folder}...")
    experiment, trainer = spec2selfies.utils.load_experiment_and_trainer(
        checkpoint_folder,
        args_str,
        do_eval=False,
        max_seq_length=max_seq_length,
        use_less_data=use_less_data,
        store_embedder=store_embedder,
    )
    return experiment, trainer


def load_model_from_alias(alias: str, max_seq_length: int = None):
    _, trainer = load_experiment_and_trainer_from_alias(
        alias, max_seq_length=max_seq_length
    )
    return trainer.model
