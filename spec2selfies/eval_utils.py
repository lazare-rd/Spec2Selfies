from spec2selfies.api import load_pretrained_corrector, invert_embeddings_eval
import argparse
from box import Box

import wandb
from torch.utils.data import DataLoader
import torch
from transformers import DataCollatorForSeq2Seq
from datasets import load_from_disk

import yaml
import random
import numpy as np
import gc
import json
import os
import petname as ptn
import time

os.environ["WANDB_MODE"] = "offline"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BAYES_SEARCH_CACHE = "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval"

"""
Logging big stuffs  
"""

class BigStuffLogger:
    def __init__(
            self,
            config: Box,
            petname: str, 
    ):

        self.buffers = {}
        self.size_buffer = config.size_buffer
        self.petname = petname

        for name, buff in config.buffers.items():
            shape = [config.size_buffer, config.steps, config.gen_kwargs.seq_beam_width, buff.signal]
            dtype = getattr(torch, buff.dtype)
            shape_mask = buff.shape_mask
            
            shape = tuple([dim for i, dim in enumerate(shape) if shape_mask[i] != 0])
            print (shape)
            self.buffers[name] = {
                "tensor": torch.empty(shape, device="cpu", dtype=dtype),
                "counter": 0,
                "shape": shape,
                "dtype": dtype,
            }
                

    def log(self, name, data):
        buffer = self.buffers[name]
        batch_size = data.shape[0]

        if buffer["counter"] + batch_size > self.size_buffer:
            buffer["tensor"] = self.purge_on_disk(buffer["tensor"], name)
            buffer["counter"] = 0

        buffer["tensor"][buffer["counter"] : buffer["counter"] + batch_size] = data
        buffer["counter"] += batch_size


    def purge_on_disk(self, tensor, name, offset=None):
        
        path =f"{BAYES_SEARCH_CACHE}/{self.petname}/{name}.bin"
        path_metadata = f"{BAYES_SEARCH_CACHE}/{self.petname}/{name}.json"

        os.makedirs(f"{BAYES_SEARCH_CACHE}/{self.petname}", exist_ok=True)
        
        try:
            with open (path_metadata, "x") as f:
                json.dump(
                    {
                        "shape" : tensor.shape[1:],
                        "dtype" : str(tensor.dtype)
                    },
                    f,
                    indent=2,
                )
        except FileExistsError:
            # We write metadata only at the first purge
            pass

        if offset is not None:
            tensor = tensor[:offset]
            
        with open(path, "ab") as f:
            arr = tensor.numpy()
            f.write(arr.tobytes())
        
        del tensor
        gc.collect()

        print (f"### Purged {name} buffer ###")
        return torch.empty(self.buffers[name]["shape"], device="cpu", dtype=self.buffers[name]["dtype"])

    def log_outputs(
            self,
            outputs: dict[str, torch.Tensor],
    ):
        for name, value in self.buffers.items(): 
            tensor = outputs.get(name).to(value["dtype"])
            assert tensor.device.type == "cpu"
            self.log(name, tensor)

    def purge_all_buffers(self):
        for name, value in self.buffers.items():
            self.purge_on_disk(value["tensor"], name, value['counter'])
    
"""
Loading stuff
"""

def load_dataloader(
        collator,
        path: str,
        batch_size: int,
        limit: int,
        use_perfect_model: bool,
):
    dataset = load_from_disk(path)
    spectra_to_keep = 'spectre'
    if use_perfect_model:
        print ("Loading model_spectra")
        spectra_to_keep = 'model_spectra' 
    
    columns_to_keep = [spectra_to_keep, 'formula']
    
    cols_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(cols_to_remove)
    
    if limit is not None:
        dataset = dataset.select(range(limit))

    if use_perfect_model:
            dataset = dataset.rename_column('model_spectra', 'spectre')
    
    dataset.set_format(type="torch")
    return DataLoader(dataset, batch_size=batch_size, collate_fn=None)

def load_config(path):
    with open(path, "r") as f:
        return Box(yaml.safe_load(f))

def load_logs(path):
    meta_path = f"{path}.json"
    bin_path = f"{path}.bin"

   
    with open(meta_path, "r") as f:
        meta = json.load(f)

    subshape = tuple(meta["shape"])        
    dtype = np.dtype(meta["dtype"][6:])
    
    
    data = np.fromfile(bin_path, dtype=dtype)
    
    total_elems = data.size
    elem_per_tensor = np.prod(subshape)
    size = total_elems // elem_per_tensor
    
    assert size * elem_per_tensor == total_elems

    tensor = data.reshape(size, *subshape)
    
    return tensor

"""
Evaluation logic
"""

def eval_step(corrector, config, batch):
    if config.gen_kwargs.task == "beam":
        gen_kwargs = {
            "do_sample": False,
            "num_beams": config.gen_kwargs.num_beams,
            "early_stopping": True, 
        } 
        outputs = invert_embeddings_eval(
            batch,
            corrector=corrector,
            gen_kwargs=gen_kwargs,
            num_steps=config.steps,
            sequence_beam_width=config.gen_kwargs.seq_beam_width,
            alpha=getattr(config, "alpha", None),
        )
    elif config.gen_kwargs.task == "sample":
        gen_kwargs = {                       
            "do_sample": True,
            "temperature": config.temp,             
            "top_k": config.gen_kwargs.top_k,                    
            "top_p": config.gen_kwargs.top_p, 
        }
        
        outputs = invert_embeddings_eval(
            batch,
            corrector=corrector,
            gen_kwargs=gen_kwargs,
            num_steps=config.steps,
            alpha=getattr(config, "alpha", None),
        )
    else :
        raise ValueError(
            f"This task is not implemented : config.gen_kwargs.task = {config.gen_kwargs.task}"
        )
    
    return outputs


def evaluate(corrector, dataloader, config):
    petname = ptn.Generate(words=3, separator="-")
    logger = BigStuffLogger(config, petname)
    sis = []
    threshold = 0.9 * torch.cuda.get_device_properties(0).total_memory

    print ("### Evaluation begins ###")
    print (f"Batch size : {config.batch_size}")
    print (f"Num_beams : {config.gen_kwargs.num_beams}")
    print (f"Num_seq_beams : {config.gen_kwargs.seq_beam_width}")
    print (f"Log period : {config.log_period}")
    print (f"Model name : {config.corrector}")
    print (f"Alpha : {getattr(config, 'alpha', None)}")
    print (f"Use perfect model : {config.use_perfect_model}")
    wandb.log(
        {
            "outputs_saves": f"{BAYES_SEARCH_CACHE}/{petname}",
            "num_seq_beams" : config.gen_kwargs.seq_beam_width,
            "num_beams" : config.gen_kwargs.num_beams,
            "num_steps" : config.steps,
            "model_name" : config.corrector,
            "alpha": getattr(config, "alpha", None),
            "use_perfect_model": config.use_perfect_model,
        }
    )
    
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        counter = 0
        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = eval_step(corrector, config, batch)

            for k, v in outputs.items():
                if torch.is_tensor(v):
                    assert v.device.type == "cpu"
            
            # Clearing memory
            del batch
            if torch.cuda.memory_reserved() > threshold:
                print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
                print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
                torch.cuda.empty_cache()
                print ("### Emptying CUDA cache ###")
            logger.log_outputs(outputs)
            
            batch_sis = outputs["scores"].max(dim=2).values.max(dim=1).values.tolist()
            log_dico = {}
            log_dico["eval_sis_step_0"] = np.mean(
                    outputs["initial_score"].tolist()
                )
            for i in range(outputs["scores"].shape[1]):
                log_dico[f"eval_sis_step_{i+1}"] = np.mean(
                    outputs["scores"][:, i,:].max(dim=1).values.tolist()
                )
                
            wandb.log(
                log_dico
            )
            
            sis += batch_sis
            counter += 1
            if counter % config.log_period == 0:
                print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
                print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
                print (f"{ (counter/ int(len(dataloader.dataset) / config.batch_size)) * 100 }%")

    logger.purge_all_buffers()
    sis_mean = np.mean(sis)
    sis_std = np.std(sis)

    wandb.log(
        {
            "mean_eval_sis": sis_mean,
            "std_eval_sis": sis_std,
        }
    )
        
    return sis_mean, sis_std

def main(config):
    wandb.init(project="spec2selfies_eval") 
    
    corrector = load_pretrained_corrector(config.corrector)
    collator = DataCollatorForSeq2Seq(tokenizer=corrector.model.tokenizer, model=corrector.model)
    dataloader = load_dataloader(
        collator,
        config.dataset,
        config.batch_size,
        config.use_less_data,
        config.use_perfect_model,
    )

    sis_mean, sis_std = evaluate(corrector, dataloader, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_less_data", type=int, default=None, help="Eval on a small subset to debug")
    parser.add_argument("--config_yaml", type=str, required=True, help="Yaml config to load from")
    
    args = parser.parse_args()
    config = load_config(args.config_yaml)
    config.use_less_data = args.use_less_data
    
    main(config)
    
