import os
os.environ["TMPDIR"] = "/lustre/fsn1/projects/rech/aik/unh74gq/cache/tmp"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer
from transformers import RobertaConfig, RobertaTokenizerFast

from spec2selfies.utils import dataset_map_multi_worker
from spec2selfies.tokenize_data import tokenize_function

import spec2selfies.models.invertion_model as im
import spec2selfies.models.MolToSpec as mts
import spec2selfies.models.model_utils as mu

from spec2selfies.collator import DataCollatorForInvertion
from spec2selfies.collator import DataCollatorForCorrection

import pandas as pd
from datasets import Dataset
from datasets import load_from_disk
from tqdm import tqdm
import gc
import torch

import logging
import argparse

from collections import Counter
from rdkit import Chem
import selfies as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)
logger = logging.getLogger(__name__)

train_dataset_path = "./data/scaffold_split_merged_spectra/train_merged_spectra_selfies.csv"
val_dataset_path = "./data/scaffold_split_merged_spectra/val_merged_spectra_selfies.csv"
test_dataset_path = "./data/scaffold_split_merged_spectra/test_merged_spectra_selfies.csv"

arrow_dataset_path = "./data/scaffold_split_merged_spectra/arrow"
tokenized_arrow_dataset_path = "./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/"
tokenized_arrow_formula_dataset_path = "./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/with_formula"
ATOM_LIST = ['I', 'C', 'S', 'N', 'Br', 'H', 'O', 'Si', 'Cl', 'P', 'F']

def csv_to_arrow_dataset():
    
    train_df = pd.read_csv(train_dataset_path, sep=',')
    #val_df = pd.read_csv(val_dataset_path, sep=',')
    #test_df = pd.read_csv(test_dataset_path, sep=',')
    dataset = Dataset.from_pandas(train_df)
    del train_df
    gc.collect()
    
    def _lambda(dataset_item, idx):
        spectre = []
        if idx % 10_000 == 0:
            logger.info(f"Processed example index: {idx}")
        for i in range(400, 4002, 2):
            spectre.append(dataset_item[str(i)])
        dataset_item['spectre'] = spectre
        return dataset_item
    
    torch.distributed.init_process_group(backend="gloo")
    dataset = dataset_map_multi_worker(dataset, _lambda)
    
    for i in tqdm(range(400, 4002, 2)):
        dataset = dataset.remove_columns(str(i))
            
    dataset.save_to_disk("./data/scaffold_split_merged_spectra/arrow/train")
    
def add_chemical_formula_dataset(dataset_path, split):
    dataset = load_from_disk(dataset_path)
    
    def _lambda(dataset_item, idx):
        smiles = sf.decoder(dataset_item['hypothesis_selfies'])
        mol = Chem.MolFromSmiles(smiles)
        atom_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        dataset_item['hypothesis_formula'] = [atom_counts.get(atom, 0) for atom in ATOM_LIST]
        
        return dataset_item
    
    #torch.distributed.init_process_group(backend="gloo")
    dataset = dataset_map_multi_worker(dataset, _lambda, batched=False, num_proc=None)
            
    dataset.save_to_disk(dataset_path)

    

def tokenize_dataset(
    args : argparse.Namespace
):
    path = f"{arrow_dataset_path}/{args.dataset}"
    if args.use_less_data:
        path = f"{arrow_dataset_path}/eval"
    
    dt = load_from_disk(path)
    if args.use_less_data:
        dt = dt.select([i for i in range(100)])
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    embedder_tokenizer = RobertaTokenizerFast.from_pretrained(args.embedder_tokenizer_name_or_path)
    tokenize_fn = tokenize_function(
        tokenizer = tokenizer,
        text_column_name = args.text_column_name,
        max_seq_length = args.max_seq_length,
        embedder_tokenizer = embedder_tokenizer,
    )
    
    torch.distributed.init_process_group(backend="gloo")
    dt = dataset_map_multi_worker(dt, tokenize_fn)
    
    save_path = f"{arrow_dataset_path}/tokenized_datasets/with_formula/{args.dataset}"
    if args.use_less_data:
        save_path = f"{arrow_dataset_path}/tokenized_datasets/with_formula/debug"
    
    print (f"Dataset will be saved to {save_path}")
    dt.save_to_disk(save_path)
    print (f"Saved dataset to {save_path}")

def precompute_model_spectra(
        args: argparse.Namespace
):
    path = args.dataset
    if args.use_less_data:
        path = path.rsplit('/', 1)[0]
        path = f"{path}/eval"
        
    dt = load_from_disk(path)
    if args.use_less_data:
        dt = dt.select([i for i in range(100)])
        
    embedder_model, embedder_tokenizer = mu.load_embedder_and_tokenizer(
        'SELFormer',
        with_formula=False,
        ATOM_LIST=['I', 'C', 'S', 'N', 'Br', 'H', 'O', 'Si', 'Cl', 'P', 'F']
    )
    
    embedder_model.to("cuda")
    
    def compute_hypothesis_fn(item, idx):
        if (idx[-1] + 1) % 512 == 0:
            logger.info(f"Processing example index: {idx[-1]}")
            
        input_ids = torch.tensor(
            [i[:-2] for i in item['embedder_input_ids']]
        ).to("cuda")
        
        attention_mask = torch.tensor(
            [i[:-2] for i in item['embedder_attention_mask']]
        ).to("cuda")
        
        threshold = 0.9 * torch.cuda.get_device_properties(0).total_memory
        
        if torch.cuda.memory_reserved() > threshold:
            print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
            torch.cuda.empty_cache()
            print ("### Emptying CUDA cache ###")
            
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            
            spectra, _, _ = embedder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            item['model_spectra'] = spectra
            
        return item

    try:
        dt = dataset_map_multi_worker(dt, compute_hypothesis_fn)
    except OSError as e:
        import traceback
        traceback.print_exc()
        
    type_dataset = path.rsplit('/', 1)[-1]
    path_dataset = path.rsplit('/', 1)[0]
    save_path = f"{path_dataset}/perfect_model/{type_dataset}"
    if args.use_less_data:
        save_path = f"{path_dataset}/perfect_model/debug"
        
    print (f"Dataset will be saved to {save_path}")
    dt.save_to_disk(save_path)
    print (f"Saved dataset to {save_path}")
    

def precompute_hypotheses(
        args : argparse.Namespace
):
    path = args.dataset
    type_dataset = path.rsplit('/', 1)[-1]
    
    dt = load_from_disk(path)
    if args.use_less_data:
        dt = dt.select([i for i in range(1500, 2000)])
        
    model = im.InversionModel.from_pretrained(args.invertion_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    embedder_model, embedder_tokenizer = mu.load_embedder_and_tokenizer(
        'SELFormer',
        with_formula=model.with_formula,
        ATOM_LIST=ATOM_LIST,
    )

    print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
    
    model.to("cuda:0")
    embedder_model.to("cuda:1")
    model.eval()
    embedder_model.eval()

    print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
    
    tokenize_fn = tokenize_function(
        tokenizer = embedder_tokenizer,
        text_column_name = args.text_column_name,
        max_seq_length = args.max_seq_length,
        padding_strategy = "max_length"
    )
    
    def compute_hypothesis_fn(item, idx):
        if (idx[-1] + 1) % 512 == 0:
            logger.info(f"Processing example index: {idx[-1]}")

        print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
        print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
            
        inputs = {
            'spectre': torch.tensor(item['model_spectra'], device="cuda:0")
        }
        if model.with_formula:
            inputs['formula'] = torch.tensor(item['formula'], device="cuda:0")

        print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
        print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
        
        threshold = 0.9 * torch.cuda.get_device_properties(0).total_memory
        
        gen_kwargs = {
            "early_stopping": True,
            "num_beams": 10,
            "do_sample": False,
            "max_length": 514,
            "num_return_sequences": 1,
        }
        
        if torch.cuda.memory_reserved() > threshold:
                print(f"Allocated : {torch.cuda.memory_allocated() / 1e6} MB")
                print(f"Reserved : {torch.cuda.memory_reserved()/1e6:.2f} MB")
                torch.cuda.empty_cache()
                print ("### Emptying CUDA cache ###")
                
        with torch.no_grad(), torch.autocast(device_type="cuda:0", dtype=torch.float16):
            outputs = model.generate(
                inputs,
                gen_kwargs
            )

        outputs_cpu = outputs.detach().cpu()
        decoded = tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)
        del outputs, inputs

        item['hypothesis_input_ids'] = outputs_cpu
        item['hypothesis_selfies'] = decoded
        
        hypothesis_tokenized = tokenize_fn(
            {args.text_column_name : decoded},
        )
        
        hypothesis_tokenized_input_ids = torch.tensor(hypothesis_tokenized['input_ids']).to("cuda:1")
        hypothesis_tokenized_attention_mask = torch.tensor(hypothesis_tokenized['attention_mask']).to("cuda:1")
        selfies = item['selfies'] if model.with_formula else None

        with torch.no_grad(), torch.autocast(device_type="cuda:1", dtype=torch.float16):
            
            hypothesis_spectra, _, hypothesis_formula = embedder_model(
                hypothesis_tokenized_input_ids, hypothesis_tokenized_attention_mask, selfies=selfies
            )

        item['hypothesis_spectra'] = hypothesis_spectra.detach().cpu()  # Optional: keep on CPU
        if hypothesis_formula is not None:
            item['hypothesis_formula'] = hypothesis_formula.detach().cpu()

        del hypothesis_tokenized_input_ids, hypothesis_tokenized_attention_mask, hypothesis_spectra, hypothesis_formula
            
        return item
    
    dt = dataset_map_multi_worker(dt, compute_hypothesis_fn)
    
    save_path = f"{arrow_dataset_path}/hypothesis_dataset/perfect_model/{type_dataset}"
    if model.with_formula:
        save_path = f"{arrow_dataset_path}/hypothesis_dataset/perfect_model/with_formula/{type_dataset}"
    if args.use_less_data:
        save_path = f"{os.environ['VEC2TEXT_CACHE']}/debug"
    
    print (f"Dataset will be saved to {save_path}")
    dt.save_to_disk(save_path)
    print (f"Saved dataset to {save_path}")
    

def main(args):
    if args.task == "csv_to_arrow":
        csv_to_arrow_dataset()
    elif args.task == "tokenize_dataset":
        tokenize_dataset(args)
    elif args.task == "precompute_hypotheses":
        precompute_hypotheses(args)
    elif args.task == 'compute_formula':
        add_chemical_formula_dataset(args.dataset, '')
    elif args.task == 'compute_model_spectra':
        precompute_model_spectra(args)
    else :
        print ("You must provide a --task argument. OPTIONS : [csv_to_arrow, tokenize_dataset, precompute_hypotheses, compute_formula]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",                  
        type=str,                       
        required=False,
        default="test",
        help="The path of the dataset you want to work with"
    )

    parser.add_argument(
        "--tokenizer_name_or_path",                  
        type=str,                       
        required=False,
        default = "./data/saved_models/models--zjunlp--MolGen-large/snapshots/3c6e8fb91a783853a3782552a85efc4df6f96d0a",
        help="The tokenizer you want to use on this dataset"
    )
    
    parser.add_argument(
        "--invertion_model_name_or_path",
        type=str,
        required=False,
        default = "./data/saved_models/Inverter_PerfectModel"
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=False,
        default=512,
    )

    parser.add_argument(
        "--text_column_name",
        type=str,
        required=False,
        default="selfies"
    )

    parser.add_argument(
        "--use_less_data",
        action="store_true",
    )

    parser.add_argument(
        "--task",
        type=str,
        required = False,
        default = None
    )

    args = parser.parse_args()
    main(args)

    
