from spec2selfies.eval_utils import load_logs
from spec2selfies.trainers.corrector_trainer import SISMeasure
from transformers import AutoTokenizer
import torch
import wandb
import argparse
import yaml
from tqdm import tqdm
import gc

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import selfies as sf
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

valid_logs = ['text_ids', 'embeddings', 'spectras', 'bm_spectra', 'scores', 'initial_score', 'bm_text_ids']
LOGS_PATH = "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/jolly-clear-chow"
tokenizer = AutoTokenizer.from_pretrained("./data/saved_models/Inverter__30epochs__encoder_decoder_frozen/checkpoint-15411")
dataset = load_from_disk("./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/test")

def decode_text_ids(text_ids, _id=0, step=0, beam=0): 
    if len(text_ids.shape) <= 2:
        decoded = tokenizer.decode(text_ids[_id,:].astype('uint8'), skip_special_tokens=True)
    else :
        decoded = tokenizer.decode(text_ids[_id, step, beam, :], skip_special_tokens=True)
    return decoded

def load(_type):
    assert _type in valid_logs
    path = f"{LOGS_PATH}/{_type}"
    return load_logs(path)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tqdm_wrapper(iterable, use_tqdm=False, **tqdm_kwargs):
    return tqdm(iterable, **tqdm_kwargs) if use_tqdm else iterable

    
def get_fp_fn(fp):
    if fp == 'morgan':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=5)
        return fpgen

    elif fp == 'rdkit':
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
        return fpgen

    elif fp == 'atom_pair':
        fpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        return fpgen

    elif fp == 'topological_torsion':
        fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
        return fpgen

    elif fp == 'none':
        def identity(mol):
            return mol
        return identity

    else:
        raise ValueError(f"This fingerprint function '{fp}' is not implemented")



def get_sim_fn(sim_fn):
    if sim_fn == 'tan':
        def tanimoto(fp1, fp2):
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        return tanimoto

    elif sim_fn == 'dice':
        def dice_sim(fp1, fp2):
            return DataStructs.DiceSimilarity(fp1, fp2)
        return dice_sim

    elif sim_fn == 'cosine':
        def cosine_sim(fp1, fp2):
            return DataStructs.CosineSimilarity(fp1,fp2)
        return cosine_sim

    elif sim_fn == 'bleu_score':
        def bleu(s1, s2):
            # SELFIES input assumed
            s1_tokens = sf.split_selfies(s1)
            s2_tokens = sf.split_selfies(s2)
            return sentence_bleu([s1_tokens], s2_tokens, smoothing_function=SmoothingFunction().method1)
        return bleu

    elif sim_fn == 'rouge_score':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        def rouge(s1, s2):
            return scorer.score(s1, s2)['rougeL'].fmeasure
        return rouge

    else:
        raise ValueError(f"This similarity function '{sim_fn}' is not implemented")


def init_fp_and_sim_fn(config):
    fp_fn = {}
    sim_fn = {}
    for fp in config.get('fingerprints'):
        fn = get_fp_fn(fp)
        fp_fn[fp] = fn
    for sim in config.get('sim_functions'):
        fn = get_sim_fn(sim)
        sim_fn[sim] = fn

    return fp_fn, sim_fn

def compare_mols(mol, mol_ref, fp_fn_dico, sim_fn_dico, config):
    scores = {}
    smiles = sf.decoder(mol)
    smiles_ref = sf.decoder(mol_ref)
    mol = Chem.MolFromSmiles(smiles)
    mol_ref = Chem.MolFromSmiles(smiles_ref)
    if mol is None:
        return None
    if mol_ref is None:
        return None
    for name_fp, fp_gen in fp_fn_dico.items():    
        fp = fp_gen.GetFingerprint(mol)
        fp_ref = fp_gen.GetFingerprint(mol_ref)
        for name_sim, sim_fn in sim_fn_dico.items():
            fp_group = config['fingerprints'][name_fp]['group']
            sim_group = config['sim_functions'][name_sim]['group']
            if fp_group == sim_group:
                scores[f"{name_fp}_{name_sim}"] = sim_fn(fp, fp_ref)
    
    return scores


def update_scores(scores, beam_scores, step):
    for name, value in beam_scores.items():
        name = f"{name}_step{step+1}"
        scores[name].append(value)

    return scores

def init_scores(config, step):
    scores = {}
    for fp in config.get('fingerprints'):
        for sim in config.get('sim_functions'):
            scores[f"{fp}_{sim}_step{step+1}"] = []
    return scores 

def log_scores(scores):

    # We return the mean accross beam
    for name, value in scores.items():
        scores[name] = np.mean(value)

    wandb.log(scores)


def eval_mol(config, text_ids):
    fp_fn_dico, sim_fn_dico = init_fp_and_sim_fn(config)
    invalid_counter = 0
    use_tqdm = config.get('use_tqdm')
    for _id in tqdm_wrapper(range(text_ids.shape[0]), use_tqdm=use_tqdm):
        if not use_tqdm and _id % 50 == 0:
            print (f"{(_id / text_ids.shape[0]) * 100}%")
        mol_ref = dataset[_id].get('selfies')
        scores_steps = []
        for step in range(text_ids.shape[1]):
            scores = init_scores(config, step)
            for beam in range(text_ids.shape[2]):
                mol = decode_text_ids(text_ids, _id, step, beam)
                beam_scores = compare_mols(mol, mol_ref, fp_fn_dico, sim_fn_dico, config)
                if beam_scores is not None:
                    scores = update_scores(scores, beam_scores, step)
                else:
                    invalid_counter += 1
            scores_steps.append(scores)

        merged_scores = {k: v for d in scores_steps for k, v in d.items()}
        log_scores(merged_scores)
    wandb.log({'invalid_molecules' : invalid_counter})


def eval_scores(config, scores):
    mean_scores = np.mean(scores, axis=2)
    use_tqdm = config.get('use_tqdm')
    for _id in tqdm_wrapper(range(scores.shape[0]), use_tqdm=use_tqdm):
        if not use_tqdm and _id % 50 == 0:
            print (f"{(_id / scores.shape[0]) * 100}%")
        scores_steps = []
        for step in range(scores.shape[1]):
            dico = {f'sis_step{step}': mean_scores[_id, step]}
            scores_steps.append(dico)
        merged_scores = {k: v for d in scores_steps for k, v in d.items()}
        wandb.log(merged_scores)

    gc.collect()
    max_scores = np.max(scores, axis=2).astype(np.float32)
    print ("### max scores ok ###")
    mean = np.mean(max_scores, axis=0)
    print (mean)
    print ("### mean scores ok ###")
    std = np.std(max_scores, axis=0)
    print (std)
    print ("### std scores ok ###")
    for i in range(mean.shape[0]):
        wandb.log({
            f'mean_sis_step{i}' : mean[i],
            f'std_sis_step{i}' : std[i],
        })

def eval_bm_text_ids(config, bm_text_ids):
    fp_fn_dico, sim_fn_dico = init_fp_and_sim_fn(config)
    invalid_counter = 0
    use_tqdm = config.get('use_tqdm')
    for _id in tqdm_wrapper(range(text_ids.shape[0]), use_tqdm=use_tqdm):
        if not use_tqdm and _id % 50 == 0:
            print (f"{(_id / text_ids.shape[0]) * 100}%")
        mol_ref = dataset[_id].get('selfies')
        scores = init_scores(config, 0)
        mol = decode_text_ids(bm_text_ids, _id)
        step_scores = compare_mols(mol, mol_ref, fp_fn_dico, sim_fn_dico, config)
        if step_scores is not None:
            scores = update_scores(scores, step_scores, 0)
        else:
            invalid_counter += 1

        log_scores(scores)
    wandb.log({'invalid_molecules' : invalid_counter})
    

def main(config):
    wandb.init()

    eval_items = config.get('eval_items')
    items = load(eval_items)
    print (f'### Analysing {eval_items} ###')
    if config.get('use_less_data') is not None:
        items = items[:config.get('use_less_data'), :]

    if eval_items == 'scores':
        eval_scores(config, items)
    elif eval_items == 'text_ids':
        eval_mol(config, items)
    elif eval_items == 'bm_text_ids':
        eval_bm_text_ids(config, items)



if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_less_data", type=int, default=None, help="Eval on a small subset to debug")
    parser.add_argument("--config_yaml", type=str, required=True, help="Yaml config to load from")
    
    args = parser.parse_args()
    config = load_config(args.config_yaml)
    config['use_less_data'] = args.use_less_data
    
    main(config)
