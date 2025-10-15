from spec2selfies.eval_utils import load_logs
from transformers import AutoTokenizer

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem import AllChem
from ctypes import ArgumentError

import selfies as sf
from datasets import load_from_disk
from collections import Counter
from tqdm import tqdm
import torch
import pandas as pd
import argparse


valid_logs = [
    'text_ids',
    'embeddings',
    'spectras',
    'bm_spectra',
    'scores',
    'initial_score',
    'bm_text_ids',
    'formulas',
    'bm_formula'
]

LOGS_PATHS = {
    #"SELFormer_PerfectModel_WithFormula" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/namely-mutual-fawn",
    "SELFormer_WithFormula" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/subtly-vocal-stag",
    "SELFormer" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/fairly-meet-jay",
    #"SELFormer_PerfectModel" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/purely-casual-maggot",
    "ogSELFormer_PerfectModel_WithFormula" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/likely-amazed-pigeon",
    "SELFormer" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/widely-finer-marmot",
    "ogSELFormer_PerfectModel" : "/lustre/fsn1/projects/rech/aik/unh74gq/cache/vec2text/eval/oddly-hot-ram",
}

tokenizer = AutoTokenizer.from_pretrained(
    "./data/saved_models/Inverter__30epochs__encoder_decoder_frozen/checkpoint-15411"
)

dataset = load_from_disk(
    "./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/test"
)

def decode_text_ids(text_ids, _id=0, step=0, beam=0): 
    if len(text_ids.shape) <= 2:
        decoded = tokenizer.decode(text_ids[_id,:].astype('uint8'), skip_special_tokens=True)
    else :
        decoded = tokenizer.decode(text_ids[_id, step, beam, :], skip_special_tokens=True)
    return decoded

def load(_type, LOGS_PATH):
    assert _type in valid_logs
    path = f"{LOGS_PATH}/{_type}"
    return load_logs(path)

def get_mol(selfies: str):
    smiles = sf.decoder(selfies)
    return Chem.MolFromSmiles(smiles)

def get_sim(mol1, mol2, fpgen, sim_fn):
    try:
        fps1 = fpgen.GetFingerprint(mol1)
        fps2 = fpgen.GetFingerprint(mol2)
        #sim = DataStructs.TanimotoSimilarity(fps1,fps2)
        #sim = DataStructs.DiceSimilarity(fps1,fps2)
        #sim = DataStructs.SokalSimilarity(fps1,fps2)
        #sim = DataStructs.RusselSimilarity(fps1,fps2)
        #sim = DataStructs.KulczynskiSimilarity(fps1,fps2)
        #sim = DataStructs.McConnaugheySimilarity(fps1,fps2)
        sim = sim_fn(fps1,fps2)
    except Exception:
        sim = 0

    return sim

def build_dict(step, beam):
    dico = {}
    dico["step_0"] = []
    for i in range(step):
        for j in range(beam):
            # We add one at the step to account for step 0, which has no beam
            dico[f"step_{i+1}_beam_{j}"] = []
    return dico

def compute_all_simils(t, bm_t, dataset, fpgen, sim_fn):
    size = t.shape[0]
    step = t.shape[1]
    beam = t.shape[2]
    dico = build_dict(step, beam)
    
    for i in range(size):
        if i%10 == 0:
            print (f"{(i/size) * 100}%")
        selfies = decode_text_ids(bm_t, _id=i)
        mol = get_mol(selfies)
        true_mol = get_mol(dataset['selfies'][i])
        sim = get_sim(mol, true_mol, fpgen, sim_fn)
        dico["step_0"].append(sim)
        for j in range(step):
            for k in range(beam):
                selfies = decode_text_ids(t, _id=i, step=j, beam=k)
                mol = get_mol(selfies)
                true_mol = get_mol(dataset['selfies'][i])
                sim = get_sim(mol, true_mol, fpgen, sim_fn)
                dico[f"step_{j+1}_beam_{k}"].append(sim)
    return pd.DataFrame(dico)

def get_fpgen(name):
    if name == "rdkit":
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
    elif name == "ecfp4":
        fpgen = AllChem.GetMorganGenerator(radius=2)
    else:
        raise ValueError("This fpgen is not implemented")
    return fpgen

def get_simil_fn(name):
    if name == "cosine":
        fn = DataStructs.CosineSimilarity
    elif name == "tan":
        fn = DataStructs.TanimotoSimilarity
    else:
        raise ValueError("This similarity function is not implemented")
    return fn

def main(args):
    name = args.source
    path = LOGS_PATHS[name]
    print ("###############")
    print (f"### {name} ###")
    print ("###############")
    t = load("text_ids", path)
    bm_t = load("bm_text_ids", path)
    if args.use_less_data is not None:
        t = t[:args.use_less_data]
        bm_t = bm_t[:args.use_less_data]
        
    fpgen = get_fpgen(args.fpgen)
    sim_fn = get_simil_fn(args.simil_fn)
    df = compute_all_simils(t, bm_t, dataset, fpgen, sim_fn)
    file_name = f"./data/eval/{name}_{args.fpgen}_{args.simil_fn}.csv"
    if args.use_less_data is not None:
        file_name = "./data/eval/debug.csv"
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_less_data", type=int, default=None, help="Eval on a small subset to debug")
    parser.add_argument("--fpgen", type=str, required=True, help="Fingerprint Generator to use")
    parser.add_argument("--simil_fn", type=str, required=True, help="Simil fonction to use")
    parser.add_argument("--source", type=str, required=True, help="Source to use, must be in LOGS_PATHS")
    
    args = parser.parse_args()
    
    main(args)
