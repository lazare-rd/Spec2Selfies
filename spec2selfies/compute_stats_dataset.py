from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from ctypes import ArgumentError

import selfies as sf
from IPython.display import display
from collections import Counter
import pandas as pd

from datasets import load_from_disk

def get_mol(selfies: str):
    smiles = sf.decoder(selfies)
    return Chem.MolFromSmiles(smiles)


def main():
    dataset = load_from_disk("./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/train")
    dataset_test = load_from_disk("./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/test")
    
    l = []
    for i in range(len(dataset_test)):
        selfies = dataset_test['selfies'][i]
        mol = get_mol(selfies)
        nb = mol.GetNumHeavyAtoms()
        l.append(nb)
    df = pd.DataFrame(l, columns=['nb_heavy_atoms'])
    df.to_csv('./data/eval/nb_heavy_atoms_test.csv')

    l = []
    for i in range(len(dataset)):
        selfies = dataset['selfies'][i]
        mol = get_mol(selfies)
        nb = mol.GetNumHeavyAtoms()
        l.append(nb)
    df = pd.DataFrame(l, columns=['nb_heavy_atoms'])
    df.to_csv('./data/eval/nb_heavy_atoms_train.csv')


if __name__ == '__main__':
    main()
        
