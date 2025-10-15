import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaPreTrainedModel,
    RobertaModel,
)

import spec2selfies.models.model_utils as utils
import selfies as sf
from collections import Counter
from rdkit import Chem

class SpectralHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        hidden_size = config.hidden_size
        bottleneck_size = hidden_size // 2
        expand_size = 2200
        
        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            
            nn.Linear(bottleneck_size, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.layers(x)
        x = self.out_proj(x)
        return x

class MolToSpec(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, freeze_selformer: bool = False):
        super(MolToSpec, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.tokenizer = utils.load_tokenizer(
            config.tokenizer_name,
            max_length=config.max_position_embeddings,
        )

        if freeze_selformer: # freeze all SELFormer parameters
            for name, param in self.named_parameters():
                param.requires_grad = False
        else : # unfreeze last 3 BERT blocks
            for name, param in self.named_parameters():
                if (
                        "encoder.layer.10" not in name
                    and "encoder.layer.11" not in name
                    and "encoder.layer.9" not in name
                ):
                    
                    param.requires_grad = False
                    
        self.classifier = SpectralHead(config)
        self.with_formula = config.with_formula
        self.ATOM_LIST = config.ATOM_LIST

        self.config.id2label = {}
        self.config.label2id = {}
        
    def forward(self, input_ids, attention_mask, selfies=None, return_embedding=False):

        formulas = None
        if self.with_formula:
            assert selfies is not None
            formulas = self.get_formula_vec(selfies)
        
        outputs = self.roberta(input_ids, attention_mask)
        sequence_output = outputs[0]
        spectras = self.classifier(sequence_output)
        spectras = torch.exp(spectras)
        embeddings = None

        if return_embedding:
            embeddings = sequence_output[:, 0, :]

        return spectras, embeddings, formulas


    def get_formula_vec(self, selfies):
        smiles = [sf.decoder(s) for s in selfies]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        
        atom_counts = [
            Counter(
                atom.GetSymbol() for atom in m.GetAtoms()
            ) if m is not None else Counter()
            for m in mols
        ]
        
        formulas = [
            [ac.get(atom, 0) for atom in self.ATOM_LIST]
            for ac in atom_counts
        ]
        
        return torch.tensor(formulas).to(next(self.parameters()).device)

   
