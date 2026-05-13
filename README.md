# Spectroscopy-Informed Inverse Molecular Design via Invertible LLMs

This repository contains the code for our paper:

> **Spectroscopy-Informed Inverse Molecular Design via Invertible Large Language Models**  
> Lazare Ricour-Dumas et al. — submitted to IEEE WCCI 2026

## Overview

We propose a novel approach to inverse molecular design: given an IR spectrum, generate a chemically valid molecular structure whose vibrational signature is consistent with the input. Rather than training an end-to-end spectrum-to-structure model, we frame the problem as **embedding inversion** using the Vec2Text framework.

The pipeline has two stages:
1. Fine-tune a chemistry-aware LLM ([SELFormer](https://github.com/HUBioDataLab/SELFormer)) on the forward structure-to-spectrum task (SELFIES → IR spectrum)
2. Apply Vec2Text to approximate the inverse of this model, enabling iterative generation of SELFIES strings from IR spectra

All generated molecules are **100% valid** in SELFIES format.

## Results

| Setting | SIS (↑) | Tanimoto (↑) |
|---|---|---|
| IR only | 0.791 ± 0.054 | 0.256 ± 0.133 |
| IR + formula | 0.860 ± 0.054 | 0.306 ± 0.168 |

High spectral consistency is achieved despite lower molecular similarity, reflecting the intrinsic non-uniqueness of the spectrum-to-structure mapping.

## Installation

```bash
git clone https://github.com/lazare-rd/spectro-inverse
cd spectro-inverse
pip install -r requirements.txt
```

## Dataset

We use the IR spectroscopy benchmark introduced by [McGill et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00055), comprising ~86K computed and ~30K experimental spectra. Dataset split follows Murcko scaffold partitioning.

## Usage

```bash
# Train the embedding model (SELFormer fine-tuning)
python train_emb.py --config configs/emb.yaml

# Train the base model and corrector (Vec2Text)
python train_inv.py --config configs/inv.yaml

# Run inference
python predict.py --spectrum data/example_spectrum.npy --steps 3
```

## Citation

```bibtex
@article{ricour2026spectro,
  title={Spectroscopy-Informed Inverse Molecular Design via Invertible Large Language Models},
  author={Ricour-Dumas, Lazare and others},
  journal={IEEE WCCI 2026},
  year={2026}
}
```
