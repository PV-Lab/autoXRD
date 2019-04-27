
autoXRD
===========
## Description


autoXRD is a python package for automatic XRD pattern classification of thin-films, tweaked for small and class-imbalanced datasets. 

autoXRD's main application is high-throughput screening of novel materials.

autoXRD performs physics-informed data augmentation to solve the small data problem, implements a state-of-the-art a-CNN architecture and allows interpretation using Average Class Activation Maps (CAMs), according to the following publications:

"**Fast and interpretable classification of small X-ray diffraction datasets using data augmentation and deep neural network, (2019), Felipe Oviedo, Zekun Ren, et. al.  Link: [arXiv:1811.08425v](https://arxiv.org/abs/1811.08425v2)**


## Installation

To install, just clone the following repository:

`$ git clone https://github.com/PV-Lab/autoXRD.git`

## Usage

Just run `space_group_a_CNN.py` , with the given datasets. 
The package contains the following module and scripts:

| Module | Description |
| ------------- | ------------------------------ |
| `space_group_a_CNN.py`      | Script for XRD space-group classification with a-CNN      |
| `autoXRD`      | Module dedicated to XRD pattern preprocessing and data augmentation       |
| `autoXRD_vis`   | Visualizer module for class activation maps (CAMs)     |
| `Old-Demo / XRD_demo.ipynb` | Notebook containing a demo for physics-informed data augmentation, as presented in the **Accelerated Materials Development (AMD) Workshop** in Singapore. |


## Authors
Felipe Oviedo and "Danny" Zekun Ren


||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Felipe Oviedo and "Danny" Ren Zekun     | 
| **VERSION**      | 0.9 / April, 2019     | 
||                    |

## Attribution

This work is under an Apache 2.0 License and data policies of Nature Partner Journal Computational Materials. Please, acknowledge use of this work with the apropiate citation.
