
autoXRD
===========
## Description


autoXRD is a python package for automatic XRD pattern classification of thin-films, tweaked for small and class-imbalanced datasets. The main application of the package is high-throughput screening of novel materials.

autoXRD performs physics-informed data augmentation to solve the small data problem, implements a state-of-the-art a-CNN architecture and allows interpretation using Average Class Activation Maps (CAMs), according to the following publications:

"**Oviedo, F., Ren, Z., Sun, S., Settens, C., Liu, Z., Hartono, N. T. P., ... & Kusne, A. G. (2019). Fast and interpretable classification of small X-ray diffraction datasets using data augmentation and deep neural networks. npj Computational Materials, 5(1), 60."  Link: [https://doi.org/10.1038/s41524-019-0196-x](https://doi.org/10.1038/s41524-019-0196-x)


"**Fast and interpretable classification of small X-ray diffraction datasets using data augmentation and deep neural networks, (2019), Felipe Oviedo, Zekun Ren, et. al.  Link: [arXiv:1811.08425v](https://arxiv.org/abs/1811.08425v2)**

**Accepted to NeurIPS 2018 ML for Molecules and Materials. Final version published npj Computational Materials 2019**


## Installation

To install, just clone the following repository:

`$ git clone https://github.com/PV-Lab/autoXRD.git`

## Usage

Just run `space_group_a_CNN.py` , with the given datasets. Note that this performs classification for patterns into 7 space-groups. Dimensionality data is not included in the code, please contact authors if interested.
The package contains the following module and scripts:

| Module | Description |
| ------------- | ------------------------------ |
| `space_group_a_CNN.py`      | Script for XRD space-group classification with a-CNN      |
| `autoXRD`      | Module dedicated to XRD pattern preprocessing and data augmentation       |
| `autoXRD_vis`   | Visualizer module for class activation maps (CAMs)     |
| `Old-Demo / XRD_demo.ipynb` | Notebook containing a demo for physics-informed data augmentation, as presented in the **Accelerated Materials Development (AMD) Workshop** in Singapore. This is an outdated version without a-CNNs or CAM|


## Authors
Felipe Oviedo and "Danny" Zekun Ren


||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Felipe Oviedo and "Danny" Ren Zekun     | 
| **VERSION**      | 0.9 / April, 2019     | 
||                    |

## Attribution

This work is under an Apache 2.0 License and data policies of Nature Partner Journal Computational Materials. Please, acknowledge use of this work with the apropiate citation.

### Citation

    @article{oviedo2019fast,
  title={Fast and interpretable classification of small X-ray diffraction datasets using data augmentation and deep neural networks},
  author={Oviedo, Felipe and Ren, Zekun and Sun, Shijing and Settens, Charles and Liu, Zhe and Hartono, Noor Titan Putri and Ramasamy, Savitha and DeCost, Brian L and Tian, Siyu IP and Romano, Giuseppe and others},
  journal={npj Computational Materials},
  volume={5},
  number={1},
  pages={60},
  year={2019},
  publisher={Nature Publishing Group}
  howpublished = {\url{https://doi.org/10.1038/s41524-019-0196-x}},
}
