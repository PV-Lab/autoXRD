Description
===========
This repository contains data for automatic XRD pattern classification of thin-film using scarce / sparse and class imbalanced datasetes. 
The main application is material screen of solution-processed materials such as perovskites and perovskite-inspired materiales. 

The algorithm performs physics-informed data augmentation and interpretation using average Class Activation Maps, according to the following publications:

XXX
XXX
XXX

Usage
======================
The code constains three scripts with differente approaches to the problem:

AUTOXRD_CNN_Dimensionality.py - Performs classification of XRD patterns into 3 dimensionalities using a-CNN
AUTOXRD_CNN_Space_Group.py - Performs classsification of XRD patterns into 7 space-groups using a-CNN.
Demo / XRD_demo.ipynb - Jupyter Notebook containing a demo for physics-informed data augmentation, as presented in the Accelerated Materials Development (AMD) Workshop in Singapore.

In the future, the code will be included as a PIP package.

Information
===========
:Authors:
    Felipe Oviedo and "Danny" Zekun Ren

:Version: AUTOXRD_CNN is in version 0.9 as of February 2019

Attribution
===========
Please cite this work as:
