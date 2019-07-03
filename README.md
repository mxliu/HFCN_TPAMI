Hierarchical Fully Convolutional Network for Structrual MRI-based Brain Disease Prognosis

The code was written by Dr. Chunfeng Lian, Department of Radiology at UNC at Chapel Hill. 

Introduction

We propose a hierarchical fully convolutional network (HFCN) to automatically identify discriminative local patches and regions in the whole brain sMRI, upon which multi-scale feature representations are then jointly learned and fused to construct hierarchical classification models for AD diagnosis. We have two implementation versions: 2) HFCN without prior knowlege (no_prior), and 2) HFCN with prior knowledge (with_prior).


Prerequisites

Linux python 2.7

Keras version 2.0.8

NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 8.0.61

Installation

Install Keras and dependencies

Install numpywith pip install numpy


Files

a. Source Code: train_all.py, train_pruned_step3.py, nhfcn_network.py, loss.py, noprior_loader_all.py, predict_all.py, predict_pru_nhfcn.py, pruned_hfcn,py, and pruned_sequential_loader.py 

b. Pre-trained Model: ..\saved_model


If you use our code, please cite the following paper:

Chunfeng Lian, Mingxia Liu, Jun Zhang, and Dinggang Shen. Hierarchical Fully Convolutional Network for Joint Atrophy Localization and Alzheimer's Disease Diagnosis using Structural MRI. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.
