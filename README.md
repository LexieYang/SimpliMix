# SimpliMix

Official implementation of `SimpliMix: A Simplified Manifold Mixup for Few-shot Point Cloud Classification` (accepted by WACV 2024).

## Installation

This project is built upon the following environment:
* Install Python 3.7
* Install CUDA 11.0
* Install PyTorch 1.10.2

The package requirements include:
* pytorch==1.10.2
* tqdm==4.63.1
* tensorboard==2.8.0

## Datasets

* Download [ModelNet40](https://modelnet.cs.princeton.edu/)
* Download [ModelNet40-C from Google Drive](https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX). In our experiments, we only use the LiDAR corruption pattern.
* Download [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)
* Download [ShapeNet](https://shapenet.org/)

## Train
Train protonet in the 5-way 1-shot setting on the ModelNet40 dataset:
```
python main.py --dataset modelnet40 --fs_head protonet --backbone dgcnn --k_way 5 --n_shot 1
```
Train protonet with simplimix in the 5-way 1-shot setting on the ModelNet40 dataset:
```
python main.py --dataset modelnet40 --fs_head protonet --backbone dgcnn_mm4 --k_way 5 --n_shot 1
```
## Evaluate
```
python main.py --train False
