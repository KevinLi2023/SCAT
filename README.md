## [ICASSP 2025] SCAT: Shared-Convolution Adaptation Tuning for Foreground Segmentation

Kaiwen Li, Dezheng Gao, Zelin Yang, and Xing Wei by Xi’an Jiaotong University

### NOTE

"The code for the paper is currently being organized and will be released soon."

## Abstract

> Fine-tuning a minimal subset of parameters in large well-trained models has emerged as a popular paradigm for transforming prior knowledge to address downstream tasks in computer vision. Although it has shown promising performance in certain vision tasks such as classification, parameter-efficient tuning remains in its infancy and suffers a significant accuracy drop compared to tuning the entire model particularly in field of segmentation. In this paper, we propose a novel tuning method named SCAT (Shared-Convolution Adaptation Tuning), designed to adapt segmentation models to various fine-grained foreground segmentation vision tasks. By injecting strong inductive bias prompts with shared convolutional features into the frozen backbones, SCAT significantly increases the transferability of the pre-trained models with only a few learnable parameters. SCAT delivers superior performance compared to other state-of-the-art fine-tuning methods, domain-specific hand-crafted networks, and even the fully-tuning paradigm across numerous foreground segmentation scenarios. 

## Overview

![image](https://github.com/KevinLi2023/SCAT/blob/main/imgs/image1.png)

An overview of our proposed SCAT. SCAT is composed of SC-Adapters, each of which is inserted before every transformer block via residual connections. Note that convolution kernels are shared among SC-Adapters within a transformer stage.

## Dataset

- **COD10K**: https://github.com/DengPingFan/SINet/
- **CAMO**: https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6
- **CHAMELEON**: https://www.polsl.pl/rau6/datasets/

- **DUT**: http://ice.dlut.edu.cn/ZhaoWenda/BTBCRLNet.html
- **CUHK**: http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/

- **CAISA**: https://github.com/namtpham/casia2groundtruth
- **IMD2020**: http://staff.utia.cas.cz/novozada/db/

- **ISTD**: https://github.com/DeepInsight-PCALab/ST-CGAN
- **SBU**: https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html

