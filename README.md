# CAMERA: Cross-feature Aligned ModEl for Recognizing top Antimicrobial peptides

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Installation

```bash
# clone project
cd CAMERA

# create conda virtual environment
conda create -n torch1.12 python=3.8 
conda activate torch1.12

# install all requirements
pip install -r requirements.txt
```

## Quick Usage of CAMERA

### 1. Prepare datasets before using CAMERA

#### Datasets introduction

Before utilizing CAMERA, it's important to prepare the datasets appropriately. Our research utilized several dataset versions, and it's crucial to have **all the following versions** of datasets ready before running CAMERA:

 - ori_datasets
    - Format: **.csv**
    - Description: This includes `train.csv`, `val.csv`, and `test.csv`. These datasets exclusively contain sequences and corresponding labels.
    - Obtaining Method: https://pan.baidu.com/s/1wOH2WdvTodhvBO_GrXj67w?pwd=8n8k
 - esm_embeddings
    - Format: **.h5**
    - Description: These datasets are in .h5 format and are generated using the esm-2 model. This version of the dataset is derived from the `ori_datasets`.
    - Obtaining Method: By running the script `tools/esm2_emb_gen.py`
 - des_info
    - Format: **.h5**
    - Description: These datasets are in .h5 format and are obtained by calculating protein descriptors based on the sequences. This version of the dataset is derived from the `ori_datasets`.
    - Obtaining Method: By running the script `tools/generate_des_csv.py` and `tools/des.py`
 - stc_info
    - Format: **.h5**
    - Description: These datasets are in .h5 format and are obtained by calculating secondary structures based on the sequences. This version of the dataset is derived from the `ori_datasets`.
    - Obtaining Method: By running the script `tools/stc.py`

### 2. Download our model checkpoints to quickly reproduce our results

https://pan.baidu.com/s/1wOH2WdvTodhvBO_GrXj67w?pwd=8n8k

### 3. Run CAMERA
In this project, the model, datasets, and hyperparameters are all setted in `config.py`. Therefore, before running `run.py`, please ensure that the corresponding `config.py` is correctly configured.

#### 3.1 Train CAMERA
For example, train CAMERA on our proposed datasets.

```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 run.py \
--config ./configs/CAMERA_Saureus.py --mode train
```

#### 3.2 Evaluate with CAMERA on the test set.
For example, evaluate with CAMERA on the test set of our proposed datasets.
(Make sure you have modified `ckpt_path` to checkpoint in the config file.)
```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 run.py \
--config ./configs/CAMERA_Saureus.py --mode test
```

## Credit

This repo is based on [SenseXAMP](https://github.com/William-Zhanng/SenseXAMP)