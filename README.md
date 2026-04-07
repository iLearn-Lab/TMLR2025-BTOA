# BTOA(TMLR 2025)

> Official Implementation of the  Paper: A Transferable Augmentation Framework to Combat Distribution Shifts(TMLR 2025)

## Authors

Weiyang Zhang<sup>1</sup>, Xinyang Chen<sup>1,​ 📧</sup>, Yu Sun<sup>2, 📧</sup>, Weili Guan<sup>3</sup>, Liqiang Nie<sup>1</sup>

<sup>1</sup> School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)
<sup>2</sup> College of Computer Science, DISSec, Nankai University
<sup>3</sup> School of Information Science and Technology, Harbin Institute of Technology (Shenzhen)

## Introduction

BTOA includes three core modules:

1. **THSS**: Selects historical samples most similar to the test distribution.
2. **TOA**: Performs two-stream augmentation in frequency domain (amplitude & phase).
3. **Prediction Block**: Uses series decomposition and dual forecasters to improve accuracy.

BTOA is a general plug-and-play module that can be integrated into any existing forecasting model.

## Dataset

We follow the experimental settings of [Proceed](https://github.com/SJTU-DMTai/OnlineTSF), and the required datasets can be found in [thuml/Time-Series-Library: A Library for Advanced Deep Time Series Models for General Time Series Analysis.](https://github.com/thuml/Time-Series-Library)

## Usage

### Install

```
conda create -n BTOA python==3.11
conda activate BTOA
pip install -r requirements.txt
```

### Arguments

Basic arguments:

- `--model` decides the forecast backbone.
- `--seq_len` decides the lookback length.
- `--pred_len` decides the forecast horizon.
- `--dataset` decides the dataset of which the file path is configured in `settings.py`.
- `--learning_rate` controls the learning rate when training on historical data.
- `--online_learning_rate`: controls the learning rate when training on online data.

Hyperparameters of BTOA:

* `--save`: Path to save the VAE model.
* `--aug_number`: Number of samples for data augmentation.

### Model Pretraining Script

Take ETTh1 as an example

```
export CUDA_VISIBLE_DEVICES=5
if [ ! -d "./logs" ]; then
mkdir ./logs
fi

if [ ! -d "./logs/ETTh1" ]; then
mkdir ./logs/ETTh1
fi

seq_len=512
data=ETTh1
model_name=PatchTST
train_epochs=100
pct=0.4

for pred_len in 24
do
for learning_rate in 0.0002
do
python -u run.py \
--dataset $data \
--border_type 'online' \
--model $model_name \
--seq_len $seq_len \
--pred_len $pred_len \
--itr 3 \
--only_test \
--pin_gpu True \
--reduce_bs False \
--save_opt \
--batch_size 128 \
--train_epochs $train_epochs \
--pct $pct \
--patience 10 \
--learning_rate $learning_rate >> ./logs/ETTh1/$model_name'_'$data'_'$pred_len'_lr'$learning_rate.log 2>&1
done
done
```

### Online Test-Time Adaptation Script

Take ETTh1 as an example

```
if [ ! -d "./logs" ]; then
mkdir ./logs
fi

if [ ! -d "./logs/ETTh1" ]; then
mkdir ./logs/ETTh1
fi

seq_len=512
data=ETTh1
model_name=PatchTST
online_method=BTOA

for pred_len in 24
do
for online_learning_rate in 0.00001
do
filename=logs/ETTh1/$model_name'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log2
python -u run.py \
--dataset $data \
--border_type 'online' \
--model $model_name \
--seq_len $seq_len \
--pred_len $pred_len \
--itr 3 \
--skip $filename \
--online_method $online_method \
--pretrain \
--pin_gpu True \
--reduce_bs False \
--save_opt \
--only_test \
--save 'ETTh1' \
--learning_rate 0.0002 \
--val_online_lr \
--lradj type3 \
--online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
```

## Citation

```
@article{zhang2025batch,
  title={Batch Training for Streaming Time Series: A Transferable Augmentation Framework to Combat Distribution Shifts},
  author={Zhang, Weiyang and Chen, Xinyang and Sun, Yu and Guan, Weili and Nie, Liqiang},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```



## License

This project is released under the ​**Apache License 2.0**​.
