export CUDA_VISIBLE_DEVICES=5
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "././logs/ETTh1" ]; then
    mkdir ././logs/ETTh1
fi

seq_len=512
data=ETTh1
model_name=PatchTST
train_epochs=100
pct=0.4

for pred_len in 24 48
do
for learning_rate in 0.0002
do
  python -u run.py \
    --dataset $data --bordBTOA_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --only_test \
    --pin_gpu True --reduce_bs False \
    --save_opt \
    --batch_size 128 \
    --train_epochs $train_epochs \
    --pct $pct \
    --patience 10 \
    --learning_rate $learning_rate >> ./logs/ETTh1/$model_name'_'$data'_'$pred_len'_lr'$learning_rate.log 2>&1
done
done
for pred_len in 96
do
for learning_rate in 0.0003
do
  python -u run.py \
    --dataset $data --bordBTOA_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --only_test \
    --pin_gpu True --reduce_bs False \
    --save_opt \
    --batch_size 128 \
    --train_epochs $train_epochs \
    --pct $pct \
    --patience 10 \
    --learning_rate $learning_rate >> ./logs/ETTh1/$model_name'_'$data'_'$pred_len'_lr'$learning_rate.log 2>&1
done
done

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
    --dataset $data --bordBTOA_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --pretrain \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --save 'ETTh1' \
    --learning_rate 0.0002 \
    --val_online_lr \
    --lradj type3 \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done

for pred_len in 48
do
for online_learning_rate in 0.000001
do
  filename=logs/ETTh1/$model_name'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log2
  python -u run.py \
    --dataset $data --bordBTOA_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --pretrain \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --save 'ETTh1' \
    --learning_rate 0.0002 \
    --val_online_lr \
    --lradj type3 \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done

for pred_len in 96
do
for online_learning_rate in 0.000003
do
  filename=logs/ETTh1/$model_name'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log2
  python -u run.py \
    --dataset $data --bordBTOA_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --pretrain \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --save 'ETTh1' \
    --learning_rate 0.0003 \
    --val_online_lr \
    --lradj type3 \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done