export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path /home/shays/LIGHTBITS/UniDataSet.csv \
  --model_id UNI \
  --model TimesNet \
  --data UNI \
  --features S \
  --seq_len 60 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 1 \
  --c_out 1 \
  --top_k 2 \
  --anomaly_ratio 1 \
  --batch_size 80 \
  --train_epochs 20

  #Notice for the flag shuffle in data_factory-if training put it on the condition, otherwise make it False
