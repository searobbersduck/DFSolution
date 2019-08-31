CUDA_VISIBLE_DEVICES=0,1,2,3 python sentiment_analysis_with_xlnet.py \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --task_name=sentiment \
  --data_dir=./data/ \
  --output_dir=./xlnet_output \
  --predict_dir=./xlnet_output \
  --model_dir=./model/ \
  --uncased=False \
  --spiece_model_file=./model/spiece.model \
  --model_config_path=./model/xlnet_config.json \
  --init_checkpoint=./model/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=4 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=5e-5 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True