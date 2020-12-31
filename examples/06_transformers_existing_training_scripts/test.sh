export TASK_NAME=mrpc
export SM_CHANNELS=["test","train"]
export SM_OUTPUT_DATA_DIR=/opt/ml/output/data
export SM_MODEL_DIR=/opt/ml/model
export SM_CHANNEL_TEST=/opt/ml/input/data/test
export SM_CHANNEL_TRAIN=/opt/ml/input/data/train


  python ../../transformers/examples/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train True \
  --do_eval True \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --train_file train.csv \
  --test_file test.csv 