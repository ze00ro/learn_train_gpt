export TRAIN_FILE=05_data/train.txt
export TEST_FILE=05_data/test.txt
export GPT2_MODEL_PATH=/Volumes/backup/models/gpt2-chinese-cluecorpussmall

python transformers/examples/pytorch/language-modeling/run_clm.py \
    --output_dir model_output_clm/ \
    --model_name_or_path $GPT2_MODEL_PATH \
    --tokenizer_name $GPT2_MODEL_PATH \
    --do_train \
    --train_file $TRAIN_FILE \
    --do_eval \
    --keep_linebreaks True \
    --validation_file=$TEST_FILE \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --save_total_limit=5 \
    --learning_rate=5e-5 \
    --num_train_epochs=5 \
    --evaluation_strategy=epoch \
    --fp16 \
    --overwrite_output_dir
