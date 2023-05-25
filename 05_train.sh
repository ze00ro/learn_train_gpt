export TRAIN_FILE=05_data/train.txt
export TEST_FILE=05_data/test.txt
export GPT2_MODEL_PATH=/llm/models/gpt2_chinese_cluecorpussmall

python ./run_clm.py \
    --output_dir model_output_clm/ \
    --model_name_or_path $GPT2_MODEL_PATH \
    --tokenizer_name $GPT2_MODEL_PATH \
    --do_train \
    --train_file $TRAIN_FILE \
    --do_eval \
    --keep_linebreaks True \
    --validation_file=$TEST_FILE \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --save_strategy=epoch \
    --save_total_limit=5 \
    --learning_rate=1e-3 \
    --num_train_epochs=10 \
    --evaluation_strategy=steps \
    --eval_steps=80 \
    --fp16 \
    --overwrite_output_dir

#    --gradient_checkpointing=True \
#    --low_cpu_mem_usage=True \
