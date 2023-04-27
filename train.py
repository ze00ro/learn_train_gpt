from transformers import GPT2LMHeadModel, BertTokenizer
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 实验配置
model_id = "uer/gpt2-chinese-cluecorpussmall"
save_dataset_path = "pretrain_data"

train_data = load_from_disk(f"{save_dataset_path}/train")
print(f"Train pretrain_data size: {len(train_data)}")
# test_data = load_from_disk(f"{save_dataset_path}/test")
# print(f"Test pretrain_data size: {len(test_data)}")

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

# todo ? we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-sub"

# Define training args
training_args = TrainingArguments(
    output_dir="./model_output/" + repository_id,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=False,
    learning_rate=1e-4,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"./model_output/{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    # metric_for_best_model="overall_f1",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    # eval_dataset=test_data,
    # compute_metrics=compute_metrics,
)

trainer.train()
tokenizer.save_pretrained(repository_id)
trainer.save_model()
