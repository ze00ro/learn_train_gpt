import math

from datasets import load_from_disk
from transformers import GPT2LMHeadModel, BertTokenizer
from transformers import Trainer, TrainingArguments

from util.common import prepare_args

model_args, = prepare_args()

model_id = model_args.model_name_or_path
save_dataset_path = "pretrain_data"

train_data = load_from_disk(f"{save_dataset_path}/train")
print(f"Train pretrain_data size: {len(train_data)}")
test_data = load_from_disk(f"{save_dataset_path}/test")
print(f"Test pretrain_data size: {len(test_data)}")


# 也可以直接从文本来
# datasets = load_dataset("text", data_files={"train": path_to_train.txt, "validation": path_to_validation.txt}

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-sub"

# Define training args
training_args = TrainingArguments(
    output_dir="./model_output/" + repository_id,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
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
    train_dataset=train_data,
    eval_dataset=test_data,
    # compute_metrics=compute_metrics,
)

trainer.train()
tokenizer.save_pretrained(repository_id)
trainer.save_model()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
