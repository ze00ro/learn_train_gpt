import os

from datasets import concatenate_datasets
from datasets import load_dataset
from transformers import BertTokenizer

model_id = "uer/gpt2-chinese-cluecorpussmall"
dataset_id = "liyucheng/chinese_metaphor_dataset" # 比喻句生成
save_dataset_path = "instruct_data"

# 定制指令提示格式
prompt_template = f"Summarize the following news article:\n{{input}}\nSummary:\n"

dataset = load_dataset(dataset_id)
print(dataset)
tokenizer = BertTokenizer.from_pretrained(model_id)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
print(f"Prompt length: {prompt_length}")
max_sample_length = tokenizer.model_max_length - prompt_length
print(f"Max input length: {max_sample_length}")

tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)
print(f"Max source length: {max_source_length}")

def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys: {list(tokenized_dataset['train'].features)}")

# save pretrain_data to disk
tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
tokenized_dataset["test"].save_to_disk(os.path.join(save_dataset_path, "test"))



