# 处理预训练数据，将数据转换为模型可接受的格式
import os

from datasets import concatenate_datasets
from datasets import load_dataset
from transformers import BertTokenizer

model_id = "uer/gpt2-chinese-cluecorpussmall"
dataset_id = "liyucheng/chinese_metaphor_dataset"  # 比喻句生成
save_dataset_path = "pretrain_data"

dataset = load_dataset(dataset_id)
print(dataset)
tokenizer = BertTokenizer.from_pretrained(model_id)

max_model_length = tokenizer.model_max_length
print(f"Max input length: {max_model_length}")

delete_columes = list(dataset['train'].features)

tokenized_inputs = concatenate_datasets([dataset["train"], ]).map(
    lambda x: tokenizer(x["sent"], truncation=True), batched=True, remove_columns=delete_columes)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_model_length)
print(f"Max source length: {max_source_length}")


def preprocess_function(sample, padding="max_length"):
    inputs = [item for item in sample["sent"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=delete_columes)
print(f"Keys of tokenized pretrain_data: {list(tokenized_dataset['train'].features)}")

# save
tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))

