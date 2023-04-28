# 处理预训练数据，将数据转换为模型可接受的格式
import os

from datasets import concatenate_datasets
from datasets import load_dataset
from transformers import BertTokenizer

model_id = "uer/gpt2-chinese-cluecorpussmall"
dataset_id = "liyucheng/chinese_metaphor_dataset"  # 比喻句生成
save_dataset_path = "pretrain_data"

dataset = load_dataset(dataset_id)
if 'test' not in dataset:
    dataset = dataset['train'].train_test_split(test_size=0.02)

print(dataset)

tokenizer = BertTokenizer.from_pretrained(model_id)
remove_columns = list(dataset['train'].features)

max_model_length = tokenizer.model_max_length
print(f"Max input length: {max_model_length}")

# 合并 train 和 test，找到最大的输入长度，后续都 padding 到了这个长度
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["sent"], truncation=True), batched=True, remove_columns=remove_columns)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_model_length)
print(f"Max source length: {max_source_length}")


def preprocess_function(sample, padding="max_length"):
    inputs = [item for item in sample["sent"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)
print(f"Keys of tokenized pretrain_data: {list(tokenized_dataset['train'].features)}")
print(tokenized_dataset)

# 这种 save 方式，用的时候，用的时候直接 load_from_disk 就可以了
tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
tokenized_dataset["test"].save_to_disk(os.path.join(save_dataset_path, "test"))
