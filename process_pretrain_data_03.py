# 处理预训练数据，将数据转换为模型可接受的格式
# 01 版本直接截断了长的，短的全都 pad 到了最长的，不太好，
# 02 版本改进了，把句子合并成为最长输入，一次输入
import os

from datasets import load_dataset
from transformers import BertTokenizer

from util.common import prepare_args

model_args, = prepare_args()

model_id = model_args.model_name_or_path
dataset_id = "./texts/jiadian_zhihu.txt"
save_dataset_path = "pretrain_data"

# 文件里的内容格式是这样的：标题一行，内容一行，然后空一行，
# 读取并遍历这个文件的的每一行，将标题和内容连接起来作为一行，删除空行，然后保存到新的文件里
with open(dataset_id, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    with open(dataset_id + "_new", 'w', encoding='utf-8') as f2:
        for i in range(0, len(lines), 2):
            f2.write(lines[i].strip() + lines[i + 1].strip() + "\n")

dataset = load_dataset("text", data_files=dataset_id)
if 'test' not in dataset:
    dataset = dataset['train'].train_test_split(test_size=0.01)

tokenizer = BertTokenizer.from_pretrained(model_id)
remove_columns = list(dataset['train'].features)

max_model_length = tokenizer.model_max_length
print(f"Max input length: {max_model_length}")


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=remove_columns)
print(f"Keys of tokenized pretrain_data: {list(tokenized_dataset['train'].features)}")
# print(tokenized_dataset['train'][:4])
print(tokenizer.decode(tokenized_dataset["train"][0]["input_ids"]))  # 可以看出前后加了 [CLS] 和 [SEP]
print(tokenizer.decode(tokenized_dataset["train"][1]["input_ids"]))  # 可以看出前后加了 [CLS] 和 [SEP]
print(tokenizer.decode(tokenized_dataset["train"][2]["input_ids"]))  # 可以看出前后加了 [CLS] 和 [SEP]
print(tokenizer.decode(tokenized_dataset["train"][3]["input_ids"]))  # 可以看出前后加了 [CLS] 和 [SEP]
print(tokenizer.decode(tokenized_dataset["train"][4]["input_ids"]))  # 可以看出前后加了 [CLS] 和 [SEP]


# todo 这样 group 后可能结束符号在另一个句子里，那么训练的时候他知道上一个句子是哪一句吗？
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 我们删除了小的剩余部分，如果模型支持它，我们可以添加填充而不是此删除，您可以根据需要自定义此部分。
    if total_length >= max_model_length:
        total_length = (total_length // max_model_length) * max_model_length

    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_model_length] for i in range(0, total_length, max_model_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


group_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)
print(group_datasets)  # 这样可以看出来条数变少了
print(tokenizer.decode(group_datasets["train"][1]["input_ids"]))  # 可以看出一条句子里有多个开始结束符号

# 这种 save 方式，用的时候，用的时候直接 load_from_disk 就可以了
group_datasets["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
group_datasets["test"].save_to_disk(os.path.join(save_dataset_path, "test"))
