# 腾讯的预处理方式
import pickle

from transformers import BertTokenizer

from util.common import prepare_args
from utils.misc import count_lines

model_args, = prepare_args()

model_id = model_args.model_name_or_path
dataset_id = "./texts/jiadian_zhihu.txt"
save_dataset_path = "pretrain_data"


tokenizer = BertTokenizer.from_pretrained(model_id)

dataset_writer = open(f"./{save_dataset_path}/dataset-tmp-04.pt", "wb")
pos = 0
seq_length = 511
end = count_lines(dataset_id)
document = ''
with open(dataset_id, mode="r", encoding="utf-8") as f:
    while True:
        line = f.readline()
        pos += 1

        if line == '\n':
            # 读到空行，说明前面文档结束了，把前面几个句子合并成一个文档开始处理
            pass
        elif line == '':
            break
        else:
            document += line.strip("\n\"\'")
            continue

        tokenized_document = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(document))
        tokenized_document = [tokenizer.cls_token_id] + tokenized_document + [tokenizer.sep_token_id]

        diff_document = tokenizer(document)

        document = ''

        instances_num = len(tokenized_document) // (seq_length + 1)
        for i in range(instances_num):
            src = tokenized_document[i * (seq_length + 1): (i + 1) * (seq_length + 1)]
            seg_pos = [seq_length]
            src = (src, 0)
            pickle.dump((src, seg_pos), dataset_writer)

        src = tokenized_document[instances_num * (seq_length + 1):]
        if len(src) > 0:
            seg_pos = [len(src)]
            pad_num = seq_length + 1 - len(src)
            src = (src, pad_num)
            pickle.dump((src, seg_pos), dataset_writer)

        print(f"Processed {pos}/{end}")
        if pos >= end:
            break

dataset_writer.close()
