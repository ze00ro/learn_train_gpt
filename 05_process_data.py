# 直接处理成每行一条的文本，问题和内容合并成一行即可
# 用hf官方的rum_clm.py试试

from util.common import prepare_args
from utils.misc import count_lines

dataset_id = "./texts/jiadian_zhihu.txt"
save_dataset_path = "05_data"

train_writer = open(f"./{save_dataset_path}/train.txt", "wb")
test_writer = open(f"./{save_dataset_path}/test.txt", "wb")
pos = 0
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

        if pos <= end * 0.95:
            train_writer.write(document.encode("utf-8"))
            train_writer.write("\n".encode("utf-8"))
        else:
            test_writer.write(document.encode("utf-8"))
            test_writer.write("\n".encode("utf-8"))
        document = ''

        print(f"Processed {pos}/{end}")
        if pos >= end:
            break

train_writer.close()
test_writer.close()
