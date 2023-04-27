# 模型输入限制的是转为token后数组的长度，而不是字符串的长度
# 这个脚本可以获取输入的最大长度，方便后续处理文本
from transformers import BertTokenizer

model_id = "uer/gpt2-chinese-cluecorpussmall"
prompt_template = f"Summarize the following news article:\n{{input}}\nSummary:\n"

tokenizer = BertTokenizer.from_pretrained(model_id)
print(tokenizer(prompt_template.format(input="")))

prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length

print(f"Model max length: {tokenizer.model_max_length}")
print(f"Prompt length: {prompt_length}")
print(f"Max input length: {max_sample_length}")

# Model max length: 1024
# Prompt length: 11
# Max input length: 1013
