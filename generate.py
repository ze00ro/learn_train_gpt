from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline

model_id = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
res = text_generator("这是很久之前的事情了", max_length=100, do_sample=True)

print(res)
