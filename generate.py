from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline

from util.common import prepare_args

model_args, = prepare_args()

model_id = model_args.model_name_or_path

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
res = text_generator("春风像", max_length=100, do_sample=True)

print(res)
