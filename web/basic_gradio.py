from transformers import GPT2LMHeadModel, BertTokenizer
from transformers import TextGenerationPipeline
import gradio as gr

from util.common import prepare_args

model_args, = prepare_args()

model_id = model_args.model_name_or_path

# model = GPT2LMHeadModel.from_pretrained(name, pad_token_id=tokenizer.eos_token_id)

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)


def predict(input):
    text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    res = text_generator(input, max_length=100, do_sample=True)
    return res


demo = gr.Interface(fn=predict,
                    inputs="text",
                    outputs="text",
                    title="GPT-2 训练演示",
                    description="使用 GPT-2 模型进行文本生成",
                    )

demo.queue(concurrency_count=3).launch(server_name='0.0.0.0', share=True, inbrowser=False)