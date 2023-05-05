from transformers import GPT2LMHeadModel, BertTokenizer
from transformers import TextGenerationPipeline
import gradio as gr

from util.common import prepare_args

model_args, = prepare_args()

model_id = model_args.model_name_or_path

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id, pad_token_id=tokenizer.eos_token_id).cuda()


def predict(input1):
    text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    res = text_generator(input1, max_length=400, do_sample=True)
    return res[0]['generated_text']


examples = [
    ["电动牙刷推荐"],
    ["电取暖器推荐"],
]

demo = gr.Interface(fn=predict,
                    inputs="text",
                    outputs="text",
                    title="GPT-2 训练演示",
                    description="使用 GPT-2 模型进行文本生成",
                    examples=examples,
                    )

demo.queue(concurrency_count=3).launch(server_name='0.0.0.0', share=False, inbrowser=False)
