from transformers import GPT2LMHeadModel, BertTokenizer
from transformers import TextGenerationPipeline
from transformers import GenerationConfig
import gradio as gr

from util.common import prepare_args

model_args, = prepare_args()

model_id = model_args.model_name_or_path

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id).cuda()
print(tokenizer)


def predict_by_pipe(input1, temperature):
    text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
    res = text_generator(input1, max_length=400, do_sample=True, temperature=temperature)
    return res[0]['generated_text']


def predict_by_model(input1):
    generation_config = GenerationConfig(
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token=model.config.pad_token_id,
    )

    inputs = tokenizer(input1, return_tensors="pt")
    outputs = model.generate(**inputs, generation_config=generation_config, early_stopping=True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


examples = [
    ["电动牙刷有什么好的"],
    ["2023年燃气热水器怎么选"],
]

demo = gr.Interface(fn=predict_by_pipe,
                    inputs=["text", gr.Slider(0.0, 1.0, step=0.1, value=0.2)],
                    outputs="text",
                    title="GPT-2 训练演示",
                    description="使用 GPT-2 模型进行文本生成 4batch, 6epoch, 1e-3",
                    examples=examples,
                    )

demo.queue(concurrency_count=3).launch(server_name='0.0.0.0', share=False, inbrowser=False)
