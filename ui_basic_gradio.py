from transformers import GPT2LMHeadModel, BertTokenizer
from transformers import TextGenerationPipeline
from transformers import GenerationConfig
import gradio as gr

from util.common import prepare_args

model_args, = prepare_args()

model_paths = {
    "story": "/llm/projects/jupyter/learn-train-gpt/model_output_clm/checkpoint-9356",
    "seller": "/llm/projects/learn_train_gpt/model_output_clm",
}

models = {}
tokenizers = {}
for k, v in model_paths:
    models[k] = GPT2LMHeadModel.from_pretrained(v).cuda()
    tokenizers[k] = BertTokenizer.from_pretrained(v)

print(tokenizers)


def predict_by_pipe(_model, _input, temperature):
    real_model = models[_model]
    tokenizer = tokenizers[_model]

    text_generator = TextGenerationPipeline(model=real_model, tokenizer=tokenizer, device=0)
    res = text_generator(_input, max_length=400, do_sample=True, temperature=temperature)
    return res[0]['generated_text']


def predict_by_model(_model, input1, temperature):
    real_model = models[_model]
    tokenizer = tokenizers[_model]

    generation_config = GenerationConfig(
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=0,
        eos_token_id=real_model.config.eos_token_id,
        pad_token=real_model.config.pad_token_id,
    )

    inputs = tokenizer(input1, return_tensors="pt")
    outputs = real_model.generate(**inputs, generation_config=generation_config, early_stopping=True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


examples = [
    ["电动牙刷有什么好的"],
    ["2023年燃气热水器怎么选"],
    ["令狐冲在珠穆朗玛峰上"],
]

demo = gr.Interface(fn=predict_by_pipe,
                    inputs=[
                        gr.Dropdown(["story", "seller"], label="模型选择", value="seller"),
                        "text",
                        gr.Slider(0.1, 1.0, step=0.1, value=0.5)
                    ],
                    outputs="text",
                    title="GPT-2 训练演示",
                    description="使用 GPT-2 模型进行文本生成",
                    examples=examples,
                    )

demo.queue(concurrency_count=3).launch(server_name='0.0.0.0', share=False, inbrowser=False)
