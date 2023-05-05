from transformers import GPT2LMHeadModel, BertTokenizer
import gradio as gr

# model = GPT2LMHeadModel.from_pretrained(name, pad_token_id=tokenizer.eos_token_id)

model_id = "/Volumes/backup/models/gpt2-chinese-cluecorpussmall"

tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)


def predict(inp):
    input_ids = tokenizer.encode(inp, return_tensors='pt')
    beam_output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."


demo = gr.Interface(fn=predict,
                    inputs="text",
                    outputs="text",
                    title="GPT-2 训练演示",
                    description="使用 GPT-2 模型进行文本生成",
                    )

demo.queue(concurrency_count=3).launch(server_name='0.0.0.0', share=True, inbrowser=False)
