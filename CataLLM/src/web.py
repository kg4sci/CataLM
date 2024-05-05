import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import numpy as np
import pickle
from utils.utils import *

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model_base = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model_base,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model_base = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model_base,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model_base = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model_base,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model_base.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model_base.config.bos_token_id = 1
    model_base.config.eos_token_id = 2

    if not load_8bit:
        model.half()
        model_base.half()# seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        model_base = torch.compile(model_base)

    def evaluate(
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            use_context=False,
            **kwargs,
    ):
        instruction_base =  (
            "Please give a suitable catalyst material and control method based on the following information"
            # "answer CAN not be the same with reference"
        )
        with_context = '\nBelows are some reference:\n\n' + context_prompt(input)
        print(with_context)
        instruction = with_context + instruction_base


        prompt_base = prompter.generate_prompt(instruction_base, input)
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs_base = tokenizer(prompt_base, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        input_ids_base = inputs_base["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }


        with torch.no_grad():
            generation_output_base = model_base.generate(
                input_ids=input_ids_base,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s_base = generation_output_base.sequences[0]
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output_base = tokenizer.decode(s_base)
        return prompter.get_response(output), prompter.get_response(output_base)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Question", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            # gr.components.Checkbox(label="Adding reference into prompt"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=10,
                label="CataLLM Output",
            ),
            gr.inputs.Textbox(
                lines=10,
                label="Original LLM Output",
            )

        ],
        title="CataLLM",
        description="Using Large Language Models to Assist Catalyst Design",
        # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """

def context_prompt(question):
    """
    根据输入的标题和摘要，通过向量匹配的方式进行相似标注文献的提取，
    同时进行上上下文的填充
    :param input_title:
    :param input_abstract:
    :return:
    """
    input_element = question.split('\n')
    product, material_type, control_method_type = "", "", ""
    print("a", input_element)
    for one_element in input_element:
        element_list = one_element.split(':')
        if 'product' in element_list[0]:
            product = element_list[1]
        if 'Material' in element_list[0]:
            material_type = element_list[1]
        if 'Control' in element_list[0]:
            control_method_type = element_list[1]
    # embedding_data = pickle.load(open('../fine_tune_data/embedding', 'rb'))
    entity_data = pickle.load(open('../fine_tune_data/entity', 'rb'))
    # question_embedding = encode_with_small_model(question)
    context = context_search(product, material_type, control_method_type, entity_data)
    return context

if __name__ == "__main__":
    fire.Fire(main)
