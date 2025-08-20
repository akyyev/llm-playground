from models import loginToHuggingFace

from huggingface_hub import login
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc

QWEN2 = "Qwen/Qwen2-7B-Instruct"
LLAMA = "meta-llama/Llama-2-7b-chat"
GEMMA2 = "google/gemma-2-2b-it"


def main():
    print('Hi there!')
    loginToHuggingFace()

    messages = [
        # {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ]

    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_type="nf4"
    # )

    tokenizer = AutoTokenizer.from_pretrained(GEMMA2, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(prompt)
    model = AutoModelForCausalLM.from_pretrained(GEMMA2, device_map="auto") #quantization_config=quant_config
    
    memory = model.get_memory_footprint() / 1e6
    print(f"Memory footprint: {memory:,.1f} MB")

    print(model)
    

if __name__ == "__main__":
    main()
