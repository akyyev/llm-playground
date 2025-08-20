
from dotenv import load_dotenv
import os

from huggingface_hub import login
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc

# instruct models
LLAMA = "meta-llama/Llama-2-7b-chat"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
GPT = "openai/gpt-oss-20b"

def main():
    loginToHuggingFace()

    encodeAndDecode(GPT, "I'm learning Artificial Intelligence!")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ]

    seeHowDataIsSentToModel(GPT, messages)

def loginToHuggingFace():
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(hf_token, add_to_git_credential=True)


def token_with_models(model, messages=[]):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", legacy=False).to("cpu")

    print(inputs)


def encodeAndDecode(model, text):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    print('Text: ', text)

    encoded = tokenizer.encode(text)
    print('Size: ', len(encoded))
    print('Text Tokenized: ', encoded)

    decoded = tokenizer.batch_decode(encoded)
    print('Text Decoded: ', decoded)

def seeHowDataIsSentToModel(model, messages):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print('--------------------')
    print('Prompt to the model: ', prompt)
    print('--------------------')


if __name__ == "__main__":
    main()