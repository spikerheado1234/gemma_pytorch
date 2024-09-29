from gemma.model import GemmaForCausalLM
from gemma.config import get_config_for_2b_v2
from gemma import tokenizer
import torch
import time

if __name__ == '__main__':
    # Parameters.
    true_seq_length = 4096
    seq_length = true_seq_length - 10
    output_len : int = 1

    ## Test inference on gemma model.
    model_config = get_config_for_2b_v2()
    ## Make 32 bit precision and turn of logit softcapping.
    model_config.dtype = 'float32'
    model_config.attn_logit_softcapping = None
    tknizer = tokenizer.Tokenizer(model_config.tokenizer)
    prompt = ['s ' for _ in range(seq_length)]
    prompt = ''.join(prompt)
    prompt_length = len(tknizer.encode(prompt))
    model_config.prompt_length = prompt_length 
    print(model_config)
    gemma = GemmaForCausalLM(model_config)
    ## Random input.
    GPU_ID : int = 0
    gemma = gemma.to(GPU_ID).eval()
    result = gemma.generate(prompt, GPU_ID, output_len=1)
    print(f'result: {result}')

    result = gemma.generate(prompt, GPU_ID, output_len=1)
    result = gemma.generate(prompt, GPU_ID, output_len=1)

