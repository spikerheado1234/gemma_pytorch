from gemma.model import GemmaForCausalLM
from gemma.config import get_config_for_2b_v2
import torch
import time

if __name__ == '__main__':
    ## Test inference on gemma model.
    model_config = get_config_for_2b_v2()
    model_config.dtype = 'float32'
    print(model_config)
    gemma = GemmaForCausalLM(model_config)
    ## Random input.
    batch = 10
    seq_length = 7000
    inp = torch.randint(0, 100, (batch, seq_length))
    GPU_ID : int = 0
    output_len : int = 1
    prompt = ['s ' for _ in range(seq_length)]
    prompt = ''.join(prompt)
    gemma = gemma.to(GPU_ID).eval()
    result = gemma.generate(prompt, GPU_ID, output_len=1)
    print(f'result: {result}')

    a = time.time()
    result = gemma.generate(prompt, GPU_ID, output_len=1)
    b = time.time()


    print(f'finished! Time taken: {b-a:.5f}')
