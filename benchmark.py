from gemma.model import GemmaModel
from gemma.config import get_config_for_2b_v2
import torch

if __name__ == '__main__':
    ## Test inference on gemma model.
    gemma = GemmaModel(get_config_for_2b_v2())

    ## Random input.
    batch = 10
    seq_length = 1024
    inp = torch.randint(0, 100, (batch, seq_length))

    gemma(inp)

    print('finished!')