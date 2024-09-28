from gemma.r_sddmm import rsddmm_launcher, rsddmm_preamble
from gemma.acsr_helpers import create_acsr, create_causal_windowed_mask
import torch
import time

if __name__ == '__main__':
    batch = 1
    seq_length = 4096
    num_heads = 8
    head_dim = 288
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    GPU_ID = 0
    out_dtype = torch.float32
    mask : list[list[int]] = create_causal_windowed_mask(seq_length, seq_length // 2)
    queries = torch.randn((batch, seq_length, num_heads, head_dim), dtype=out_dtype).to(GPU_ID)
    keys = torch.randn((batch, seq_length, head_dim, num_heads), dtype=out_dtype).to(GPU_ID)
    values = torch.randn((batch, seq_length, num_heads, head_dim), dtype=out_dtype).to(GPU_ID)

    for _ in range(5):
        attn = torch.matmul(queries, keys)

    torch.cuda.synchronize()
    torch_sddmm_start = time.time()
    attn = torch.matmul(queries, keys)
    torch.cuda.synchronize()
    torch_sddmm_end = time.time()

    print(f'torch output: {torch_sddmm_end-torch_sddmm_start}')

    dTos_linear_transformations, dTos_translations, \
    sTod_linear_transformations, sTod_translations, nnzs, \
    acsr_trailing_dimension, _, _ = create_acsr(
        mask, BLOCK_SIZE_X, GPU_ID
        )
    
    output_tensor, grid_dim, \
    tb_map_x, tb_map_y = rsddmm_preamble(mask, (batch, num_heads, seq_length, acsr_trailing_dimension), 
                                         BLOCK_SIZE_X, BLOCK_SIZE_Y, GPU_ID, out_dtype)

    ## Call the rsddmm launcher.
    for _ in range(5):
        rsddmm_output, sTod_linear_transformations, \
            sTod_translations, nnzs = rsddmm_launcher(queries, keys, output_tensor, 
                                                    dTos_linear_transformations, dTos_translations,
                                                    sTod_linear_transformations, sTod_translations,
                                                    acsr_trailing_dimension, nnzs, grid_dim, 
                                                    tb_map_x, tb_map_y, 
                                                    BLOCK_SIZE_Y, BLOCK_SIZE_X)

    
    rsddmm_start = time.time()
    rsddmm_output, sTod_linear_transformations, \
        sTod_translations, nnzs = rsddmm_launcher(queries, keys, output_tensor, 
                                                dTos_linear_transformations, dTos_translations,
                                                sTod_linear_transformations, sTod_translations,
                                                acsr_trailing_dimension, nnzs, grid_dim, 
                                                tb_map_x, tb_map_y, 
                                                BLOCK_SIZE_Y, BLOCK_SIZE_X)
    rsddmm_end = time.time()
    print(f'rsddmm timing: {rsddmm_end - rsddmm_start}')
