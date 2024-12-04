from gemma.model import GemmaForCausalLM, GemmaDecoderLayer, precompute_freqs_cis
from gemma.config import get_config_for_2b_v2, get_config_for_2b_v2_attn_only, AttentionType
from gemma.model import Embedding
from gemma import tokenizer
import torch
import time
import pdb

def decoder_only(tokenizer, config, 
                 device, GemmaDecoder : GemmaDecoderLayer,
                 batch_size : int, seq_length: int,
                 output_len : int, cache_size : int):

    prompt_tokens = [[2 for _ in range(seq_length)] for _ in range(batch_size)]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len
    assert max_seq_len <= config.max_position_embeddings

    # build KV caches
    kv_caches = []
    for _ in range(config.num_hidden_layers):
        size = (batch_size, cache_size, config.num_key_value_heads,
                config.head_dim)
        dtype = config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=dtype, device=device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=device)
        kv_caches.append((k_cache, v_cache))
    ## We modify the kv_caches to be exportable to ONNX.

    # prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                  tokenizer.pad_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        tokenizer.pad_id,
                                        dtype=torch.int64)
    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, :len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    ## Generate the mask.
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                          dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, cache_size, cache_size),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    ## Generate sliding window mask.
    all_ones = torch.ones_like(mask_tensor)
    sliding_mask = torch.triu(
        all_ones, -1 * (seq_length // 2) + 1
    ) * torch.tril(all_ones)
    mask_tensor = torch.where(sliding_mask == 1, mask_tensor, -2.3819763e38)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    freqs_cis = precompute_freqs_cis(config.head_dim, max_seq_len * 2, theta=10000).to(GPU_ID)
    freqs_cis = freqs_cis.index_select(0, input_positions_tensor).to(GPU_ID)
    #kv_write_indices = input_positions_tensor
    kv_write_indices = torch.tensor([cache_size-1], dtype=torch.int64).to(GPU_ID)

    # [batch_size, input_len, hidden_size]
    embed = Embedding(config.vocab_size, config.hidden_size, config.quant).to(GPU_ID)
    next_state = embed(input_token_ids_tensor)
    torch.cuda.synchronize()
    # decode and ignore output.
    for i in range(max_seq_len - min_prompt_len):
        torch.cuda.synchronize()
        start = time.time()
        # Call a decoder layer. 
        GemmaDecoder(
            hidden_states=next_state,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_caches[0],
            mask=curr_mask_tensor,
        )
        torch.cuda.synchronize()
        end = time.time()
        print(f'decoding time: {end-start}')
    
        import io
        from onnxsim import simplify
        import onnx
        buffer = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(GemmaDecoder, (next_state,
                freqs_cis,
                kv_write_indices,
                kv_caches[0],
                curr_mask_tensor),
                buffer
            )
            buffer.seek(0, 0)

            onnx_model = onnx.load_model(buffer)
            onnx_model, success = simplify(onnx_model)
            assert success
            new_buffer = io.BytesIO()
            onnx.save(onnx_model, new_buffer)
            buffer = new_buffer
            buffer.seek(0, 0)

        if buffer.getbuffer().nbytes > 0:
            with open("temp.onnx", "wb") as f:
                f.write(buffer.read())
        break


if __name__ == '__main__':
    # Parameters.
    kv_cache_size = 4096
    batch_size = 16
    seq_length = 1
    output_len : int = 1

    ## Test inference on gemma model.
    model_config = get_config_for_2b_v2_attn_only()
    #model_config = get_config_for_2b_v2()
    ## Make 32 bit precision and turn of logit softcapping.
    tknizer = tokenizer.Tokenizer(model_config.tokenizer)
    #model_config.dtype = 'float32'
    model_config.quant = False
    model_config.dtype = 'bfloat16'
    model_config.attn_logit_softcapping = None
    print(model_config)
    GPU_ID : int = 0
    gemma_decoder = GemmaDecoderLayer(
            model_config
        ).to(GPU_ID)
    ## Random input.

    ## We call our attn_only function.
    decoder_only(tknizer, model_config, GPU_ID, gemma_decoder, batch_size, seq_length, output_len, kv_cache_size)
    decoder_only(tknizer, model_config, GPU_ID, gemma_decoder, batch_size, seq_length, output_len, kv_cache_size)
    decoder_only(tknizer, model_config, GPU_ID, gemma_decoder, batch_size, seq_length, output_len, kv_cache_size)
