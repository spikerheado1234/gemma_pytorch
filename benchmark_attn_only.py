from gemma.model import GemmaForCausalLM, GemmaAttention, precompute_freqs_cis
from gemma.config import get_config_for_2b_v2, AttentionType
from gemma.model import Embedding
from gemma import tokenizer
import torch
import time

def attn_only(tokenizer, config, 
              device, GemmaAttn : GemmaAttention, 
              tokens : torch.Tensor):
    is_str_prompt = isinstance(tokens, str)
    if is_str_prompt:
        prompts = [tokens]

    batch_size = len(prompts)
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len
    assert max_seq_len <= config.max_position_embeddings

    # build KV caches
    kv_caches = []
    for _ in range(config.num_hidden_layers):
        size = (batch_size, max_seq_len, config.num_key_value_heads,
                config.head_dim)
        dtype = config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=dtype, device=device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=device)
        kv_caches.append((k_cache, v_cache))

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
    prompt_mask_tensor = token_ids_tensor != tokenizer.pad_id
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                          dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    freqs_cis = precompute_freqs_cis(config.head_dim, max_seq_len * 2, theta=10000)
    freqs_cis = freqs_cis.index_select(0, input_positions_tensor)
    kv_write_indices = input_positions_tensor

    # [batch_size, input_len, hidden_size]
    embed = Embedding(config.vocab_size, config.hidden_size, config.quant)
    next_state = embed(tokens)
    torch.cuda.synchronize()
    start = time.time()
    # decode and ignore output.
    for i in range(max_seq_len - min_prompt_len):
        ## Call the forward function on Gemma Attention with Local Sliding Window.
        GemmaAttn(
            hidden_states=next_state,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_caches[0],
            mask=curr_mask_tensor,
        )
    torch.cuda.synchronize()
    end = time.time()
    print(f'decoding time: {end-start}')


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
    gemma_attn_only = GemmaAttention(
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.num_key_value_heads,
        attn_logit_softcapping=model_config.attn_logit_softcapping,
        query_pre_attn_scalar=model_config.query_pre_attn_scalar,
        head_dim=model_config.head_dim,
        quant=model_config.quant,
        attn_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=model_config.sliding_window_size,
        prompt_length=model_config.prompt_length,
        )
    ## Random input.
    GPU_ID : int = 0

    ## We call our attn_only function.
    attn_only(tknizer, model_config, GPU_ID, gemma_attn_only, prompt)