import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    batch, seq_len, d_k = Q.shape

    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    s = scores - np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(s)
    attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    if np.isnan(attn).any():
        attn = np.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.matmul(attn, V)
    return out, attn
