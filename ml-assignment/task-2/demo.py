import numpy as np
from attention import scaled_dot_product_attention

np.random.seed(7)

batch = 1
seq_len = 4
d_k = 3
d_v = 3

Q = np.random.rand(batch, seq_len, d_k)
K = np.random.rand(batch, seq_len, d_k)
V = np.random.rand(batch, seq_len, d_v)

mask = np.ones((batch, seq_len, seq_len), dtype=bool)

out, attn = scaled_dot_product_attention(Q, K, V, mask)

print("Q:\n", Q)
print("\nK:\n", K)
print("\nV:\n", V)
print("\nAttention Weights:\n", attn)
print("\nOutput:\n", out)
