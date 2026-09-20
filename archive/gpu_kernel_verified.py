"""SIEVE permutation kernel on GPU.

The loop is memory bound: 387 MB of rule bitsets, streamed once per
permutation on CPU.  The win is not raw parallelism, it is REUSE -- hold the
rule matrix resident and AND it against a BATCH of permuted label vectors, so
the 387 MB is read once per batch rather than once per permutation.
"""
import time
import numpy as np
import cupy as cp

SRC = r'''
extern "C" __global__
void popcount_batch(const unsigned long long* __restrict__ rules,
                    const unsigned long long* __restrict__ ys,
                    int* __restrict__ out,
                    const int R, const int W, const int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per rule
    if (i >= R) return;

    const unsigned long long* rp = rules + (size_t)i * W;
    int acc[32];
    #pragma unroll
    for (int b = 0; b < 32; ++b) acc[b] = 0;

    for (int w = 0; w < W; ++w) {
        unsigned long long rv = rp[w];          // rule word, read ONCE
        if (rv == 0ULL) continue;               // sparse rules skip the batch
        for (int b = 0; b < B; ++b)
            acc[b] += __popcll(rv & ys[(size_t)b * W + w]);
    }
    for (int b = 0; b < B; ++b) out[(size_t)b * R + i] = acc[b];
}
'''
_mod = cp.RawModule(code=SRC, options=('-std=c++11',))
_kern = _mod.get_function('popcount_batch')


def counts_batch(rules_gpu, ys_gpu, B, R, W, threads=256):
    out = cp.empty((B, R), dtype=cp.int32)
    blocks = (R + threads - 1) // threads
    _kern((blocks,), (threads,), (rules_gpu, ys_gpu, out, R, W, B))
    return out


def pack_gpu(bools_np):
    """(B, n) bool on host -> (B, W) uint64 on device.

    Packed on CPU deliberately: the label batch is ~185 KB, so the transfer is
    free, and cupy.packbits has no axis argument.
    """
    b = np.ascontiguousarray(bools_np, dtype=bool)
    n = b.shape[-1]
    W = (n + 63) // 64
    pad = W * 64 - n
    if pad:
        b = np.concatenate([b, np.zeros(b.shape[:-1] + (pad,), dtype=bool)], -1)
    packed = np.packbits(b, axis=-1, bitorder='little').view(np.uint64)
    return cp.asarray(np.ascontiguousarray(packed.reshape(*b.shape[:-1], W)))


def block_perm_batch(n, block, B, rng):
    """B circular block permutations as an index matrix (B, n)."""
    starts = np.arange(0, n, block)
    nb = len(starts)
    idx = np.empty((B, n), dtype=np.int64)
    for j in range(B):
        order = rng.permutation(nb)
        idx[j] = np.concatenate([np.arange(starts[k], min(starts[k] + block, n))
                                 for k in order])
    return idx
