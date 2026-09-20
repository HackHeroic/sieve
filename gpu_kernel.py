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


# NOTE: pack_gpu and block_perm_batch were REMOVED.
#
# block_perm_batch had the same defect as gpu_full.block_idx_gpu -- no strata --
# and additionally shuffled the ragged tail, so roughly 253 of 275 blocks landed
# in the wrong calendar month.  pack_gpu was superseded by gpu_full.gather_pack,
# which packs on device instead of round-tripping through the host.
#
# counts_batch below is the only thing this module still exports, and it is
# driven by backends.GPUBackend.
