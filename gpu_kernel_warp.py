"""
Warp-per-rule counting kernel.

Why the original is slow
------------------------
`popcount_batch` assigns ONE THREAD per rule.  Thread i walks `rules[i*W + w]`,
so at a fixed w adjacent lanes are W*8 = 5,792 bytes apart: every load is a
32-way scatter.  Measured on an L4 it reaches 44.2 GB/s (14.7% of peak) and sits
5.3x off BOTH the memory roof (1.647 ms) and the POPC issue roof (1.636 ms).
Being equally far from both is the signature of an access-pattern problem
rather than a bandwidth or compute limit.

The fix
-------
One WARP per rule.  Lane l handles words l, l+32, l+64, ... so the 32 lanes of a
warp read 32 CONSECUTIVE uint64 = 256 bytes, which coalesces into 2 transactions
instead of 32.  Each lane keeps its own B accumulators, then the warp reduces
them with __shfl_down_sync.

Cost: 32x more threads and a log2(32)=5-step shuffle reduction per accumulator.
Both are cheap next to turning a 32-way scatter into a coalesced read.
"""
import numpy as np
import cupy as cp

SRC = r'''
extern "C" __global__
void popcount_warp(const unsigned long long* __restrict__ rules,
                   const unsigned long long* __restrict__ ys,
                   int* __restrict__ out,
                   const int R, const int W, const int B)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= R) return;

    const unsigned long long* rp = rules + (size_t)warp * W;

    int acc[32];
    #pragma unroll
    for (int b = 0; b < 32; ++b) acc[b] = 0;

    // lane l takes words l, l+32, l+64, ...  -> 32 consecutive words per step
    for (int w = lane; w < W; w += 32) {
        unsigned long long rv = rp[w];
        if (rv == 0ULL) continue;
        for (int b = 0; b < B; ++b)
            acc[b] += __popcll(rv & ys[(size_t)b * W + w]);
    }

    // reduce each accumulator across the warp
    for (int b = 0; b < B; ++b) {
        int v = acc[b];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) out[(size_t)b * R + warp] = v;
    }
}
'''
_mod = cp.RawModule(code=SRC, options=('-std=c++11',))
_kern = _mod.get_function('popcount_warp')


def counts_batch_warp(rules_gpu, ys_gpu, B, R, W, threads=256):
    """Same signature and same result as gpu_kernel.counts_batch."""
    out = cp.empty((B, R), dtype=cp.int32)
    warps_per_block = threads // 32
    blocks = (R + warps_per_block - 1) // warps_per_block
    _kern((blocks,), (threads,), (rules_gpu, ys_gpu, out, R, W, B))
    return out
