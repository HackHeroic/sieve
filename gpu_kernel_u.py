"""
Warp-per-rule with a COMPILE-TIME batch size, so the inner loop unrolls.

Why this should matter
----------------------
In every earlier kernel `B` is a runtime argument:

    for (int b = 0; b < B; ++b) acc[b] += __popcll(rv & ys[b*W + w]);

nvcc cannot unroll a loop whose trip count it does not know, and it cannot prove
`acc[b]` is a constant index, so the loop runs with full loop overhead and a
dependent address computation per iteration.  The warp kernel sits at 31% of the
POPC issue roof, which is the signature of issue-slot overhead rather than a
memory limit -- and the shared-memory experiment (6% SLOWER) already ruled out
label-matrix traffic.

Fix: bake B in as a literal at NVRTC compile time and `#pragma unroll` the loop.
One module is compiled and cached per distinct B.
"""
import cupy as cp

_SRC = r'''
extern "C" __global__
void popcount_u(const unsigned long long* __restrict__ rules,
                const unsigned long long* __restrict__ ys,
                int* __restrict__ out,
                const int R, const int W)
{
    const int BB = %d;                       // compile-time batch
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= R) return;

    const unsigned long long* rp = rules + (size_t)warp * W;

    int acc[BB];
    #pragma unroll
    for (int b = 0; b < BB; ++b) acc[b] = 0;

    for (int w = lane; w < W; w += 32) {
        const unsigned long long rv = rp[w];
        if (rv == 0ULL) continue;
        const unsigned long long* yp = ys + w;
        #pragma unroll
        for (int b = 0; b < BB; ++b)
            acc[b] += __popcll(rv & yp[(size_t)b * W]);
    }

    #pragma unroll
    for (int b = 0; b < BB; ++b) {
        int v = acc[b];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) out[(size_t)b * R + warp] = v;
    }
}
'''
_cache = {}


def _get(B):
    k = _cache.get(B)
    if k is None:
        mod = cp.RawModule(code=_SRC % B, options=('-std=c++11',))
        k = _cache[B] = mod.get_function('popcount_u')
    return k


def counts_batch_u(rules_gpu, ys_gpu, B, R, W, threads=256):
    """Same signature and same result as gpu_kernel.counts_batch."""
    out = cp.empty((B, R), dtype=cp.int32)
    kern = _get(B)
    warps = threads // 32
    kern(((R + warps - 1) // warps,), (threads,), (rules_gpu, ys_gpu, out, R, W))
    return out
