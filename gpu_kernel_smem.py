"""
Shared-memory staged counting kernel.

The problem the warp kernel still has
-------------------------------------
Both earlier kernels read the permuted label matrix `ys` straight from global
memory inside the inner loop.  `ys` is only B*W*8 = 185 KB, but EVERY warp reads
all of it: with R = 66,844 rules that is 66,844 x 185 KB = 12.4 GB of requests
for 185 KB of data.  L2 absorbs most of it (the L4 has 48 MB) but L2 bandwidth
then becomes the limit, which is why the warp kernel stalls at 31% of roof.

The fix
-------
Tile over words.  For each 32-word chunk, the whole BLOCK cooperatively stages
ys[:, chunk] into shared memory, then every warp in the block reads it from
there.  With 8 warps per block that cuts ys traffic by 8x.

Shared memory cost: B * 32 * 8 = 8 KB per block at B = 32, against 100 KB/SM on
Ada, so occupancy is untouched.

Care: no thread may return early, because every thread has to reach both
__syncthreads() calls in the tile loop.  Out-of-range warps participate in the
staging and then discard their result.
"""
import numpy as np
import cupy as cp

SRC = r'''
extern "C" __global__
void popcount_smem(const unsigned long long* __restrict__ rules,
                   const unsigned long long* __restrict__ ys,
                   int* __restrict__ out,
                   const int R, const int W, const int B)
{
    extern __shared__ unsigned long long sh[];      // B * 32 words

    const int lane   = threadIdx.x & 31;
    const int wid    = threadIdx.x >> 5;
    const int nwarps = blockDim.x  >> 5;
    const int rule   = blockIdx.x * nwarps + wid;
    const int live   = (rule < R);

    int acc[32];
    #pragma unroll
    for (int b = 0; b < 32; ++b) acc[b] = 0;

    // clamp so an out-of-range warp still reads something legal
    const unsigned long long* rp = rules + (size_t)(live ? rule : 0) * W;

    for (int w0 = 0; w0 < W; w0 += 32) {
        const int nw = min(32, W - w0);

        // whole block stages ys[:, w0 .. w0+nw-1]
        for (int i = threadIdx.x; i < B * nw; i += blockDim.x) {
            const int b = i / nw, j = i - b * nw;
            sh[b * 32 + j] = ys[(size_t)b * W + w0 + j];
        }
        __syncthreads();

        if (live && lane < nw) {
            const unsigned long long rv = rp[w0 + lane];
            if (rv != 0ULL) {
                for (int b = 0; b < B; ++b)
                    acc[b] += __popcll(rv & sh[b * 32 + lane]);
            }
        }
        __syncthreads();
    }

    for (int b = 0; b < B; ++b) {
        int v = acc[b];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0 && live) out[(size_t)b * R + rule] = v;
    }
}
'''
_mod = cp.RawModule(code=SRC, options=('-std=c++11',))
_kern = _mod.get_function('popcount_smem')


def counts_batch_smem(rules_gpu, ys_gpu, B, R, W, threads=256):
    """Same signature and same result as gpu_kernel.counts_batch."""
    out = cp.empty((B, R), dtype=cp.int32)
    nwarps = threads // 32
    blocks = (R + nwarps - 1) // nwarps
    shmem = B * 32 * 8
    _kern((blocks,), (threads,), (rules_gpu, ys_gpu, out, R, W, B),
          shared_mem=shmem)
    return out
