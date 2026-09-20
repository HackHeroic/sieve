"""Keep the whole permutation loop on device.

Before: GPU counted, then the CPU packed bitsets, generated the next batch of
block permutations in numpy, and ran the p-value scan.  The card sat idle ~78%
of the time waiting for numpy.

Now: block permutation, gather, bit-packing and the min-p reduction all run on
device.  The CPU only draws block orders, which is microseconds.
"""
import numpy as np, cupy as cp

PACK_SRC = r'''
extern "C" __global__
void gather_pack(const unsigned char* __restrict__ y,   // n
                 const int* __restrict__ idx,           // B x n
                 unsigned long long* __restrict__ out,  // B x W
                 const int n, const int W, const int B)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;      // word
    int b = blockIdx.y;                                 // permutation
    if (w >= W || b >= B) return;
    unsigned long long v = 0ULL;
    int base = w * 64;
    const int* ip = idx + (size_t)b * n;
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        int r = base + k;
        if (r < n && y[ip[r]]) v |= (1ULL << k);
    }
    out[(size_t)b * W + w] = v;
}
'''

# NOTE: a GPU min-p kernel used to live here.  It was removed deliberately.
#
#   1. It was SLOWER than the CPU scan by 14x (79 ms vs 0.2 ms per batch).  On
#      CPU the admissible bound is a single running global best, so it prunes
#      99.91% of rules.  On GPU each thread carries its own weak local bound and
#      prunes almost nothing, and the survivors then serialise through warp
#      divergence.
#   2. It carried the same hypergeometric support-boundary bug that was fixed in
#      fastp.py, and dead code with a latent divide-by-zero is a trap.
#
# The p-value scan belongs on the CPU.  The GPU does the counting.

_pack = cp.RawModule(code=PACK_SRC).get_function('gather_pack')


def gather_pack(y_gpu, idx_gpu, n, W, B):
    out = cp.empty((B, W), dtype=cp.uint64)
    t = 128
    _pack(((W + t - 1)//t, B), (t,), (y_gpu, idx_gpu, out, n, W, B))
    return out


# NOTE: block_idx_gpu was REMOVED.
#
# It generated its own block orders, vectorised, but it ignored the `strata`
# argument entirely and pinned the ragged tail.  core.block_permutation
# stratifies by calendar month and (since the fix) also pins the tail.  Two
# generators with different semantics behind one name meant that scripts
# importing this one were silently running a DIFFERENT null from the one every
# document describes -- which is exactly what happened to the holdout and the
# cross-BA replication.
#
# Every permutation now comes from core.block_permutation via backends.py, so
# CPU and GPU consume identical indices.  See backends.py for the measurement
# that made sharing affordable (4.3 ms/batch against a 5.6 ms kernel).
