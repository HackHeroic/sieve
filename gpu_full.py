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


def block_idx_gpu(n, block, B, rng):
    """
    B block permutations as a (B, n) int32 device array, vectorised.

    The obvious Python loop over blocks costs 11.7 ms per batch, which was
    three times the counts kernel.  Reshaping into (nblocks, block) and
    permuting rows does the same work in numpy with no Python loop.
    """
    nfull = (n // block) * block
    base = np.arange(nfull, dtype=np.int32).reshape(-1, block)
    nb = base.shape[0]
    orders = np.argsort(rng.random((B, nb)), axis=1)          # B random block orders
    idx = base[orders].reshape(B, nfull)                       # vectorised gather
    if nfull < n:                                              # ragged tail stays put
        tail = np.tile(np.arange(nfull, n, dtype=np.int32), (B, 1))
        idx = np.concatenate([idx, tail], axis=1)
    return cp.asarray(np.ascontiguousarray(idx))
