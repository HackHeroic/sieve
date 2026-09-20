# L6 measured: the GPU rung

RunPod L4, 2026-09-18. Same lattice, same data, same statistic as the CPU run.

## The card is honest

| | |
|---|---|
| GPU | NVIDIA L4, 23 GB, driver 580.126.20 |
| Theoretical peak bandwidth | **300 GB/s** (matches NVIDIA spec) |
| Measured on a plain AND kernel | **235 GB/s = 78% of peak** |

Checked before benchmarking anything. A throttled or oversubscribed host would
have made every number after this worthless.

## Correctness

The GPU counts match the CPU popcount **exactly**, for both the first and the
last permutation in a batch. Not approximately: `np.array_equal`.

## The measurement

66,844 rules x 46,324 rows, 387 MB of rule bitsets, batch of 32 permutations.

| implementation | ms per permutation | speedup |
|---|---:|---:|
| CPU, 1 core (M-series) | 57.9 | 1x |
| CPU, 14 cores | 26.5 | 2.2x |
| **GPU L4, batched kernel** | **0.175** | **331x** |

**Headline against the honest baseline: 151x over a fully parallel CPU.**

J = 10,000 permutations: **4.4 minutes on 14 CPU cores, 1.8 seconds on one L4.**

## Three things worth saying out loud

**Fourteen cores only buy 2.2x over one core.** The CPU is memory bandwidth
bound, not compute bound. That is precisely why a GPU wins here, and it is a
better argument than any raw speedup number.

**The win is reuse, not parallelism.** The naive port reads the 387 MB rule
matrix once per permutation. Batching 32 permutations reads it once per batch
and does 32 popcounts per loaded word. That single change is most of the 151x.

**We are now compute bound, not bandwidth bound.** The kernel achieves 69 GB/s
of traffic, 23% of peak, because each loaded word feeds 32 popcounts. There is
headroom left, but the bottleneck has moved.

## The baseline we refused to use

The pod's own CPU measures **2,868 ms per permutation**, because RunPod
allocated 6 vCPU while `nproc` reports 48. Quoting that would have produced a
**16,000x** speedup.

It would also have been dishonest, and it is the exact trick last year's winner
used by benchmarking CUDA against a naive Python loop. The number above is
against the fastest CPU implementation we have, on the machine where it runs
fastest.

## Projection to the GX10

The GX10 (GB10) has **273 GB/s** against the L4's 300, so on a bandwidth bound
kernel it would be about 0.91x. But the batched kernel is **compute** bound,
and the GB10 is Blackwell with more compute than the L4's Ada. So expect the
GX10 to land at or slightly above these numbers. **UNVERIFIED until Saturday** --
do not put a GX10 figure on a slide before measuring it.

## The kernel

`gpu_kernel.py`, CuPy RawKernel, compiled through NVRTC so no CUDA toolkit is
needed. One thread per rule; each thread streams the rule's 724 words once and
accumulates 32 permutation counters in registers. That is the slide.
