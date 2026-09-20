"""Is the kernel actually well-optimised?  Measure, do not assume."""
import numpy as np, cupy as cp, time
from gpu_kernel import _kern, counts_batch

print("=== kernel resource usage ===")
print(f"  registers per thread : {_kern.num_regs}")
print(f"  local memory / thread: {_kern.local_size_bytes} bytes   "
      f"{'*** SPILLING -- acc[32] is in local memory ***' if _kern.local_size_bytes else 'no spill'}")
print(f"  shared memory / block: {_kern.shared_size_bytes} bytes")
print(f"  max threads per block: {_kern.max_threads_per_block}")

R, W, B = 66844, 724, 32
rules = cp.asarray(np.random.randint(0, 2**63, (R, W), dtype=np.int64).view(np.uint64))
ys = cp.asarray(np.random.randint(0, 2**63, (B, W), dtype=np.int64).view(np.uint64))

for _ in range(3):
    counts_batch(rules, ys, B, R, W)
cp.cuda.Stream.null.synchronize()
t = time.time()
N = 40
for _ in range(N):
    counts_batch(rules, ys, B, R, W)
cp.cuda.Stream.null.synchronize()
ms = (time.time() - t) / N * 1000

bytes_rules = R * W * 8
gbs = bytes_rules / 1e9 / (ms / 1000)
popc = R * W * B
print(f"\n=== measured, batch of {B} ===")
print(f"  kernel time          : {ms:.3f} ms/batch = {ms/B:.4f} ms/perm")
print(f"  rule matrix          : {bytes_rules/1e6:.0f} MB")
print(f"  achieved bandwidth   : {gbs:.1f} GB/s   ({100*gbs/300:.1f}% of L4 peak 300)")
print(f"  popcounts issued     : {popc/1e9:.3f} G")

# roofline: L4 = 58 SMs, 2.04 GHz boost, POPC 16/clk/SM, __popcll ~= 2 POPC
popc_roof_ms = popc * 2 / (58 * 16 * 2.04e9) * 1000
mem_roof_ms = bytes_rules / 235e9 * 1000     # 235 GB/s measured on a plain AND kernel
print(f"\n=== roofline ===")
print(f"  memory roof @235 GB/s : {mem_roof_ms:.3f} ms")
print(f"  POPC issue roof       : {popc_roof_ms:.3f} ms   <- binding" if popc_roof_ms > mem_roof_ms
      else f"  POPC issue roof       : {popc_roof_ms:.3f} ms")
binding = max(popc_roof_ms, mem_roof_ms)
print(f"  measured              : {ms:.3f} ms")
print(f"  EFFICIENCY vs binding roof: {100*binding/ms:.1f}%   (headroom {ms/binding:.2f}x)")
print(f"  well-tuned kernels land at 65-85%")
