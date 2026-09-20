"""A/B the two kernels: identical output first, then speed."""
import time
import numpy as np, cupy as cp
from gpu_kernel import counts_batch, _kern as k_thread
from gpu_kernel_warp import counts_batch_warp, _kern as k_warp

R, W, B = 66844, 724, 32
rng = np.random.default_rng(0)
# realistic density: rules are sparse-ish, many all-zero words
dense = rng.random((R, W)) < 0.35
rl = np.where(dense, rng.integers(0, 2**63, (R, W), dtype=np.int64), 0).view(np.uint64)
rules = cp.asarray(np.ascontiguousarray(rl))
ys = cp.asarray(np.ascontiguousarray(rng.integers(0, 2**63, (B, W), dtype=np.int64).view(np.uint64)))

a = cp.asnumpy(counts_batch(rules, ys, B, R, W))
b = cp.asnumpy(counts_batch_warp(rules, ys, B, R, W))
same = np.array_equal(a, b)
print(f"CORRECTNESS: outputs identical = {same}")
if not same:
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    print(f"  max diff {d.max()}, mismatched {int((d>0).sum()):,} of {d.size:,}")
    raise SystemExit(1)

print(f"\nregisters/thread   thread-per-rule {k_thread.num_regs}   warp-per-rule {k_warp.num_regs}")
print(f"local mem/thread   thread-per-rule {k_thread.local_size_bytes}   warp-per-rule {k_warp.local_size_bytes}")

def bench(fn, n=30):
    for _ in range(5): fn(rules, ys, B, R, W)
    cp.cuda.Stream.null.synchronize()
    t = time.time()
    for _ in range(n): fn(rules, ys, B, R, W)
    cp.cuda.Stream.null.synchronize()
    return (time.time()-t)/n*1000

t1, t2 = bench(counts_batch), bench(counts_batch_warp)
nbytes = R*W*8
roof = max(nbytes/235e9*1000, R*W*B*2/(58*16*2.04e9)*1000)
print(f"\n{'kernel':22s} {'ms/batch':>9} {'ms/perm':>9} {'GB/s':>8} {'% of roof':>10}")
for name, t in (("thread per rule (old)", t1), ("warp per rule (new)", t2)):
    print(f"{name:22s} {t:9.3f} {t/B:9.4f} {nbytes/1e9/(t/1000):8.1f} {100*roof/t:9.1f}%")
print(f"\nSPEEDUP: {t1/t2:.2f}x")
