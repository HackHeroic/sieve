"""A/B the two kernels: identical output first, then speed."""
import time
import numpy as np, cupy as cp
from gpu_kernel import counts_batch, _kern as k_thread
from gpu_kernel_warp import counts_batch_warp, _kern as k_warp
from gpu_kernel_smem import counts_batch_smem, _kern as k_smem
from gpu_kernel_u import counts_batch_u, _get

R, W, B = 66844, 724, 32
rng = np.random.default_rng(0)
# realistic density: rules are sparse-ish, many all-zero words
dense = rng.random((R, W)) < 0.35
rl = np.where(dense, rng.integers(0, 2**63, (R, W), dtype=np.int64), 0).view(np.uint64)
rules = cp.asarray(np.ascontiguousarray(rl))
ys = cp.asarray(np.ascontiguousarray(rng.integers(0, 2**63, (B, W), dtype=np.int64).view(np.uint64)))

a = cp.asnumpy(counts_batch(rules, ys, B, R, W))
for nm, fn in (("warp", counts_batch_warp), ("smem", counts_batch_smem), ("unrolled", counts_batch_u)):
    c = cp.asnumpy(fn(rules, ys, B, R, W))
    ok = np.array_equal(a, c)
    print(f"CORRECTNESS {nm}: identical to baseline = {ok}")
    if not ok:
        d = np.abs(a.astype(np.int64) - c.astype(np.int64))
        print(f"  max diff {d.max()}, mismatched {int((d>0).sum()):,} of {d.size:,}")
        raise SystemExit(1)

for nm, k in (("thread", k_thread), ("warp", k_warp), ("smem", k_smem), ("unroll", _get(B))):
    print(f"  {nm:7s} regs {k.num_regs:3d}  local {k.local_size_bytes:4d}B  shared {k.shared_size_bytes:5d}B")

def bench(fn, n=30):
    for _ in range(5): fn(rules, ys, B, R, W)
    cp.cuda.Stream.null.synchronize()
    t = time.time()
    for _ in range(n): fn(rules, ys, B, R, W)
    cp.cuda.Stream.null.synchronize()
    return (time.time()-t)/n*1000

t1, t2, t3 = bench(counts_batch), bench(counts_batch_warp), bench(counts_batch_smem)
t4 = bench(counts_batch_u)
nbytes = R*W*8
roof = max(nbytes/235e9*1000, R*W*B*2/(58*16*2.04e9)*1000)
print(f"\n{'kernel':22s} {'ms/batch':>9} {'ms/perm':>9} {'GB/s':>8} {'% of roof':>10}")
for name, t in (("thread per rule", t1), ("warp per rule", t2), ("warp + shared mem", t3), ("warp + unrolled B", t4)):
    print(f"{name:22s} {t:9.3f} {t/B:9.4f} {nbytes/1e9/(t/1000):8.1f} {100*roof/t:9.1f}%")
print(f"\nwarp vs thread {t1/t2:.2f}x | smem vs warp {t2/t3:.2f}x | UNROLLED vs warp {t2/t4:.2f}x | unrolled vs thread {t1/t4:.2f}x")
