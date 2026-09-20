# Against last year: "Parallelizing Multivariate Distributions"

Advay Goel and Videet Mehta, Voloridge challenge, HackMIT 2025.
Read from their own deck, not from memory.

---

## What they built

**Problem.** Hedge funds compare strategies by Value at Risk. That needs
sampling from a joint distribution over thousands of variables. Current
practice bootstraps and adds Gaussian noise to parameters, which alters the
internal joint distribution and produces non-robust backtests.

**Method.** Copulas, via Sklar's theorem, so the joint decomposes into marginal
CDFs plus a dependence function. Then CUDA throughout: fit a CDF per parameter
in parallel, fit several candidate distributions per parameter and select by log
likelihood and AIC, compute the empirical CDF concurrently, learn the copula
with GPU XGBoost, sample in parallel.

**Engineering.** Welford's streaming mean and variance, tiled into CUDA threads.
A tiled covariance kernel with shared memory. A Cholesky decomposition kernel.

**Results.**

| | |
|---|---|
| Multivariate Gaussian | 22.2 ms -> 0.0873 ms, **250x** |
| Multi-model distribution estimation | 480 ms -> 8.8 ms, **55x** |
| Validation | ground-truth sigma vs sample std with 95% CI, 10 dimensions |

Their optimisation ladder, in their own numbers:

```
Method 0  Naive Python       22.2   ms     1.00x
Method 1  Naive CUDA          4.71  ms     4.70x
Method 2  Shared Memory       0.169 ms   131.23x
Method 3  Sum of Squares      0.0883 ms  251.15x
Method 4  Welford             0.0873 ms  254.00x
Method 5  Welford + Warp      0.119 ms   186.77x
```

---

## Where they are genuinely stronger

**1. Domain fit.** VaR and backtesting *is* Voloridge's business. Day-ahead
electricity forecasting is not. Their judges did not have to be persuaded the
problem mattered.

**2. GPU sophistication.** Welford's streaming algorithm, tiling, shared memory,
Cholesky, warp primitives, and a six-rung ablation. Ours is one popcount kernel
with 32 register accumulators. On kernel engineering they are ahead, and it is
not close.

**3. Legibility.** "Funds need this, it is slow, we made it 250x faster" lands in
one sentence. Ours needs three.

**4. They reported a failure.** Method 5 is *slower* than Method 4, 0.119 against
0.0873, and they put it on the chart anyway. That is honest and it deserves
credit.

---

## Where we are stronger, and by how much

### The baseline

Their headline **250x is against Method 0, naive Python**. Their own table shows
naive CUDA at 4.71 ms, so the GPU-to-GPU optimisation is 4.71 -> 0.0873, about
**54x**. The other 4.7x is the Python-to-CUDA jump, which any port gets.

We made the opposite choice, deliberately and in writing. Our pod's own
throttled CPU would have supported a **16,000x** claim and we refused it. Our
figure is **122x end to end**, and `REPORT.md` section 8 states that our CPU
baseline is single-threaded unfused numpy at 6.8 GB/s, so against a tuned
implementation the win would be low double digits.

**They quote the flattering baseline. We quote the unflattering one and say why.**

### The validation

This is the real gap.

| | theirs | ours |
|---|---|---|
| correctness check | reconstructed sigma vs ground truth, 10 dims | GPU vs CPU **bit-identical**, plus 18 unit tests |
| false positive rate on data with no signal | not measured | **0.058** vs naive **1.000**, n = 120 |
| out-of-sample test | none | **20 / 20**, lift 4.47 -> 5.18 |
| independent replication | none | **57 / 58** across four other grids |
| confounder control | none | NOAA weather, **79-98%** retained |
| different dataset entirely | none | NOAA daily, different agency and cadence |
| sensitivity to design choices | none | full range reported, **7,991 to 26,005** |
| self-audit | none published | 24 defects confirmed and disclosed |

Their validation answers "does my sampler reproduce the marginal standard
deviations?" It is a sanity check on the marginals. It does not test the copula,
the dependence structure, or whether the resulting VaR is more accurate than the
bootstrap-plus-noise practice they set out to replace.

**Their claim is that it is faster. Ours is that it is correct, and we measured
correctness against a null where the right answer is known.**

### The kind of claim

Theirs is a **speed** claim about a known computation. Ours is a **guarantee**:
family-wise error control, with a measured false positive rate.

Speed claims are checkable and theirs check out. But a speed claim cannot be
wrong in an interesting way, and it does not tell a fund anything it did not
already believe. Ours can be wrong, which is why it needed 120 null datasets to
defend.

---

## The honest verdict

**On engineering, they win.** More sophisticated kernels, a cleaner ablation, a
more legible pitch, and a problem statement aimed squarely at the sponsor.

**On evidence, it is not close.** They have one validation chart. We have a
measured false positive rate, a temporal holdout, cross-grid replication, a
confounder control, a second dataset from a different agency, a sensitivity
sweep, and a published list of our own defects.

**The lesson we already took from them.** Their 250x is against naive Python.
Our documents say, in writing, that we will not do that, and the pod's 16,000x
number sits in the repository as the one we threw away. If a judge remembers one
contrast, that should be it.

**What we should not do.** Try to out-engineer them on GPU. We would lose, and
the GPU is not our contribution -- PBOOST (*Bioinformatics* 2014) and HWY (ACM
BCB 2015) got there a decade ago. Compete on the thing they did not do: showing
that the method is right.
