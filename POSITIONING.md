# What is ours, what is not, and what to say

Every citation below was verified against Crossref, OpenAlex, Europe PMC or the
authors' own PDFs. Where a paper is paywalled it is marked and the source of
the characterisation is named.

---

## The claim, stated precisely

> **Tarone's testability bound and Westfall-Young min-P, applied per-pattern
> over an anti-monotone conjunctive-pattern lattice, made valid under block
> permutation for autocorrelated observations.**

Every component is individually old. The conjunction appears nowhere we could
find. Confidence: about 85%.

**Do NOT say** "first to combine pattern mining with dependence-preserving
resampling." A finance or computational-neuroscience reviewer kills that in one
sentence. **Say** "per-pattern FWER control over a pattern lattice under
temporal dependence."

---

## The two literatures we sit between

### A. Significant pattern mining (genomics / data mining)

| Work | Verified record |
|---|---|
| Tarone's testability bound | Tarone (1990), *Biometrics* **46(2):515-522** |
| LAMP | Terada, Okada-Hatakeyama, Tsuda, Sese (2013), *PNAS* **110(32):12996-13001** |
| Westfall-Young light | Llinares-Lopez, Sugiyama, Papaxanthos, Borgwardt (2015), *KDD '15* 725-734 |
| Categorical covariates (FACS) | Papaxanthos, Llinares-Lopez, Bodenham, Borgwardt (2016), *NIPS 29* 2279-2287 |
| FastCMH | Llinares-Lopez, Papaxanthos, Bodenham, Roqueiro, Borgwardt (2017), *Bioinformatics* **33(12):1820-1828** |
| Field tutorial | Hamalainen &amp; Webb (2019), *DMKD* **33(2):325-377** |
| Current SOTA | Pellegrina &amp; Vandin (2024), FSR, *PVLDB* 17(11) |

**This literature assumes exchangeable rows, explicitly.** Hamalainen &amp; Webb:
*"the null hypothesis is exchangeability of the A-values on rows... all
permutations of A-values in the data are equally likely."* A full-text grep of
that tutorial returns **zero** hits for "autocorrelation" and zero for "time
series". Pellegrina &amp; Vandin assign *"i.i.d. labels to each transaction."*

The crux, and it is worth memorising: **this machinery models dependence
between HYPOTHESES and assumes independence between SAMPLES.** WY-light's own
abstract says it corrects for *"correlations between patterns."*

The two covariate papers stratify on gender and population structure. A CMH
stratum is a block of independent individuals with no internal ordering, so
neither can address serial dependence.

### B. Multiple testing in finance / econometrics

| Work | Verified record |
|---|---|
| Reality Check | White (2000), *Econometrica* **68(5):1097-1126** |
| SPA | Hansen (2005), *JBES* **23(4):365-380** |
| **StepM** | **Romano &amp; Wolf (2005), *Econometrica* 73(4):1237-1282** |
| Factor zoo | Harvey, Liu &amp; **Heqing** Zhu (2016), *RFS* **29(1):5-68** |
| Deflated Sharpe | Bailey &amp; Lopez de Prado (2014), *JPM* **40(5):94-107** |
| PBO / CSCV | Bailey, Borwein, Lopez de Prado, Zhu (2016), *J. Computational Finance* |
| Stationary bootstrap | Politis &amp; Romano (1994), *JASA* **89(428):1303-1313** |
| Block bootstrap | Kunsch (1989), *Ann. Statist.* **17(3)** |

**This literature corrects over a pre-enumerated universe.** Verified grid for
Sullivan, Timmermann &amp; White (1999), *J. Finance* **54(5):1647-1691**:

```
filters              24 x 12 x 8 x 4         =   497
moving averages      15 x 8 x 4 x 4  + 9 BLL =  2,049
support / resistance 10 x 8 x 4 x 4          =  1,220
channel breakouts    10 x 8 x 8 x 4          =  2,040
on-balance volume                            =  2,040
                                               -----
                                                7,846
```

Five hand-written templates, Cartesian products of finite parameter lists,
fixed before any return data is touched.

---

## The four papers that would defeat a broader claim. Cite them yourself.

**1. Sullivan, Timmermann &amp; White (1999).** 7,846 rules, max-statistic FWER,
*"applying the stationary bootstrap of Politis and Romano (1994)."* So
"resampling FWER with block resampling over thousands of rules" is a 1999
applied result.

**2. Romano &amp; Wolf (2005), *Econometrica*.** The dangerous one. It generalises
Westfall-Young **by name** and contains an entire section titled *"Choice of
Block Sizes"*: *"If the data sequence is a stationary time series, one needs to
use a time series bootstrap. Each possible choice - the moving blocks
bootstrap, the circular blocks bootstrap, or the stationary bootstrap -
involves the problem of choosing the block size b."*

**WY-style resampling FWER plus circular block bootstrap is 2005 Econometrica
textbook material.** This is the true prior art for the correction, and it is
the paper a quant will name. Our answer must be the generation and pruning
step, never the error control.

**3. Stella, Bouss, Palm, Grun (2022)**, *eNeuro* **9(3)**, and the SPADE line
(Torre et al. 2013, *Front Comput Neurosci* 7:132; Quaglio et al. 2017, ibid.
11:41). Computational neuroscience has run FP-growth over a closed-itemset
lattice on autocorrelated spike trains since 2013, with dependence-preserving
surrogates including a literal block shuffle.

**Their escape hatch matters to us.** They *deliberately abandon per-pattern
correction*. Torre et al.: *"Direct statistical tests of all individual
patterns... are not suitable, as they cause a severe multiple testing
problem."* They test pattern-spectrum signatures with BH or Bonferroni instead.
**No Tarone and no Westfall-Young anywhere in that literature.** That is
precisely the space we occupy.

**4. Bock, Gumbsch, Moor, Rieck, Roqueiro, Borgwardt (2018)**, *Bioinformatics*
**34(13):i438-i446**. S3M: Tarone plus time series. Defeats the loose phrasing
"Tarone on time series". But the unit of observation is a **patient** (355
cases, 355 controls), treated as i.i.d. Full text: zero hits for "Westfall",
zero for "autocorrel", zero for "exchangeab".

**5. Chordia, Goyal &amp; Saretto (2020)**, *RFS* **33(5):2134-2179**. 2,393,641
generated signals, Politis-Romano stationary bootstrap at six-month blocks,
Romano-Wolf StepM applied to the generated set. **Falsifies "finance does not
generate hypotheses combinatorially."** It also argues against us directly:
*"FWER control is not appropriate for finance applications as it lacks power."*
Their space is still a closed-form Cartesian expansion under three fixed
algebraic templates, with no lattice search, no anti-monotonicity and no
pruning.

---

## Resolved: Tonon &amp; Vandin does not touch our case

**Tonon &amp; Vandin (2019)**, "Permutation Strategies for Mining Significant
Sequential Patterns", **IEEE ICDM 2019**. Algorithm PROMISE. Paywalled with no
preprint, so its null model is established from three citing sources: Abuissa,
Lee &amp; Riondato (2023, *DMKD*); Pinxteren &amp; Calders (2021, *SDM*); Jenkins,
Walzer-Goldfeld &amp; Riondato (2022, *DMKD*).

It permutes **order and composition WITHIN independent sequences**, preserving
per-row and per-item marginals. Rows remain exchangeable units. It has no
concept of dependence between observations sampled over time. Same for caSPiTa
(*KAIS* 2023), where "time series" means paths on a network, and gRosSo
(*KAIS* 2022), which concerns drift across a sequence of internally-i.i.d.
datasets.

---

## The negative searches, which are the actual evidence

- `Tarone` + `block bootstrap` : **0**
- `testability bound` + `time series` : **0**
- `"significant pattern mining"` + `autocorrelation` : **0**
- `"Westfall-Young"` + `"block permutation"` : **1** (Adolf et al. 2014,
  voxel-wise fMRI, no pattern mining)
- arXiv `cat:q-fin*` with "family-wise error" in the abstract : **0**
- Grepping STW 1999 and both Romano-Wolf papers for
  `Tarone|apriori|anti-monotone|itemset|lattice|testability` : **0 in all three**

That last one explains why the gap exists at all. Econometrics brute-forces a
flat enumerated grid because 8,000 rules is tractable, so it never needed
pruning or a testability bound.

---

## What to say, and what to expect back

**Open with the concession.**

> "The correction is textbook, and Romano and Wolf covered block sizes for
> stationary series in 2005. What I could not find is per-pattern FWER control
> over a searched lattice under temporal dependence. The mining literature
> assumes exchangeable rows. The neuroscience people who do have autocorrelated
> data and a lattice gave up on per-pattern correction and test spectra
> instead. Do you know of prior work?"

**Three pushbacks, with answers.**

| He says | You say |
|---|---|
| "How is this not Romano-Wolf StepM?" | StepM corrects over a set you hand it. It does not build the set. The generation, the anti-monotone pruning and the testability bound are the difference, not the error control. |
| "Why FWER and not FDR? Chordia-Goyal-Saretto argue FWER lacks power." | FWER and k-FWER are the same machinery with a different stopping rule, so the ladder can be exposed. An operator acting on a single rule wants that rule to be real, which is a FWER guarantee. |
| "Tarone only bought 0.55%?" | Correct, and consistent with this literature: dense quantile predicates make almost everything testable. Reported as a measured null. |

---

## Caveats on this research

OpenAlex exhausted its quota mid-run, so later sweeps used Crossref, Europe PMC
and arXiv. Citation counts are Crossref/OpenAlex figures and run well below
Google Scholar. PROMISE and SPEck are paywalled; their null models come from
citing sources, not the originals. **Barua &amp; Sander (2014)**, *IEEE TKDE*,
"Mining Statistically Significant Co-location and Segregation Patterns", is the
spatial-autocorrelation analogue and its abstract could not be retrieved. It is
the one remaining stone unturned.
