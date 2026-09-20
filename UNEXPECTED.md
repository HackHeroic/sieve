# Making it find UNEXPECTED patterns, not just significant ones

> **Re-run after audit.** The figures below were originally produced by
> `verify_unexpected.py`, which drew permutations from `gpu_full.block_idx_gpu`
> -- a generator that ignored the `strata` argument -- and cut with
> `np.quantile`. That script is archived and the generator is deleted.
>
> Everything here has now been **re-run on the shipped null**
> (`unexpected_clean.py`, canonical J=1000 month-stratified permutation,
> order-statistic cutoff) and every figure reproduced unchanged: 24/50 surviving
> FWER against 39/50 under the naive shuffle, lift-surprise correlation -0.073,
> top-200 overlap 1/200, and 1,111 reversals of which 0 are significant.
> Artifact: `unexpected_clean.json`.


## The research

The gap between "statistically significant" and "surprising" is a solved
problem with a name.

**De Bie's FORSIED framework** -- *Maximum entropy models and subjective
interestingness* (De Bie, 2010, 147 citations), extended in *Subjective
interestingness of subgraph patterns* (van Leeuwen et al., 2016, 34 cites),
*Subjectively Interesting Component Analysis* (Kang et al., KDD 2016, 20
cites), and Kontonasios et al. (2013) for real-valued data.

The idea: score a pattern by how much it departs from an **explicit background
model of what the analyst already expects**, not from a flat prior. Maximum
entropy gives the principled completion of "everything I already know".

**Exceptional Model Mining** -- Duivesteijn, Feelders & Knobbe, *Data Mining
and Knowledge Discovery* 2015, DOI `10.1007/s10618-015-0403-4`, 129 citations
(original: Leman et al. 2008, 119 cites). Find subgroups where the
*relationship between variables* differs from the global model, rather than
where a mean is elevated.

*(Note: our own SOTA-DISCOVERY.md cites the DOI as `...0436-8`. That is wrong;
the correct DOI is `...0403-4`.)*

## What we implemented

Our lift measures deviation from the marginal base rate. That rewards a rule
for restating its own parts: if A raises the error rate and B raises the error
rate, nobody is surprised that A AND B does.

Background model: the analyst already knows every singleton effect. The
maximum-entropy completion consistent with those marginals is additive in
log-odds, so

> **surprise = logit(observed rate) - [ logit(p0) + SUM_i ( logit(p_i) - logit(p0) ) ]**

A rule with high surprise is doing something its own components do not
explain. That is an interaction, and it is unexpected by construction.

## Result 1: our old headline was the LEAST surprising thing in the data

| ranked by | mean surprise |
|---|---|
| top 200 by **lift** (what we were reporting) | **-0.130** |
| top 200 by **surprise** | **+0.363** |
| overlap between the two sets | **1 of 200** |
| correlation(lift, surprise) | **-0.074** |

Our best rules by lift had *negative* surprise -- they performed slightly
**worse** than their own parts predicted. "Yesterday's error was high AND the
forecast says today will swing" is two known facts stapled together.

Lift and surprise are essentially orthogonal. We had been ranking on the wrong
axis the entire time.

## Result 2: surprise without correction is dangerous, and we can prove it

Of the **50 most surprising** rules:

| | survive |
|---|---|
| block permutation FWER (correct) | **24 / 50** |
| naive row shuffle (what shipped tools do) | **39 / 50** |

**Half of the most surprising findings are noise.** This is exactly the failure
our research predicted for Ai2's Asta AutoDiscovery, which ranks findings by
"Bayesian surprise" with no multiplicity control: a fluke contradicts your
priors *by definition*, so surprise-ranking actively **rewards** spurious
results.

Ranking by surprise and correcting by FWER is the combination nobody ships.

## Result 3: the unexpected finding

We searched for **reversals** -- a factor that raises the error rate on its own
but is associated with a *lower* rate inside a particular combination. Simpson's
paradox shaped, and catnip for an analyst.

**1,111 candidates found. Some dramatic:** high coal share alone gives 11.5%
error against a 10.2% base, but inside one combination the error is 4.2% with it
versus 13.0% without -- a 68% drop.

**Zero of the 1,111 survive correction.**

> *Every single dramatic reversal in this dataset is noise.*

That is the unexpected pattern, it is about the data, and it is exactly what the
tool exists to show.

**Method note:** our first attempt reported 0/1,111 using the RIGHT tail, which
tests for excess. A reversal is depletion, so that null was guaranteed for
mechanical reasons. We added a left-tail test (verified against
`scipy.hypergeom.logcdf` to 4.8e-12) and re-ran two-sided. The answer held --
but it only counts because it was tested correctly.

## Result 4: a direction nobody looked at

Under the two-sided test, **65 rules are significantly DEPLETED** -- conditions
under which forecasts are reliably *better* than chance. We had only ever tested
for excess error. Untouched, and operationally the more useful half: it says
when you can trust the forecast.

## What to add to the pitch

1. Rank by **surprise**, not lift, citing De Bie 2010.
2. Show that **half of the most surprising rules are noise**, and that the
   shipped alternative would have passed 39 of 50.
3. Lead the finding with **"1,111 dramatic reversals, zero real."**
4. Report the **65 depletion rules** as the actionable half.

## Loose end

`surprise.py:38` warns on a predicate with zero true rows (empty slice -> NaN).
Harmless to the results above but should be guarded.
