# Superseded documents. Do not hand these out.

Every file here predates two bug fixes and contains numbers that contradict the
current results.

| doc | what is stale in it |
|---|---|
| RESULTS.md | calibration 0.067; 75,729 rules; 29,199 / 49,533 significant |
| VERDICT.md | calibration 0.067; holdout 19/20 and lift 4.38 -> 5.10; 151x; "tested on 2023-2026" |
| SCORECARD.md | holdout 19/20 and lift 4.38 -> 5.10; 58x; "tested on 2023 to 2026" |
| GPU-RESULTS.md | headlines 151x (kernel only, not end to end); repeats the incorrect "the CPU is memory bandwidth bound" argument |

Current equivalents: `FINAL.md` (numbers), `CONCEPTS.md` (concepts and defences),
`WALKTHROUGH.md` (worked example), `POSITIONING.md` (prior art),
`CROSS-DATASET.md` (generalisation).
