# Superseded scripts

Kept for provenance, not for running. Every script here imports a permutation
generator that has since been deleted (`gpu_full.block_idx_gpu` or
`gpu_kernel.block_perm_batch`), both of which ignored the `strata` argument, so
none of them runs the null that the documents describe.

Their live replacements:

| archived | replaced by |
|---|---|
| redo_holdout.py | `holdout_clean.py` |
| suite3.py (E6) | `replicate_rules.py` |
| reverify.py | `calib_gpu.py` + `replicate_gpu.py` |
| run2.py | `recanon.py` |
| find_unexpected.py, verify_unexpected.py | **not yet replaced** -- the surprise and reversal figures in UNEXPECTED.md were produced on the unstratified null and have not been re-run |
| gpu_kernel_verified.py | byte-identical copy of the pre-fix gpu_kernel.py |
