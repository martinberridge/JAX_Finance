# PV01 benchmark before the refactor

Output of `pv01_jax/benchmark.py` from `target/benchmark.md` (2026-10-31), kept for the Step 5 comparison in `notes/refactor-plan.md`. Not regenerated.

| Settings suffix | Function | Compile time | Median / wall time |
|---|---|---:|---:|
| -linear-settings | market_quote_pv01 | 2.392191s | 0.000692s |
| -linear-settings | parallel_pv01 | 2.036558s | 0.000675s |
| -linear-settings | cross_gamma | 3.478771s | 0.001056s |
| -linear-settings | batch_market_quote_pv01 (100 swaps) | 2.459185s | 0.002008s |
| -linear-settings | bump_pv01 | — | 18.034769s |
| -dq-settings | market_quote_pv01 | 9.057063s | 0.001502s |
| -dq-settings | parallel_pv01 | 10.637499s | 0.001231s |
| -dq-settings | cross_gamma | 19.503108s | 0.002344s |
| -dq-settings | batch_market_quote_pv01 (100 swaps) | 12.668318s | 0.006251s |
| -dq-settings | bump_pv01 | — | 35.358650s |
| -ncs-settings | market_quote_pv01 | 3.911806s | 0.000759s |
| -ncs-settings | parallel_pv01 | 4.863210s | 0.000770s |
| -ncs-settings | cross_gamma | 8.266554s | 0.001918s |
| -ncs-settings | batch_market_quote_pv01 (100 swaps) | 5.609272s | 0.002802s |
| -ncs-settings | bump_pv01 | — | SKIPPED (pass --slow) |
