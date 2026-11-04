| Settings suffix | Function | Compile time | Median / wall time |
|---|---|---:|---:|
| -linear-settings | market_quote_pv01 | 2.390869s | 0.000700s |
| -linear-settings | parallel_pv01 | 2.028125s | 0.000575s |
| -linear-settings | bump_market_quote_pv01 (JAX, vmap) | 1.927257s | 0.003141s |
| -linear-settings | bump_market_quote_pv01_sequential (JAX, lax.map) | 1.835636s | 0.010548s |
| -linear-settings | cross_gamma | 3.898068s | 0.001040s |
| -linear-settings | batch_market_quote_pv01 (100 swaps) | 2.746938s | 0.005238s |
| -linear-settings | bump_pv01 (legacy NumPy) | — | 18.396577s |
| -dq-settings | market_quote_pv01 | 9.176353s | 0.001268s |
| -dq-settings | parallel_pv01 | 10.886789s | 0.001108s |
| -dq-settings | bump_market_quote_pv01 (JAX, vmap) | 10.890671s | 0.005032s |
| -dq-settings | bump_market_quote_pv01_sequential (JAX, lax.map) | 9.785157s | 0.012887s |
| -dq-settings | cross_gamma | 19.405426s | 0.002273s |
| -dq-settings | batch_market_quote_pv01 (100 swaps) | 13.924060s | 0.006646s |
| -dq-settings | bump_pv01 (legacy NumPy) | — | 36.557963s |
| -ncs-settings | market_quote_pv01 | 4.309597s | 0.000797s |
| -ncs-settings | parallel_pv01 | 4.893027s | 0.000782s |
| -ncs-settings | bump_market_quote_pv01 (JAX, vmap) | 4.836421s | 0.004359s |
| -ncs-settings | bump_market_quote_pv01_sequential (JAX, lax.map) | 4.626146s | 0.015596s |
| -ncs-settings | cross_gamma | 9.091698s | 0.001459s |
| -ncs-settings | batch_market_quote_pv01 (100 swaps) | 6.105105s | 0.005155s |
| -ncs-settings | bump_pv01 (legacy NumPy) | — | SKIPPED (pass --slow) |
