| Settings suffix | Function | Compile time | Median / wall time |
|---|---|---:|---:|
| -linear-settings | market_quote_pv01 | 2.309737s | 0.000607s |
| -linear-settings | parallel_pv01 | 2.047798s | 0.000579s |
| -linear-settings | cross_gamma | 3.453166s | 0.000976s |
| -linear-settings | batch_market_quote_pv01 (100 swaps) | 2.484260s | 0.002503s |
| -linear-settings | bump_pv01 | — | 18.037223s |
| -dq-settings | market_quote_pv01 | 10.790254s | 0.001438s |
| -dq-settings | parallel_pv01 | 10.651603s | 0.001667s |
| -dq-settings | cross_gamma | 19.238008s | 0.002100s |
| -dq-settings | batch_market_quote_pv01 (100 swaps) | 12.947585s | 0.004352s |
| -dq-settings | bump_pv01 | — | 33.956717s |
| -ncs-settings | market_quote_pv01 | 4.189436s | 0.000757s |
| -ncs-settings | parallel_pv01 | 5.029355s | 0.000915s |
| -ncs-settings | cross_gamma | 8.385646s | 0.001587s |
| -ncs-settings | batch_market_quote_pv01 (100 swaps) | 6.721354s | 0.004178s |
| -ncs-settings | bump_pv01 | — | SKIPPED (pass --slow) |
