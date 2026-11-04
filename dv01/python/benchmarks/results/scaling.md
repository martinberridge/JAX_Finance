Synthetic two-curve problem, Linear interpolation. Median of 20 runs after
compilation, in milliseconds; compile time (first call) in seconds.

| N quotes | AD reverse (ms) | AD jvp parallel (ms) | Bump vmap (ms) | Bump lax.map (ms) | Bump vmap / AD | Bump lax.map / AD | Compile AD / bump vmap (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.322 | 0.276 | 0.6 | 2.7 | 2 | 8 | 1.6 / 1.2 |
| 20 | 0.506 | 0.342 | 3.7 | 6.8 | 7 | 13 | 1.3 / 1.3 |
| 40 | 0.623 | 0.635 | 20.9 | 25.3 | 33 | 41 | 1.3 / 1.3 |
| 80 | 2.255 | 2.207 | 156.6 | 256.6 | 69 | 114 | 1.3 / 1.4 |
| 160 | 7.321 | 7.070 | 2301.6 | 1772.8 | 314 | 242 | 1.6 / 3.7 |
