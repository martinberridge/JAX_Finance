Add tests/jax_port/test_phase1_jit_ordering.py (unittest), using resolve_calibration_problem on the
real config:</br>
(a) a jax.jit function of the CalibrationProblem traces once across two calls with the
    same problem (count calls inside the function), and traces again for a problem built
    from a different interpolator setting (static field changed);</br>
(b) jax.grad of sum over blocks of (sign * q[quote_idx]) with respect to q0 is nonzero
    exactly at the node quote indices, proving q enters as a traced input;</br>
(c) problem.quote_labels equals, in order, the labels of the regenerated
    tests/golden/*-mqs.csv (reuse _parse_mqs from
    tests/test_calibration_interpolation_example.py; it returns labels in file order);</br>
(d) the CurveSpec param_slices tile 0..len(q0) contiguously with no gaps, and the
    concatenated CurveSpec labels equal quote_labels.