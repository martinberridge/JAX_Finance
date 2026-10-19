from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from pv01_jax.arrays import CalibrationProblem
from pv01_jax.pricers import residuals


def newton_solve(
    f: Callable[[jax.Array], jax.Array],
    z0: jax.Array,
    tol: float = 1e-14,
    max_iter: int = 30,
) -> jax.Array:
    """Find a root of a vector function using JAX Newton iterations."""
    solution, _, _ = newton_solve_with_info(f, z0, tol, max_iter)
    return solution


def _tangent_solve(
    linearized_residual: Callable[[jax.Array], jax.Array],
    value: jax.Array,
) -> jax.Array:
    return jnp.linalg.solve(jax.jacfwd(linearized_residual)(value), value)


def calibrate(problem: CalibrationProblem, q: jax.Array) -> jax.Array:
    """Calibrate curve parameters using residuals and an implicit JAX root."""
    return _calibrate_with_solver(problem, q, newton_solve)


def _calibrate_with_solver(
    problem: CalibrationProblem,
    q: jax.Array,
    solve: Callable[..., jax.Array],
) -> jax.Array:
    """Calibrate with a supplied root solver, retaining implicit differentiation."""
    q = jnp.asarray(q)

    def f(z: jax.Array) -> jax.Array:
        return residuals(problem, z, q)

    return jax.lax.custom_root(
        f,
        jax.lax.stop_gradient(q),
        solve,
        _tangent_solve,
    )


def calibrate_checked(
    problem: CalibrationProblem,
    q: jax.Array,
    tol: float = 1e-12,
    max_iter: int = 30,
) -> jax.Array:
    """Calibrate with a convergence guard; use ``eqx.filter_jit`` to retain its message."""
    q = jnp.asarray(q)
    solve = lambda f, z0: newton_solve(f, z0, tol=tol, max_iter=max_iter)
    z = _calibrate_with_solver(problem, q, solve)
    max_residual = jnp.max(jnp.abs(residuals(problem, z, q)))
    return eqx.error_if(
        z,
        ~(max_residual <= tol),
        "calibration did not converge",
    )


def calibrate_with_info(
    problem: CalibrationProblem,
    q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calibrate curves and return iterations and maximum residual diagnostics."""
    q = jnp.asarray(q)

    def f(z: jax.Array) -> jax.Array:
        return residuals(problem, z, q)

    return newton_solve_with_info(f, jax.lax.stop_gradient(q))


def newton_solve_with_info(
    f: Callable[[jax.Array], jax.Array],
    z0: jax.Array,
    tol: float = 1e-14,
    max_iter: int = 30,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a Newton solution, iteration count, and final maximum residual."""
    z0 = jnp.asarray(z0)
    iteration0 = jnp.asarray(0, dtype=jnp.int32)

    def max_abs_residual(z: jax.Array) -> jax.Array:
        return jnp.max(jnp.abs(f(z)))

    def condition(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        z, iteration = state
        return (iteration < max_iter) & (max_abs_residual(z) > tol)

    def step(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        z, iteration = state
        residual = f(z)
        jacobian = jax.jacfwd(f)(z)
        next_z = z - jnp.linalg.solve(jacobian, residual)
        return next_z, iteration + 1

    solution, iterations = jax.lax.while_loop(
        condition,
        step,
        (z0, iteration0),
    )
    return solution, iterations, max_abs_residual(solution)


__all__ = [
    "newton_solve",
    "newton_solve_with_info",
    "calibrate",
    "calibrate_with_info",
    "calibrate_checked",
]
