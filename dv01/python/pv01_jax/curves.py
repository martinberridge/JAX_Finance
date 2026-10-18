from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from pv01_jax.arrays import CalibrationProblem, CurveSpec
from pv01_jax.interp import INTERPOLATORS


class Curve(eqx.Module):
    spec: CurveSpec
    z: jax.Array = eqx.field(converter=jnp.asarray)

    def zero(self, t):
        return INTERPOLATORS[self.spec.interpolator](self.spec.x_nodes, self.z, t)

    def df(self, t):
        t = jnp.asarray(t)
        return jnp.where(t <= 1e-10, 1.0, jnp.exp(-t * self.zero(t)))

    def forward(self, s, e, tau):
        tau = jnp.asarray(tau)
        ratio = self.df(s) / self.df(e)
        fallback = self.zero(e)
        return jnp.where(tau <= 1e-10, fallback, (ratio - 1.0) / tau)

    def compounded(self, s, e):
        return self.df(s) / self.df(e) - 1.0


def curves_from(problem: CalibrationProblem, z: jax.Array) -> dict[str, Curve]:
    """Create named curves from the global parameter vector."""
    return {
        spec.name: Curve(spec=spec, z=z[spec.param_slice[0] : spec.param_slice[1]])
        for spec in problem.curves
    }


__all__ = ["Curve", "curves_from"]
