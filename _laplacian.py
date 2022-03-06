from jaxfg.noises import NoiseModelBase

import jax_dataclasses as jdc
from overrides import overrides
from jax import numpy as jnp

from jaxfg import hints


@jdc.pytree_dataclass
class LaplacianWrapper(NoiseModelBase):
    """Wrapper for applying a Huber loss to standard (eg Gaussian) noise models.

    TODO(brentyi): this should be functional in terms of optimization, but is still an
    experimental state and has minor issues. Notably causes inaccuracy in the value
    returned by `StackedFactorGraph.compute_cost()`."""

    wrapped: NoiseModelBase
    """Underlying noise model."""

    eps: hints.Scalar = 1e-8
    """Term to add to avoid dividing by zero"""

    @overrides
    def get_residual_dim(self) -> int:
        return self.wrapped.get_residual_dim()
    
    @overrides
    def whiten_residual_vector(self, residual_vector: hints.Array) -> hints.Array:
        residual_vector = self.wrapped.whiten_residual_vector(residual_vector)
        residual_norm = jnp.linalg.norm(residual_vector)
        return residual_vector / (jnp.sqrt(residual_norm) + self.eps)

    @overrides
    def whiten_jacobian(
        self,
        jacobian: hints.Array,
        residual_vector: hints.Array,
    ) -> hints.Array:
        jacobian = self.wrapped.whiten_jacobian(jacobian, residual_vector)
        residual_norm = jnp.linalg.norm(residual_vector)
        return jacobian / (jnp.sqrt(residual_norm) + self.eps)
        