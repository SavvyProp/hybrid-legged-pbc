# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""


from brax.training.distribution import ParametricDistribution, NormalDistribution
import jax.numpy as jnp
import jax

class IdentityPostprocessor:
  """Identity postprocessor."""

  def forward(self, x):
    return x

  def inverse(self, x):
    return x

  def forward_log_det_jacobian(self, x):
    return jnp.zeros_like(x)
  
class _NormalDistribution2:
  """Normal distribution."""

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

  def sample(self, seed):
    return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

  def mode(self):
    return self.loc

  def log_prob(self, x):
    log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
    log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
    return log_unnormalized - log_normalization

  def entropy(self):
    log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy * jnp.ones_like(self.loc)
  
class NormalDistribution2(ParametricDistribution):
  """Normal distribution."""

  def __init__(self, event_size: int) -> None:
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
    """
    super().__init__(
        param_size=event_size,
        postprocessor=IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )

  def create_dist(self, parameters):
    return _NormalDistribution2(*parameters)

class _NormalDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001, var_scale=1.0):
    """Initialize the distribution.
    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
      var_scale: adjust the gaussian's scale parameter.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size=2 * event_size,
        postprocessor=IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )
    self._min_std = min_std
    self._var_scale = var_scale

  def create_dist(self, parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
    return _NormalDistribution2(loc=loc, scale=scale)



class RawNormalDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001, var_scale=1.0):
    """Initialize the distribution.
    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
      var_scale: adjust the gaussian's scale parameter.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size=2 * event_size,
        postprocessor=IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )
    self._min_std = min_std
    self._var_scale = var_scale

  def create_dist(self, parameters):
    loc, log_scale = jnp.split(parameters, 2, axis=-1)
    log_scale = -5 + 0.5 * (jnp.tanh(log_scale) + 1) * (0.5 + 5)
    std = jnp.exp(log_scale).clip(min = 1e-3, max = None)
    #scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
    return _NormalDistribution2(loc=loc, scale=std)
