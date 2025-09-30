from brax.training.distribution import ParametricDistribution, IdentityPostprocessor, _NormalDistribution
import jax.numpy as jnp
import jax
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
from brax.training import types
from flax import linen as nn
from brax.training.networks import FeedForwardNetwork, MLP, _get_obs_state_size, normalizer_select

from brax.training import distribution
from brax.training import networks
from brax.training import types
import jax
from brax.training.agents.ppo.networks import PPONetworks

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

class NormalSoftplusDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001, var_scale=1):
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
    return _NormalDistribution(loc=loc, scale=scale)
  
def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
) -> FeedForwardNetwork:
  """Creates a policy network."""
  if distribution_type == 'tanh_normal':
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )
  elif distribution_type == 'normal':
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  def apply(processor_params, policy_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))

  def init(key):
    policy_module_params = policy_module.init(key, dummy_obs)
    return policy_module_params

  return FeedForwardNetwork(init=init, apply=apply)



def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = nn.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
  value_kernel_init_kwargs = value_network_kernel_init_kwargs or {}

  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = NormalSoftplusDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
      distribution_type=distribution_type,
      noise_std_type=noise_std_type,
      init_noise_std=init_noise_std,
      state_dependent_std=state_dependent_std,
      kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )