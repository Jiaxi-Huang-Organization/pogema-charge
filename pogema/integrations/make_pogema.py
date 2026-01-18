from typing import Union

from gymnasium import Wrapper

from pogema import GridConfig
from pogema.envs import _make_pogema
from pogema.integrations.pettingzoo import parallel_env
from pogema.integrations.pymarl import PyMarlPogema
from pogema.integrations.sample_factory import AutoResetWrapper, IsMultiAgentWrapper, MetricsForwardingWrapper
from pogema.integrations.stable_baselines import SingleAgentWrapper


def _make_sample_factory_integration(grid_config):
    env = _make_pogema(grid_config)
    env = MetricsForwardingWrapper(env)
    env = IsMultiAgentWrapper(env)
    if grid_config.auto_reset is None or grid_config.auto_reset:
        env = AutoResetWrapper(env)
    return env


def _make_py_marl_integration(grid_config, *_, **__):
    return PyMarlPogema(grid_config)

def _make_stable_baselines_integration(grid_config):
    env = _make_pogema(grid_config)
    if grid_config.num_agents == 1:
        return SingleAgentWrapper(env)
    else:
        raise KeyError('Multi-agent integration is not supported for stable-baselines')


def make_pogema(grid_config: Union[GridConfig, dict] = GridConfig(), *args, **kwargs):
    if isinstance(grid_config, dict):
        grid_config = GridConfig(**grid_config)

    if grid_config.integration != 'SampleFactory' and grid_config.auto_reset:
        raise KeyError(f"{grid_config.integration} does not support auto_reset")

    if grid_config.integration is None:
        return _make_pogema(grid_config)
    elif grid_config.integration == 'SampleFactory':
        return _make_sample_factory_integration(grid_config)
    elif grid_config.integration == 'PyMARL':
        return _make_py_marl_integration(grid_config, *args, **kwargs)
    elif grid_config.integration == 'rllib':
        raise NotImplementedError('Please use PettingZoo integration for rllib')
    elif grid_config.integration == 'PettingZoo':
        return parallel_env(grid_config)
    elif grid_config.integration == 'gymnasium':
        return _make_stable_baselines_integration(grid_config)

    raise KeyError(grid_config.integration)


pogema_v0 = make_pogema
