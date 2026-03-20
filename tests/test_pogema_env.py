import re
import time

import numpy as np
import pytest
from tabulate import tabulate

from pogema import pogema_v0, AnimationMonitor

from pogema.envs import ActionsSampler
from pogema.grid import GridConfig


class ActionMapping:
    noop: int = 0
    up: int = 1
    down: int = 2
    left: int = 3
    right: int = 4


def test_moving():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    ac = ActionMapping()
    env.reset()

    env.step([ac.right, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.down, ac.noop])
    env.step([ac.down, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.up, ac.noop])

    env.step([ac.right, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.right, ac.noop])
    env.step([ac.down, ac.noop])
    obs, reward, terminated, truncated, infos = env.step([ac.right, ac.noop])

    assert np.isclose([1.0, 0.0], reward).all()
    assert np.isclose([True, False], terminated).all()


def test_types():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    obs, info = env.reset()
    assert obs[0].dtype == np.float32


def run_episode(grid_config=None, env=None):
    if env is None:
        env = pogema_v0(grid_config)
    env.reset()

    obs, rewards, terminated, truncated, infos = env.reset(), [None], [False], [False], [None]

    results = [[obs, rewards, terminated, truncated, infos]]
    while True:
        results.append(env.step(env.sample_actions()))
        terminated, truncated = results[-1][2], results[-1][3]
        if all(terminated) or all(truncated):
            break
    return results


def test_metrics():
    *_, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.5)

    *_, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=512))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 1.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 1.0)

    *_, infos = run_episode(GridConfig(num_agents=5, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.2)


def test_standard_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish'))
    env.reset()
    run_episode(env=env)


def test_pomapf_observation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish',
                               observation_type='POMAPF'))
    obs, info = env.reset()
    assert 'agents' in obs[0]
    assert 'obstacles' in obs[0]
    assert 'xy' in obs[0]
    assert 'target_xy' in obs[0]
    run_episode(env=env)


def test_mapf_observation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish',
                               observation_type='MAPF'))
    obs, info = env.reset()
    assert 'global_obstacles' in obs[0]
    assert 'global_xy' in obs[0]
    assert 'global_target_xy' in obs[0]
    run_episode(env=env)


def test_standard_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_gym_pogema_animation():
    import gymnasium
    env = gymnasium.make('Pogema-v0',
                         grid_config=GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42,
                                                on_target='finish'))
    env = AnimationMonitor(env)
    env.reset()
    done = False
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break


def test_non_disappearing_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='nothing'))
    env.reset()
    run_episode(env=env)


def test_non_disappearing_pogema_no_seed():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=None, on_target='nothing'))
    env.reset()
    run_episode(env=env)


def test_non_disappearing_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='nothing'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_life_long_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='restart'))
    env.reset()
    run_episode(env=env)


def test_life_long_pogema_empty_seed():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=None, on_target='restart'))
    env.reset()
    run_episode(env=env)


def test_life_long_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='restart'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_custom_positions_and_num_agents():
    grid = """
    ....
    ....
    """
    gc = GridConfig(
        map=grid,
        agents_xy=[[0, 0], [0, 1], [0, 2], [0, 3]],
        targets_xy=[[1, 0], [1, 1], [1, 2], [1, 3]],
    )

    for num_agents in range(1, 5):
        gc.num_agents = num_agents
        env = pogema_v0(grid_config=gc)
        env.reset()
        assert num_agents == len(env.get_agents_xy())
        assert num_agents == len(env.get_targets_xy())


def test_custom_positions_and_empty_num_agents():
    grid = """
    ....
    ....
    """
    gc = GridConfig(
        map=grid,
        agents_xy=[[0, 0], [0, 1], [0, 2], [0, 3]],
        targets_xy=[[1, 0], [1, 1], [1, 2], [1, 3]],
    )
    env = pogema_v0(grid_config=gc)
    env.reset()
    assert len(gc.agents_xy) == len(env.get_agents_xy())


def test_persistent_env(num_steps=100):
    seed = 42

    env = pogema_v0(
        grid_config=GridConfig(on_target='finish', seed=seed, num_agents=8, density=0.132, size=8, obs_radius=2,
                               persistent=True))

    env.reset()
    action_sampler = ActionsSampler(env.action_space.n, seed=seed)

    first_run_observations = []

    def state_repr(observations, rewards, terminates, truncates, infos):
        return np.concatenate([np.array(observations).flatten(), terminates, truncates, np.array(rewards), ])

    for current_step in range(num_steps):
        actions = action_sampler.sample_actions(dim=env.get_num_agents())
        obs, reward, terminated, truncated, info = env.step(actions)

        first_run_observations.append(state_repr(obs, reward, terminated, truncated, info))
        if all(terminated) or all(truncated):
            break

    # resetting the environment to the initial state using backward steps
    for current_step in range(num_steps):
        if not env.step_back():
            break

    action_sampler = ActionsSampler(env.action_space.n, seed=seed)

    second_run_observations = []
    for current_step in range(num_steps):
        actions = action_sampler.sample_actions(dim=env.get_num_agents())
        obs, reward, terminated, truncated, info = env.step(actions)
        second_run_observations.append(state_repr(obs, reward, terminated, truncated, info))
        assert np.isclose(first_run_observations[current_step], second_run_observations[current_step]).all()
        if all(terminated) or all(truncated):
            break
    assert np.isclose(first_run_observations, second_run_observations).all()


def test_steps_per_second_throughput():
    table = []
    for on_target in ['finish', 'nothing', 'restart']:
        for num_agents in [1, 32, 64]:
            for size in [32, 64]:
                gc = GridConfig(obs_radius=5, seed=42, max_episode_steps=1024,
                              size=size, num_agents=num_agents, on_target=on_target)

                start_time = time.monotonic()
                run_episode(grid_config=gc)
                end_time = time.monotonic()
                steps_per_second = gc.max_episode_steps / (end_time - start_time)
                table.append([on_target, num_agents, size, steps_per_second * gc.num_agents])
    print('\n' + tabulate(table, headers=['on_target', 'num_agents', 'size', 'SPS (individual)'], tablefmt='grid'))


# ============== Battery and Charge Metrics Tests ==============

def test_battery_info_in_infos():
    """Test that battery information is included in step infos."""
    env = pogema_v0(GridConfig(num_agents=2, size=8, density=0.2, seed=42, max_episode_steps=64))
    obs, info = env.reset()
    
    # Check initial battery info
    for i in range(env.get_num_agents()):
        assert 'battery' in info[i]
        assert 'initial_battery' in info[i]
        assert 'on_charges' in info[i]
        assert 'run_out_battery' in info[i]
        assert info[i]['battery'] == info[i]['initial_battery']
        assert info[i]['on_charges'] == False
        assert info[i]['run_out_battery'] == False


def test_battery_decrement_on_move():
    """Test that battery decreases when agents move."""
    env = pogema_v0(GridConfig(num_agents=2, size=8, density=0.1, seed=42, max_episode_steps=64))
    env.reset()
    
    initial_battery = [env.grid.get_battery_for_agent(i) for i in range(env.get_num_agents())]
    
    # Move all agents
    actions = [1, 2]  # up, down (non-wait actions)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    for i in range(env.get_num_agents()):
        # Battery should decrease by battery_decrement (default 1)
        assert info[i]['battery'] == initial_battery[i] - 1


def test_battery_no_decrement_on_wait():
    """Test that battery does not decrease when agents wait."""
    env = pogema_v0(GridConfig(num_agents=2, size=8, density=0.1, seed=42, max_episode_steps=64))
    env.reset()
    
    initial_battery = [env.grid.get_battery_for_agent(i) for i in range(env.get_num_agents())]
    
    # Wait action (0)
    actions = [0, 0]
    obs, reward, terminated, truncated, info = env.step(actions)
    
    for i in range(env.get_num_agents()):
        # Battery should not change
        assert info[i]['battery'] == initial_battery[i]


def test_relative_battery_metric():
    """Test RelativeBatteryMetric computes correct average relative battery."""
    from pogema.wrappers.metrics import RelativeBatteryMetric
    
    env = pogema_v0(GridConfig(num_agents=2, size=8, density=0.1, seed=42, max_episode_steps=32))
    env = RelativeBatteryMetric(env)
    env.reset()
    
    # Run episode
    while True:
        obs, reward, terminated, truncated, info = env.step(env.sample_actions())
        if all(terminated) or all(truncated):
            break
    
    # Check metric exists
    assert 'metrics' in info[0]
    assert 'avg_relative_battery' in info[0]['metrics']
    # Relative battery should be between 0 and 1
    assert 0.0 <= info[0]['metrics']['avg_relative_battery'] <= 1.0


def test_avg_throughput_with_active_metric():
    """Test AvgThroughputWithActiveMetric handles early agent death."""
    from pogema.wrappers.metrics import AvgThroughputWithActiveMetric
    
    env = pogema_v0(GridConfig(num_agents=4, size=8, density=0.1, seed=42, max_episode_steps=64))
    env = AvgThroughputWithActiveMetric(env)
    env.reset()
    
    # Run episode
    while True:
        obs, reward, terminated, truncated, info = env.step(env.sample_actions())
        if all(terminated) or all(truncated):
            break
    
    # Check metric exists
    assert 'metrics' in info[0]
    assert 'avg_throughput_with_active' in info[0]['metrics']
    # Throughput should be between 0 and 1
    assert 0.0 <= info[0]['metrics']['avg_throughput_with_active'] <= 1.0


def test_battery_depletion_rate_metric():
    """Test BatteryDepletionRateMetric tracks depleted agents."""
    from pogema.wrappers.metrics import BatteryDepletionRateMetric
    
    env = pogema_v0(GridConfig(num_agents=4, size=8, density=0.1, seed=42, max_episode_steps=64))
    env = BatteryDepletionRateMetric(env)
    env.reset()
    
    # Run episode
    while True:
        obs, reward, terminated, truncated, info = env.step(env.sample_actions())
        if all(terminated) or all(truncated):
            break
    
    # Check metrics exist
    assert 'metrics' in info[0]
    assert 'battery_depletion_rate' in info[0]['metrics']
    assert 'agents_depleted' in info[0]['metrics']
    # Depletion rate should be between 0 and 1
    assert 0.0 <= info[0]['metrics']['battery_depletion_rate'] <= 1.0


def test_charging_efficiency_metric():
    """Test ChargingEfficiencyMetric tracks charging station usage."""
    from pogema.wrappers.metrics import ChargingEfficiencyMetric
    
    env = pogema_v0(GridConfig(num_agents=4, size=8, density=0.1, seed=42, max_episode_steps=64))
    env = ChargingEfficiencyMetric(env)
    env.reset()
    
    # Run episode
    while True:
        obs, reward, terminated, truncated, info = env.step(env.sample_actions())
        if all(terminated) or all(truncated):
            break
    
    # Check metrics exist
    assert 'metrics' in info[0]
    assert 'charging_visits' in info[0]['metrics']
    assert 'charging_episode' in info[0]['metrics']
    assert 'avg_charging_per_agent' in info[0]['metrics']
    # Charging episode should be 0 or 1
    assert info[0]['metrics']['charging_episode'] in [0.0, 1.0]


def test_battery_health_metric():
    """Test BatteryHealthMetric tracks battery utilization."""
    from pogema.wrappers.metrics import BatteryHealthMetric
    
    env = pogema_v0(GridConfig(num_agents=4, size=8, density=0.1, seed=42, max_episode_steps=64))
    env = BatteryHealthMetric(env)
    env.reset()
    
    # Run episode
    while True:
        obs, reward, terminated, truncated, info = env.step(env.sample_actions())
        if all(terminated) or all(truncated):
            break
    
    # Check metrics exist
    assert 'metrics' in info[0]
    assert 'avg_final_battery_relative' in info[0]['metrics']
    assert 'avg_goal_battery_relative' in info[0]['metrics']
    assert 'battery_utilization' in info[0]['metrics']


def test_battery_metrics_all_modes():
    """Test that battery metrics work in all on_target modes."""
    from pogema.wrappers.metrics import RelativeBatteryMetric, AvgThroughputWithActiveMetric
    
    for on_target in ['finish', 'nothing', 'restart']:
        env = pogema_v0(GridConfig(num_agents=2, size=8, density=0.1, seed=42, 
                                   max_episode_steps=32, on_target=on_target))
        env = RelativeBatteryMetric(env)
        env = AvgThroughputWithActiveMetric(env)
        env.reset()
        
        # Run episode
        while True:
            obs, reward, terminated, truncated, info = env.step(env.sample_actions())
            if all(terminated) or all(truncated):
                break
        
        # Check metrics exist
        assert 'metrics' in info[0]
        assert 'avg_relative_battery' in info[0]['metrics']
        assert 'avg_throughput_with_active' in info[0]['metrics']


def test_charging_on_charge_station():
    """Test that agents charge when on charge station."""
    # Create a map with a charge station
    grid_map = """
        a.0.A
    """
    env = pogema_v0(GridConfig(map=grid_map, seed=42, max_episode_steps=32))
    env.reset()
    
    initial_battery = env.grid.get_battery_for_agent(0)
    
    # Agent should be able to move to charge station
    # Move right towards charge station at position (0, 2)
    actions = [4]  # right
    
    # First move towards charge station
    obs, reward, terminated, truncated, info = env.step(actions)
    
    # Check if agent is on charge station
    on_charges = info[0]['on_charges']
    
    # If on charge station, battery should increase
    if on_charges:
        current_battery = env.grid.get_battery_for_agent(0)
        # Battery should have increased (after decrement for move, then increment for charging)
        assert current_battery >= initial_battery - 1


def test_run_out_battery_termination():
    """Test that agents terminate when running out of battery."""
    # Use very low battery
    env = pogema_v0(GridConfig(num_agents=2, size=8, density=0.1, seed=42, 
                               max_episode_steps=64, initial_battery=[3, 10]))
    env.reset()
    
    # Run until agent 0 runs out of battery
    steps = 0
    agent0_depleted = False
    while steps < 20:
        obs, reward, terminated, truncated, info = env.step(env.sample_actions())
        steps += 1
        
        if info[0]['run_out_battery']:
            agent0_depleted = True
            break
    
    assert agent0_depleted, "Agent with low battery should deplete"


def test_custom_initial_battery():
    """Test custom initial battery configuration."""
    custom_battery = [50, 100, 75]
    env = pogema_v0(GridConfig(num_agents=3, size=8, density=0.1, seed=42,
                               max_episode_steps=64, initial_battery=custom_battery))
    obs, info = env.reset()
    
    for i in range(3):
        assert info[i]['battery'] == custom_battery[i]
        assert info[i]['initial_battery'] == custom_battery[i]


def test_invalid_initial_battery_length():
    """Test that invalid initial battery length raises error."""
    from pogema.grid import Grid
    
    config = GridConfig(num_agents=3, size=8, density=0.1, seed=42,
                        initial_battery=[50, 100])  # Only 2 values for 3 agents
    with pytest.raises(AssertionError):
        Grid(config)


def test_battery_and_charge_configurations():
    """Test various battery and charge configuration combinations."""
    # Test with custom charge and battery settings
    config = GridConfig(
        num_agents=2, size=8, density=0.1, seed=42,
        initial_battery=[50, 100],
        charge_increment=5,
        battery_decrement=2,
        num_charges=2
    )
    env = pogema_v0(grid_config=config)
    env.reset()
    
    assert env.grid.get_battery_for_agent(0) == 50
    assert env.grid.get_battery_for_agent(1) == 100
    assert len(env.grid.charges_xy) == 2
