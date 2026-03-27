import time

import numpy as np
from gymnasium import Wrapper


class AbstractMetric(Wrapper):
    def _compute_stats(self, step, is_on_goal, finished):
        raise NotImplementedError

    def __init__(self, env):
        super().__init__(env)
        self._current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        finished = all(truncated) or all(terminated)

        metric = self._compute_stats(self._current_step, self.was_on_goal, finished)
        self._current_step += 1
        if finished:
            self._current_step = 0

        if metric:
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(**metric)

        return obs, reward, terminated, truncated, infos


class RelativeBatteryMetric(AbstractMetric):
    """
    Computes the average relative battery level across all agents and all timesteps.
    Relative battery = current_battery / initial_battery for each agent at each step.
    """
    def __init__(self, env):
        super().__init__(env)
        self._battery_sum = 0.0
        self._battery_count = 0

    def _compute_stats(self, step, is_on_goal, finished):
        for agent_idx in range(self.get_num_agents()):
            battery = self.grid.get_battery_for_agent(agent_idx)
            initial_battery = self.grid.get_initial_battery_for_agent(agent_idx)
            self._battery_sum += battery / initial_battery
            self._battery_count += 1
        
        if finished:
            result = {'avg_relative_battery': self._battery_sum / self._battery_count if self._battery_count > 0 else 0.0}
            self._battery_sum = 0.0
            self._battery_count = 0
            return result


class LifeLongAvgThroughputWithActiveMetric(AbstractMetric):
    """
    Computes throughput considering active episode.
    In pogema-charge, agents may die early due to battery depletion,
    so this metric change valid_episode_steps rather than max_episode_steps.
    """
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0
        self._valid_episode_steps = 0

    def _compute_stats(self, step, is_on_goal, finished):
        for agent_idx, on_goal in enumerate(is_on_goal):
            if on_goal:
                self._solved_instances += 1
        
        self._valid_episode_steps += 1
        
        if finished:
            result = {'avg_throughput_with_active': self._solved_instances / self._valid_episode_steps if self._valid_episode_steps > 0 else 0.0, 
                      'valid_episode_relative': self._valid_episode_steps / self.grid_config.max_episode_steps if self._valid_episode_steps > 0 else 0.0
            }
            self._solved_instances = 0
            self._valid_episode_steps = 0
            return result


class BatteryDepletionRateMetric(AbstractMetric):
    """
    Computes the rate at which agents run out of battery during an episode.
    Also tracks the average battery level at episode end for agents that survived.
    """
    def __init__(self, env):
        super().__init__(env)
        self._depleted_count = 0

    def _compute_stats(self, step, is_on_goal, finished):
        for agent_idx in range(self.get_num_agents()):
            if self.env.was_run_out_battery[agent_idx]:
                self._depleted_count += 1
        
        if finished:
            result = {
                'agents_depletion_rate': self._depleted_count / self.get_num_agents(),
                'agents_depleted_count': self._depleted_count
            }
            self._depleted_count = 0
            return result


class ChargingEfficiencyMetric(AbstractMetric):
    """
    Computes how effectively agents use charging stations.
    Metrics include:
    - charging_visits: total number of times agents stepped on charging stations
    - avg_charging_rate: percentage of agents that stepped on charging stations
    - avg_charging_per_agent: average charging events per agent
    """
    def __init__(self, env):
        super().__init__(env)
        self._charging_events = 0
        self._had_charging = self.get_num_agents() * [False]

    def _compute_stats(self, step, is_on_goal, finished):
        for agent_idx in range(self.get_num_agents()):
            if self.grid.on_charges(agent_idx) and self.grid.is_active[agent_idx]:
                self._charging_events += 1
                self._had_charging[agent_idx] = True
        
        if finished:
            result = {
                'charging_visits': self._charging_events,
                'avg_charging_rate': sum(self._had_charging), 
                'avg_charging_per_agent': self._charging_events / self.get_num_agents()
            }
            self._charging_events = 0
            self._had_charging = self.get_num_agents() * [False]
            return result


class BatteryHealthMetric(AbstractMetric):
    """
    Computes battery health metrics:
    - avg_goal_battery_relative: average remaining battery (relative to initial) for agents that reached goal
    """
    def __init__(self, env):
        super().__init__(env)
        self._goal_battery_sum = 0.0
        self._goal_count = 0

    def _compute_stats(self, step, is_on_goal, finished):
        for agent_idx in range(self.get_num_agents()):
            battery = self.grid.get_battery_for_agent(agent_idx)
            initial_battery = self.grid.get_initial_battery_for_agent(agent_idx)
            if self.env.was_on_goal[agent_idx]:
                self._goal_battery_sum += battery / initial_battery
                self._goal_count += 1
                    
        if finished:
            result = {
                'avg_goal_battery_relative': self._goal_battery_sum / self._goal_count if self._goal_count > 0 else 0.0,
            }
            self._goal_battery_sum = 0.0
            self._goal_count = 0
            return result


class LifeLongAverageThroughputMetric(AbstractMetric):

    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished):
        for agent_idx, on_goal in enumerate(is_on_goal):
            if on_goal:
                self._solved_instances += 1
        if finished:
            result = {'avg_throughput': self._solved_instances / self.grid_config.max_episode_steps}
            self._solved_instances = 0
            return result


class NonDisappearCSRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished):
        if finished:
            return {'CSR': float(all(is_on_goal))}


class NonDisappearISRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished):
        if finished:
            return {'ISR': float(sum(is_on_goal)) / self.get_num_agents()}


class NonDisappearEpLengthMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished):
        if finished:
            return {'ep_length': step + 1}


class EpLengthMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solve_time = [None for _ in range(self.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, finished):
        for idx, on_goal in enumerate(is_on_goal):
            if self._solve_time[idx] is None:
                if on_goal or finished:
                    self._solve_time[idx] = step

        if finished:
            result = {'ep_length': sum(self._solve_time) / self.get_num_agents() + 1}
            self._solve_time = [None for _ in range(self.get_num_agents())]
            return result


class CSRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished):
        self._solved_instances += sum(is_on_goal)
        if finished:
            results = {'CSR': float(self._solved_instances == self.get_num_agents())}
            self._solved_instances = 0
            return results


class ISRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished):
        self._solved_instances += sum(is_on_goal)
        if finished:
            results = {'ISR': self._solved_instances / self.get_num_agents()}
            self._solved_instances = 0
            return results


class SumOfCostsAndMakespanMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solve_time = [None for _ in range(self.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, finished):
        for idx, on_goal in enumerate(is_on_goal):
            if self._solve_time[idx] is None and (on_goal or finished):
                self._solve_time[idx] = step
            if not on_goal and not finished:
                self._solve_time[idx] = None

        if finished:
            result = {'SoC': sum(self._solve_time) + self.get_num_agents(), 'makespan': max(self._solve_time) + 1}
            self._solve_time = [None for _ in range(self.get_num_agents())]
            return result


class AgentsDensityWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._avg_agents_density = None

    def count_agents(self, observations):
        avg_agents_density = []
        for obs in observations:
            traversable_cells = np.size(obs['obstacles']) - np.count_nonzero(obs['obstacles'])
            avg_agents_density.append(np.count_nonzero(obs['agents']) / traversable_cells)
        self._avg_agents_density.append(np.mean(avg_agents_density))

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.count_agents(observations)
        if all(terminated) or all(truncated):
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(avg_agents_density=float(np.mean(self._avg_agents_density)))
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        self._avg_agents_density = []
        observations, info = self.env.reset(**kwargs)
        self.count_agents(observations)
        return observations, info


class RuntimeMetricWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None
        self._env_step_time = None

    def step(self, actions):
        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start
        if all(terminated) or all(truncated):
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._start_time = time.monotonic()
        self._env_step_time = 0.0
        return obs
