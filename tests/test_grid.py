import numpy as np
from pydantic import ValidationError

from pogema import GridConfig
from pogema.grid import Grid
import pytest

from pogema.integrations.make_pogema import pogema_v0


def test_obstacle_creation():
    config = GridConfig(seed=1, obs_radius=2, size=5, num_agents=1, density=0.2)
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(Grid(config).obstacles, obstacles).all()

    config = GridConfig(seed=3, obs_radius=1, size=4, num_agents=1, density=0.4)
    obstacles = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    assert np.isclose(Grid(config).obstacles, obstacles).all()


def test_initial_positions():
    config = GridConfig(seed=1, obs_radius=2, size=5, num_agents=1, density=0.2)
    positions_xy = [(2, 4)]
    assert np.isclose(Grid(config).positions_xy, positions_xy).all()

    config = GridConfig(seed=1, obs_radius=2, size=12, num_agents=10, density=0.2)
    positions_xy = [(13, 10), (7, 4), (4, 3), (2, 11), (12, 6), (8, 11), (6, 8), (2, 12), (2, 10), (9, 11)]
    assert np.isclose(Grid(config).positions_xy, positions_xy).all()


def test_goals():
    config = GridConfig(seed=1, obs_radius=2, size=5, num_agents=1, density=0.4)
    finishes_xy = [(5, 2)]
    assert np.isclose(Grid(config).finishes_xy, finishes_xy).all()

    config = GridConfig(seed=2, obs_radius=2, size=12, num_agents=10, density=0.2)
    finishes_xy = [(11, 10), (8, 11), (2, 13), (3, 5), (12, 6), (9, 12), (9, 6), (9, 2), (10, 2), (6, 11)]
    assert np.isclose(Grid(config).finishes_xy, finishes_xy).all()


def test_overflow():
    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=100, density=0.0))

    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=1, density=1.0))


def test_overflow_warning():
    with pytest.warns(Warning):
        for _ in range(1000):
            Grid(GridConfig(obs_radius=2, size=4, num_agents=6, density=0.3), num_retries=10000)


def test_edge_cases():
    with pytest.raises(ValidationError):
        GridConfig(seed=1, obs_radius=2, size=1, num_agents=1, density=0.4)

    with pytest.raises(ValidationError):
        GridConfig(seed=1, obs_radius=2, size=4, num_agents=0, density=0.4)

    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=1, density=1.0))

    with pytest.raises(ValidationError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=1, density=2.0))


def test_edge_cases_for_custom_map():
    test_map = [[0, 0, 0]]
    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=2, obs_radius=2, size=4, num_agents=4, map=test_map))


def test_custom_map():
    test_map = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    grid = Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(grid.obstacles, obstacles).all()

    test_map = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    grid = Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(grid.obstacles, obstacles).all()

    test_map = [
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
    ]
    grid = Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(grid.obstacles, obstacles).all()


def test_overflow_for_custom_map():
    test_map = [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ]
    with pytest.raises(OverflowError):
        Grid(GridConfig(obs_radius=2, size=4, num_agents=5, density=0.3, map=test_map), num_retries=100)


def test_str_custom_map():
    grid_map = """
        .a...#.....
        .....#.....
        ..C.....b..
        .....#.....
        .....#.....
        #.####.....
        .....###.##
        .....#.....
        .c...#.....
        .B.......A.
        .....#.....
    """
    grid = Grid(GridConfig(obs_radius=2, size=4, density=0.3, map=grid_map))
    assert (grid.config.num_agents == 3)
    assert (np.isclose(0.1404958, grid.config.density))
    assert (np.isclose(11, grid.config.size))

    grid_map = """.....#...."""
    grid = Grid(GridConfig(seed=2, num_agents=3, map=grid_map))
    assert (grid.config.num_agents == 3)
    assert (np.isclose(0.1, grid.config.density))
    assert (np.isclose(10, grid.config.size))


def test_custom_starts_and_finishes_random():
    agents_xy = [(x, x) for x in range(8)]
    targets_xy = [(x, x) for x in range(8, 16)]
    grid_config = GridConfig(seed=12, size=16, num_agents=8, agents_xy=agents_xy, targets_xy=targets_xy)
    env = pogema_v0(grid_config=grid_config)
    env.reset()
    r = grid_config.obs_radius
    assert [(x - r, y - r) for x, y in env.grid.positions_xy] == agents_xy and \
           [(x - r, y - r) for x, y in env.grid.finishes_xy] == targets_xy


def test_out_of_bounds_for_custom_positions():
    Grid(GridConfig(seed=12, size=17, agents_xy=[[0, 16]], targets_xy=[[16, 0]]))

    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[0, 17]], targets_xy=[[0, 0]])
    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[0, 0]], targets_xy=[[0, 17]])
    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[-1, 0]], targets_xy=[[0, 0]])
    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[0, 0]], targets_xy=[[0, -1]])


def test_duplicated_params():
    grid_map = "Aa"
    with pytest.raises(KeyError):
        GridConfig(agents_xy=[[0, 0]], targets_xy=[[0, 0]], map=grid_map)


def test_custom_grid_with_empty_agents_and_targets():
    grid_map = """...."""
    Grid(GridConfig(agents_xy=None, targets_xy=None, map=grid_map, num_agents=1))


def test_custom_grid_with_specific_positions():
    grid_map = """
        !!!!!!!!!!!!!!!!!!
        !@@!@@!$$$$$$$$$$!
        !@@!@@!##########!
        !@@!@@!$$$$$$$$$$!
        !!!!!!!!!!!!!!!!!!
        !@@!@@!$$$$$$$$$$!
        !@@!@@!##########!
        !@@!@@!$$$$$$$$$$!
        !!!!!!!!!!!!!!!!!!
    """
    Grid(GridConfig(obs_radius=2, size=4, num_agents=24, map=grid_map))
    with pytest.raises(OverflowError):
        Grid(GridConfig(obs_radius=2, size=4, num_agents=25, map=grid_map))

    grid_map = """
        !!!!!!!!!!!
        !@@!@@!$$$$
        !@@!@@!####
        !@@!@@!$$$$
        !!!!!!!!!!!
        !@@!@@!$$$$
        !@@!@@!####
        !@@!@@!$$$$
        !!!!!!!!!!!
    """
    Grid(GridConfig(obs_radius=2, num_agents=16, map=grid_map))
    with pytest.raises(OverflowError):
        Grid(GridConfig(obs_radius=2, num_agents=17, map=grid_map))

    grid_map = """
            !!!!!!!!!!!
            !@@!@@!.Ab.
            !@@!@@!####
            !@@!@@!.aB.

        """
    with pytest.raises(KeyError):
        Grid(GridConfig(obs_radius=2, map=grid_map))


def test_restricted_grid():
    grid = """
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           !@@!@@!$$$$$$$$$$!$$$$$$$$$$!$$$$$$$$$$!@@!@@!
           !@@!@@!##########!##########!##########!@@!@@!
           !@@!@@!$$$$$$$$$$!$$$$$$$$$$!$$$$$$$$$$!@@!@@!
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           """
    env = pogema_v0(grid_config=GridConfig(map=grid, num_agents=24, seed=0, obs_radius=2))
    env.reset()

    with pytest.raises(OverflowError):
        env = pogema_v0(grid_config=GridConfig(map=grid, num_agents=25, seed=0, obs_radius=2))
        env.reset()


def test_rectangular_grid_basic():
    config = GridConfig(width=12, height=8)
    assert np.isclose(config.width, 12)
    assert np.isclose(config.height, 8)
    assert np.isclose(config.size, 12)


def test_rectangular_grid_backward_compatibility():
    config = GridConfig(size=10)
    assert np.isclose(config.size, 10)
    assert np.isclose(config.width, 10)
    assert np.isclose(config.height, 10)


def test_rectangular_grid_mixed_config():
    config = GridConfig(size=8, width=12, height=6)
    assert np.isclose(config.width, 12)
    assert np.isclose(config.height, 6)
    assert np.isclose(config.size, 12)


def test_rectangular_grid_validation():
    with pytest.raises(ValueError):
        GridConfig(width=12)
    
    with pytest.raises(ValueError):
        GridConfig(height=8)
    
    GridConfig(width=12, height=8)
    GridConfig(size=10)
    GridConfig(size=10, width=12, height=8)


def test_rectangular_grid_position_validation():
    config = GridConfig(width=12, height=8, agents_xy=[[0, 11], [7, 0]], targets_xy=[[7, 11], [0, 0]])
    assert len(config.agents_xy) == 2
    assert len(config.targets_xy) == 2
    
    with pytest.raises(IndexError):
        GridConfig(width=12, height=8, agents_xy=[[8, 0]], targets_xy=[[0, 0]])
    
    with pytest.raises(IndexError):
        GridConfig(width=12, height=8, agents_xy=[[0, 12]], targets_xy=[[0, 0]])


def test_rectangular_grid_creation():
    config = GridConfig(width=12, height=8, seed=1, num_agents=2)
    grid = Grid(config)
    
    assert np.isclose(grid.config.width, 12)
    assert np.isclose(grid.config.height, 8)
    assert np.isclose(grid.config.size, 12)


def test_goal_sequences_validation():
    config = GridConfig(
        width=8, height=8,
        agents_xy=[[0, 0], [1, 1]],
        targets_xy=[
            [[2, 2], [3, 3], [4, 4]],
            [[2, 4], [3, 5]]
        ]
    )
    assert np.isclose(len(config.targets_xy), 2)
    assert np.isclose(len(config.targets_xy[0]), 3)
    assert np.isclose(len(config.targets_xy[1]), 2)
    
    config = GridConfig(
        width=8, height=8,
        agents_xy=[[0, 0], [1, 1]],
        targets_xy=[[7, 7], [6, 6]]
    )
    assert np.isclose(len(config.targets_xy), 2)
    
    with pytest.raises(ValueError):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0], [1, 1]],
            targets_xy=[[[2, 2], [3, 3]], [4, 4]]
        )
    
    with pytest.raises(ValueError):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0]],
            targets_xy=[[[2, 2]]]
        )
    
    with pytest.raises(ValueError):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0]],
            targets_xy=[[[2.5, 2], [3, 3]]]
        )
    
    with pytest.raises(IndexError):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0]],
            targets_xy=[[[2, 2], [10, 10]]]
        )
    
    with pytest.raises(ValueError, match="on_target='restart' requires goal sequences"):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0], [1, 1]],
            targets_xy=[[2, 2], [3, 3]],
            on_target='restart'
        )


def test_grid_with_goal_sequences():
    config = GridConfig(
        width=8, height=8,
        agents_xy=[[0, 0], [1, 1]],
        targets_xy=[
            [[2, 2], [3, 3], [4, 4]],
            [[2, 4], [3, 5]]
        ]
    )
    
    grid = Grid(config)
    
    expected_initial_targets = [[2, 2], [2, 4]]
    r = config.obs_radius
    expected_with_offset = [(x + r, y + r) for x, y in expected_initial_targets]
    
    assert np.isclose(grid.finishes_xy, expected_with_offset).all()


def test_pogema_lifelong_with_sequences():
    from pogema.envs import PogemaLifeLong
    import warnings
    
    config = GridConfig(
        width=8, height=8,
        agents_xy=[[1, 1], [1, 2]],
        targets_xy=[
            [[2, 2], [3, 3], [4, 4]],
            [[2, 4], [3, 5]]
        ],
        on_target='restart'
    )
    
    env = PogemaLifeLong(grid_config=config)
    obs = env.reset()
    
    assert env.has_custom_sequences == True
    assert np.isclose(env.current_goal_indices, [0, 0]).all()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        target1 = env._generate_new_target(0)
        assert np.isclose(env.current_goal_indices[0], 1)
        
        target2 = env._generate_new_target(0)
        assert np.isclose(env.current_goal_indices[0], 2)
        
        target3 = env._generate_new_target(0)
        assert np.isclose(env.current_goal_indices[0], 0)
        
        cycling_warnings = [warning for warning in w if "completed all 3 provided targets" in str(warning.message)]
        assert np.isclose(len(cycling_warnings), 1)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        env._generate_new_target(1)
        env._generate_new_target(1)
        
        assert np.isclose(len(w), 1)
        assert "completed all 2 provided targets" in str(w[0].message)
        assert "cycling back to the beginning" in str(w[0].message)


def test_pogema_lifelong_reset():
    from pogema.envs import PogemaLifeLong
    
    config = GridConfig(
        width=8, height=8,
        agents_xy=[[1, 1], [1, 2]],
        targets_xy=[
            [[2, 2], [3, 3]],
            [[2, 4], [3, 5]]
        ],
        on_target='restart'
    )
    
    env = PogemaLifeLong(grid_config=config)
    env.reset()
    
    env._generate_new_target(0)
    env._generate_new_target(1)
    assert np.isclose(env.current_goal_indices, [1, 1]).all()
    
    env.reset()
    assert np.isclose(env.current_goal_indices, [0, 0]).all()


def test_pogema_lifelong_without_sequences():
    from pogema.envs import PogemaLifeLong
    
    config = GridConfig(
        width=8, height=8,
        num_agents=2,
        on_target='restart'
    )
    
    env = PogemaLifeLong(grid_config=config)
    obs = env.reset()
    
    assert env.has_custom_sequences == False
    
    target = env._generate_new_target(0)
    assert isinstance(target, tuple)
    assert np.isclose(len(target), 2)


def test_goal_sequences_position_format():
    with pytest.raises(ValueError, match="Position must be a list/tuple of length 2"):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0]],
            targets_xy=[[[2, 2, 3], [4, 4]]]
        )

    with pytest.raises(ValueError, match="Position coordinates must be integers"):
        GridConfig(
            width=8, height=8,
            agents_xy=[[0, 0]],
            targets_xy=[[[2.5, 2], [4, 4]]]
        )


# ============== Charge Station Tests ==============

def test_charges_xy_in_grid():
    """Test that charge stations are properly created in the grid."""
    config = GridConfig(seed=42, size=8, num_agents=2, num_charges=3, density=0.2)
    grid = Grid(config)
    
    assert len(grid.charges_xy) == 3
    # Check charge stations are within bounds
    r = config.obs_radius
    for cx, cy in grid.charges_xy:
        assert r <= cx < config.height + r
        assert r <= cy < config.width + r


def test_charges_xy_custom_positions():
    """Test custom charge station positions."""
    charges_xy = [[2, 2], [5, 5], [7, 3]]
    agents_xy = [[0, 0], [1, 1]]
    targets_xy = [[8, 8], [7, 7]]
    config = GridConfig(
        size=10, num_agents=2, num_charges=3,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy, density=0.2
    )
    grid = Grid(config)
    
    r = config.obs_radius
    expected = [[x + r, y + r] for x, y in charges_xy]
    assert np.isclose(grid.charges_xy, expected).all()


def test_charges_xy_from_map_string():
    """Test charge stations defined in map string using digits."""
    grid_map = """
        a.0.A
        .....
        .1...
    """
    config = GridConfig(map=grid_map)
    grid = Grid(config)
    
    # Should have 2 charge stations (0 and 1)
    assert len(grid.charges_xy) == 2


def test_charges_xy_relative():
    """Test relative charge station positions."""
    charges_xy = [[4, 4], [8, 6]]
    agents_xy = [[1, 1], [2, 2]]
    targets_xy = [[7, 7], [6, 6]]
    
    config = GridConfig(
        size=10, num_agents=2, num_charges=2,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy
    )
    env = pogema_v0(grid_config=config)
    env.reset()
    
    charges_relative = env.get_charges_xy_relative()
    assert len(charges_relative) == 2  # One list per agent


def test_on_charges_method():
    """Test on_charges method in Grid."""
    charges_xy = [[3, 3]]
    agents_xy = [[3, 3]]  # Agent starts on charge station
    targets_xy = [[7, 7]]
    
    config = GridConfig(
        size=10, num_agents=1, num_charges=1,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy
    )
    grid = Grid(config)
    
    assert grid.on_charges(0) == True
    
    # Move agent away
    grid.move_without_checks(0, 1)  # up
    assert grid.on_charges(0) == False


def test_battery_charging():
    """Test that battery increases when on charge station."""
    charges_xy = [[3, 3]]
    agents_xy = [[3, 3]]  # Agent starts on charge station
    targets_xy = [[7, 7]]
    
    config = GridConfig(
        size=10, num_agents=1, num_charges=1,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy,
        initial_battery=[50], charge_increment=5
    )
    grid = Grid(config)
    
    initial_battery = grid.get_battery_for_agent(0)
    
    # Wait on charge station (action 0)
    grid.move_without_checks(0, 0)
    
    # Battery should increase by charge_increment (capped at initial)
    new_battery = grid.get_battery_for_agent(0)
    assert new_battery == min(initial_battery + 5, 50)


def test_battery_capped_at_initial():
    """Test that battery cannot exceed initial battery level."""
    charges_xy = [[3, 3]]
    agents_xy = [[3, 3]]
    targets_xy = [[7, 7]]
    
    config = GridConfig(
        size=10, num_agents=1, num_charges=1,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy,
        initial_battery=[50], charge_increment=10
    )
    grid = Grid(config)
    
    # Battery starts at 50
    assert grid.get_battery_for_agent(0) == 50
    
    # Wait on charge station multiple times
    for _ in range(10):
        grid.move_without_checks(0, 0)
    
    # Battery should still be capped at 50
    assert grid.get_battery_for_agent(0) == 50


def test_run_out_battery():
    """Test run_out_battery method."""
    config = GridConfig(
        size=10, num_agents=1,
        initial_battery=[3], battery_decrement=1
    )
    grid = Grid(config)
    
    assert grid.run_out_battery(0) == False
    
    # Move until battery depletes
    for _ in range(3):
        grid.move_without_checks(0, 1)  # up
    
    assert grid.run_out_battery(0) == True


def test_charge_increment_validation():
    """Test charge_increment must be in valid range."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        GridConfig(charge_increment=0)
    
    with pytest.raises(ValidationError):
        GridConfig(charge_increment=101)
    
    # Valid values
    GridConfig(charge_increment=1)
    GridConfig(charge_increment=50)
    GridConfig(charge_increment=100)


def test_battery_decrement_validation():
    """Test battery_decrement must be in valid range."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        GridConfig(battery_decrement=0)
    
    with pytest.raises(ValidationError):
        GridConfig(battery_decrement=101)
    
    # Valid values
    GridConfig(battery_decrement=1)
    GridConfig(battery_decrement=50)
    GridConfig(battery_decrement=100)


def test_num_charges_validation():
    """Test num_charges must be positive."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        GridConfig(num_charges=0)
    
    # Valid values
    GridConfig(num_charges=1)
    GridConfig(num_charges=100)


def test_charges_xy_validation():
    """Test charges_xy must be within bounds."""
    with pytest.raises(IndexError):
        GridConfig(size=8, charges_xy=[[10, 10]])
    
    with pytest.raises(IndexError):
        GridConfig(size=8, charges_xy=[[-1, 0]])


def test_get_charges_direction():
    """Test get_charges returns direction vectors to charge stations."""
    charges_xy = [[5, 5]]
    agents_xy = [[3, 3]]
    targets_xy = [[7, 7]]
    
    config = GridConfig(
        size=10, num_agents=1, num_charges=1,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy
    )
    grid = Grid(config)
    
    directions = grid.get_charges(0)
    # Should return list of direction tuples
    assert isinstance(directions, list)
    if len(directions) > 0:
        assert len(directions[0]) == 2
        # Direction should be normalized
        assert 0.0 <= directions[0][0] <= 1.0
        assert 0.0 <= directions[0][1] <= 1.0


def test_get_square_charges():
    """Test get_square_charges returns heatmap in observation radius."""
    charges_xy = [[5, 5]]
    agents_xy = [[3, 3]]
    targets_xy = [[7, 7]]
    
    config = GridConfig(
        size=10, num_agents=1, num_charges=1, obs_radius=3,
        charges_xy=charges_xy, agents_xy=agents_xy, targets_xy=targets_xy
    )
    grid = Grid(config)
    
    square_charges = grid.get_square_charges(0)
    # Should return 2D array of size (2*obs_radius+1, 2*obs_radius+1)
    expected_size = 2 * config.obs_radius + 1
    assert square_charges.shape == (expected_size, expected_size)
    # Should have exactly one charge station marked
    assert np.sum(square_charges) == 1.0


def test_map_with_charge_stations():
    """Test full map string with charge stations."""
    grid_map = """
        .........
        .a.....A.
        ....0....
        .........
    """
    config = GridConfig(map=grid_map, num_agents=1)
    grid = Grid(config)
    
    assert len(grid.charges_xy) == 1
    assert grid.config.num_charges == 1


def test_multiple_charge_stations_map():
    """Test map with multiple charge stations."""
    grid_map = """
        .........
        .a.0.1.A.
        .........
    """
    config = GridConfig(map=grid_map)
    grid = Grid(config)
    
    assert len(grid.charges_xy) == 2


def test_charges_not_on_obstacles():
    """Test that charge stations are not placed on obstacles."""
    charges_xy = [[3, 3]]
    agents_xy = [[1, 1]]
    targets_xy = [[7, 7]]
    
    # Create map with obstacle at charge station location
    grid_map = """
        .........
        .a......A
        .........
        ...#.....
        .........
    """
    # Override with custom positions
    config = GridConfig(
        map=grid_map,
        charges_xy=[[3, 3]],
        agents_xy=[[1, 1]],
        targets_xy=[[1, 7]]
    )
    grid = Grid(config)
    
    # Charge station should be on free cell
    cx, cy = grid.charges_xy[0]
    assert grid.obstacles[cx, cy] == config.FREE
