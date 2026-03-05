import numpy as np
from typing import Tuple, Dict, Union
from typing import Optional
from collections import namedtuple

import gymnasium as gym
from gymnasium import spaces
import pygame
gym.logger.set_level(40)  # noqa

from gym_hybrid.agents import BaseAgent
from gym_hybrid.agents import MovingAgent
from gym_hybrid.agents import SlidingAgent


# Action Id
ACCELERATE = 0
TURN = 1
BREAK = 2


Target = namedtuple('Target', ['x', 'y', 'radius'])


# get the exact type and parameters of the action from a concatenated action (Sapce.Dict)
class ActionConverter:
    """"
    Action class to store and standardize the action for the environment.
    """
    def __init__(self, action_setting: Dict):
        """"
        Initialization of an action converter.

        Args:
            action_space: The action space of the environment.
        """
        self.n = len(action_setting)
        self.parameters_min = []
        self.parameters_max = []
        for key in action_setting:
            _params_min = []
            _params_max = []
            _params_min.append([action_setting[key][k][0] for k in action_setting[key]])
            _params_max.append([action_setting[key][k][1] for k in action_setting[key]])
            self.parameters_min.append(np.array(_params_min).flatten())
            self.parameters_max.append(np.array(_params_max).flatten())
                   
    def gym_space_setting(self) -> spaces.Dict:
        """"
        Method to return the action space setting in Gym.

        Returns:
            The action space setting.
        """
        action_space_dic = {
            'id': spaces.Discrete(self.n),
        }

        for i in range(self.n):
            action_space_dic['params'+str(i)] = spaces.Box(self.parameters_min[i], self.parameters_max[i])  # avoid use "parameters" name to avoid conflict with the nn.DictModule

        action_space = spaces.Dict(action_space_dic)

        return action_space

    # get the exact type and parameters of the action from a concatenated action (Sapce.Dict)
    def convert(self, action: Dict[str, Union[int, np.ndarray]]) -> Tuple[int, list]:
        """"
        Method to convert the action from the concatenated form to the separated form.

        Args:
            action: The concatenated action.

        Returns:
            The separated action.
        """
        id_ = action['id']
        act_parameters_ = action['params'+str(id_)]
        return id_, act_parameters_


class BaseEnv(gym.Env):
    """"
    Gym environment parent class.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
            self,
            render_mode = None,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 100,
            penalty: float = 0.001,
            break_value: float = 0.1,
            proximity_reward_scalar: float = 3.0,
            reward_scale: float = 0.1,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
            proximity_reward_scalar (float): Scalar for proximity-based reward.
            reward_scale (float): Global reward scaling factor.
        """
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.speed_threshold = 0.1
        self.penalty = penalty
        self.proximity_reward_scalar = proximity_reward_scalar
        self.reward_scale = reward_scale

        # Initialization
        self.target = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 400
        self.screen_height = 400
        self.agent_radius = 0.05
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)

        # parameters_min = np.array([0, -1])
        # parameters_max = np.array([1, +1])
        # self.action_space = spaces.Tuple((spaces.Discrete(3),
        #                                   spaces.Box(parameters_min, parameters_max)))
        parameterized_action_set = {
            ACCELERATE: {
                'acc': [0, 1]
            },
            TURN: {
                'roa': [-1, 1]
            },
            BREAK: {}
        }

        # parameterized_action_set = {
        #     ACCELERATE: {
        #         'acc': [-1, 1] # normalized value
        #     },
        #     TURN: {
        #         'roa': [-1, 1]
        #     },
        #     BREAK: {}
        # }

        self.action_converter = ActionConverter(parameterized_action_set)
        self.action_space = self.action_converter.gym_space_setting()
        
        # HER-compatible observation space
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.ones(8), np.ones(8), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32),
            'desired_goal': spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32),
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_step = 0

        limit = self.field_size-self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        return self.get_state(), {}

    def step(self, raw_action: Dict[str, Union[int, np.ndarray]]):
        id_, parameters_ = self.action_converter.convert(raw_action)
        last_distance = self.distance
        self.current_step += 1

        if id_ == TURN:
            rotation = self.max_turn * max(min(parameters_[0], 1), -1)
            self.agent.turn(rotation)
        elif id_ == ACCELERATE:
            # remap the acceleration from [-1, 1] to [0, 1]
            # parameters_[0] = (parameters_[0] + 1) / 2
            acceleration = self.max_acceleration * max(min(parameters_[0], 1), 0)
            self.agent.accelerate(acceleration)
        elif id_ == BREAK:
            self.agent.break_()

        state = self.get_state()
        achieved_goal = state['achieved_goal']
        desired_goal = state['desired_goal']
        info = {'speed': self.agent.speed}
        reward = self.compute_reward(achieved_goal, desired_goal, info) * self.reward_scale

        is_success = False
        truncated = False
        terminated = False
        if self.current_step < self.max_step:
            if abs(self.agent.x) > self.field_size or abs(self.agent.y) > self.field_size:
                reward = -1.0 * self.reward_scale
                terminated = True
        else:
            if reward > 0.0:
                is_success = True
            terminated = True

        if self.render_mode is not None:
            self.render()

        # add info to the return 
        info['reward'] = reward
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['is_success'] = is_success

        return self.get_state(), reward, terminated, truncated, info

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state in HER-compatible format.
        
        Returns:
            Dict with 'observation', 'achieved_goal', and 'desired_goal'.
        """
        observation = np.array([
            self.agent.x,
            self.agent.y,
            self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta),
            self.distance,
            0 if self.distance > self.target_radius else 1,
            self.current_step / self.max_step
        ], dtype=np.float32)

        achieved_goal = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        desired_goal = np.array([self.target.x, self.target.y], dtype=np.float32)

        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    # def get_reward(self, last_distance: float, goal: bool = False) -> float:
    #     # return (last_distance - self.distance) * self.proximity_reward_scalar - self.penalty + (1 if goal else 0)
    #     return 1 if goal else 0
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> Union[float, np.ndarray]:
        """
        Compute the reward for HER compatibility.
        Supports both single and vectorized (batch) computation.
        
        Args:
            achieved_goal: The achieved goal (agent position).
                          Shape: (2,) for single or (N, 2) for batch.
            desired_goal: The desired goal (target position).
                         Shape: (2,) for single or (N, 2) for batch.
            info: Additional information (dict for single, list of dicts for batch).
            
        Returns:
            Reward: 0.0 if goal is reached, -1.0 otherwise.
                   Returns float for single, ndarray for batch.
        """
        # Handle both single and batch cases
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        
        # Extract speed from info (handle both dict and list of dicts)
        if isinstance(info, dict):
            # Single case: info is a dict
            speed = np.array([info['speed']])
        else:
            # Batch case: info is a list of dicts
            speed = np.array([inf['speed'] for inf in info])

        # Compute distance between positions
        distance = np.sqrt(
            (achieved_goal[:, 0] - desired_goal[:, 0]) ** 2 +
            (achieved_goal[:, 1] - desired_goal[:, 1]) ** 2
        )
        
        # Goal is reached if within target radius and agent has stopped
        rewards = np.where(
            (distance < self.target_radius) & (speed == 0.0),
            1.0,
            0.0
        )
        
        # Return scalar if input was 1D, otherwise return array
        return rewards.item() if rewards.shape[0] == 1 else rewards

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod   
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # White background
        unit_x = self.screen_width // 2
        unit_y = self.screen_height // 2

        # Draw agent as a circle
        agent_position = (unit_x + int(self.agent.x * unit_x), unit_y + int(self.agent.y * unit_y))
        pygame.draw.circle(self.screen, (26, 77, 230), agent_position, int(self.agent_radius * unit_x))

        # Draw target as a circle
        target_position = (unit_x + int(self.target.x * unit_x), unit_y + int(self.target.y * unit_y))
        pygame.draw.circle(self.screen, (255, 128, 128), target_position, int(self.target_radius * unit_x))

        # Draw arrow indicating the agent's direction
        arrow_length = 30
        arrow_end = (agent_position[0] + arrow_length * np.cos(self.agent.theta),
                     agent_position[1] + arrow_length * np.sin(self.agent.theta))
        pygame.draw.line(self.screen, (0, 0, 0), agent_position, arrow_end, 3)

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])  # Limit to 60 FPS
        elif self.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
            
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class MovingEnv(BaseEnv):
    def __init__(
            self,
            render_mode = None,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 100,
            penalty: float = 0.001,
            break_value: float = 0.1,
            reward_scale: float = 0.1,
    ):

        super(MovingEnv, self).__init__(
            render_mode=render_mode,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
            reward_scale=reward_scale,
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )


class SlidingEnv(BaseEnv):
    def __init__(
            self,
            render_mode = None,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 100,
            penalty: float = 0.001,
            break_value: float = 0.1,
            reward_scale: float = 0.1,
    ):

        super(SlidingEnv, self).__init__(
            render_mode=render_mode,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
            reward_scale=reward_scale,
        )

        self.agent = SlidingAgent(
            break_value=break_value,
            delta_t=delta_t
        )
