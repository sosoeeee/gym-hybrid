import os
import sys

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gym_hybrid


ENV_ID = "Moving-v0"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_STEPS = 250


def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def make_action(action_id: int, param_value: float, action_space):
    action = {"id": action_id}
    for key, space in action_space.spaces.items():
        if key == "id":
            continue
        if space.shape[0] == 0:
            action[key] = np.zeros((0,), dtype=space.dtype)
        else:
            action[key] = np.zeros(space.shape, dtype=space.dtype)
    param_key = f"params{action_id}"
    if param_key in action and action[param_key].shape[0] > 0:
        action[param_key][0] = param_value
    return action


def concat_params(action, action_space):
    params = [action[k].astype(np.float32) for k in action_space.spaces.keys() if k != "id"]
    return np.concatenate(params) if params else np.zeros((0,), dtype=np.float32)


def expert_policy(obs_dict, target_radius, env=None):
    """
    Expert policy for HER-compatible Dict observations.
    
    Args:
        obs_dict: Dict with 'observation', 'achieved_goal', 'desired_goal'
        target_radius: Target radius
        env: Environment instance
        
    Returns:
        action_id, param_value
    """
    obs = obs_dict['observation']
    achieved_goal = obs_dict['achieved_goal']
    desired_goal = obs_dict['desired_goal']
    
    agent_x, agent_y = achieved_goal[0], achieved_goal[1]
    target_x, target_y = desired_goal[0], desired_goal[1]
    speed = obs[2]
    cos_theta, sin_theta = obs[3], obs[4]
    distance = obs[5]
    
    heading = np.arctan2(sin_theta, cos_theta)
    desired = np.arctan2(target_y - agent_y, target_x - agent_x)
    delta = normalize_angle(desired - heading)

    if distance <= target_radius * 1.2:
        return 2, 0.0

    angle_thresh = 0.05
    if abs(delta) > angle_thresh:
        return 1, float(np.clip(delta / (np.pi / 2), -1.0, 1.0))

    # Adaptive stopping: use env params when available for accurate stopping distance
    # Fallback to defaults otherwise
    if env is not None:
        break_value = float(env.unwrapped.break_value)
        delta_t = float(env.unwrapped.delta_t)
    else:
        break_value = 0.1
        delta_t = 0.005

    # Estimate stopping distance by simulating repeated BREAK steps
    est_stop_dist = 0.0
    s = float(speed)
    while s > 1e-6:
        est_stop_dist += s * delta_t
        s = max(0.0, s - break_value)

    margin = max(0.05, target_radius * 0.4)
    hysteresis = 0.1

    if distance <= est_stop_dist + margin:
        return 2, 0.0
    if distance > est_stop_dist + margin + hysteresis:
        return 0, 1.0
    return 0, 0.0


def run_episode(env, render=False):
    """
    Run a single episode with expert policy and collect data in HER-compatible format.
    
    Args:
        env: Gym environment
        render: Whether to render the environment during collection
    
    Returns:
        success: Whether the episode was successful
        obs_observation, obs_achieved_goal, obs_desired_goal: Separated observations
        action_ids, action_params: Actions taken
        rewards, dones, infos: Step information
        start, target: Episode metadata
    """
    # Reset environment and capture the initial start/target
    obs_dict, _ = env.reset()
    start = (float(env.unwrapped.agent.x), float(env.unwrapped.agent.y), float(env.unwrapped.agent.theta))
    target = (float(env.unwrapped.target.x), float(env.unwrapped.target.y))

    # Determine max parameter dimension across all action types
    max_param_dim = max(
        env.action_space.spaces[k].shape[0]
        for k in env.action_space.spaces.keys()
        if k != "id"
    )
    
    # Store step-level data (HER-compatible format)
    obs_observation_list = []  # observation component
    obs_achieved_goal_list = []  # achieved_goal component
    obs_desired_goal_list = []  # desired_goal component
    action_ids = []  # action IDs
    action_params = []  # normalized parameters (pmax format)
    rewards = []
    dones = []
    infos = []
    
    done = False
    steps = 0
    success = False

    if render:
        env.render()

    while not done and steps < MAX_STEPS:
        action_id, param = expert_policy(obs_dict, env.unwrapped.target_radius, env)
        action = make_action(action_id, param, env.action_space)
        
        # Store observation components before action
        obs_observation_list.append(obs_dict['observation'].copy())
        obs_achieved_goal_list.append(obs_dict['achieved_goal'].copy())
        obs_desired_goal_list.append(obs_dict['desired_goal'].copy())
        action_ids.append(action_id)
        
        # Create normalized parameter in pmax format (aligned dimension)
        param_key = f"params{action_id}"
        current_param = action[param_key].astype(np.float32)
        # normalize parameter to -1 to 1 range for consistency
        param_low = env.action_space.spaces[param_key].low
        param_high = env.action_space.spaces[param_key].high
        if len(param_low) == 0 or len(param_high) == 0:
            current_param_norm = np.zeros_like(current_param, dtype=np.float32)
        else:
            current_param_norm = 2.0 * (current_param - param_low) / (param_high - param_low) - 1.0
        pmax = np.zeros(max_param_dim, dtype=np.float32)
        pmax[:len(current_param_norm)] = current_param_norm
        action_params.append(pmax)

        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render()
        
        rewards.append(float(reward))
        dones.append(terminated or truncated)
        infos.append(dict(info))
        
        done = terminated or truncated
        if terminated and info.get("is_success", False):
            success = True
        steps += 1

    return (success, obs_observation_list, obs_achieved_goal_list, obs_desired_goal_list,
            action_ids, action_params, rewards, dones, infos, start, target)


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-n", type=int, default=1, help="number of successful demos to collect")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="where to save demo files")
    parser.add_argument("--max-attempts", type=int, default=10, help="max attempts per demo")
    parser.add_argument("--render", "-r", action='store_true', help="render the environment during collection (human mode)")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Set render mode based on --render flag
    if args.render:
        # Remove dummy driver to allow screen display
        if 'SDL_VIDEODRIVER' in os.environ:
            del os.environ['SDL_VIDEODRIVER']
        if 'SDL_AUDIODRIVER' in os.environ:
            del os.environ['SDL_AUDIODRIVER']
        render_mode = 'human'
    else:
        # Keep dummy driver for headless collection
        render_mode = None
    
    env = gym.make(ENV_ID, render_mode=render_mode)

    # Collect all episodes data (HER-compatible format)
    all_obs_observation = []
    all_obs_achieved_goal = []
    all_obs_desired_goal = []
    all_action_ids = []
    all_action_params = []
    all_rewards = []
    all_dones = []
    all_infos = []
    all_episode_ids = []
    all_meta = []
    
    success_count = 0
    attempts = 0

    while success_count < args.episodes and attempts < args.episodes * args.max_attempts:
        result = run_episode(env, render=args.render)
        ok, obs_obs, obs_ag, obs_dg, ids, params, rewards, dones, infos, start, target = result
        attempts += 1
        if not ok:
            continue

        # Accumulate step-level data
        all_obs_observation.extend(obs_obs)
        all_obs_achieved_goal.extend(obs_ag)
        all_obs_desired_goal.extend(obs_dg)
        all_action_ids.extend(ids)
        all_action_params.extend(params)
        all_rewards.extend(rewards)
        all_dones.extend(dones)
        all_infos.extend(infos)
        all_episode_ids.extend([success_count] * len(obs_obs))
        
        # Store episode metadata
        all_meta.append({
            'start': np.asarray(start, dtype=np.float32) if start is not None else np.asarray([], dtype=np.float32),
            'target': np.asarray(target, dtype=np.float32) if target is not None else np.asarray([], dtype=np.float32),
            'steps': len(obs_obs),
        })
        
        success_count += 1
        print(f"Collected expert demo {success_count}/{args.episodes} (attempt {attempts})")

    env.close()

    if success_count < args.episodes:
        raise RuntimeError(f"Failed to collect {args.episodes} successful demo(s) after {attempts} attempts")

    # Create output dir if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save in HER-compatible format with separate observation components
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_name = os.path.join(args.output_dir, f"expert_demos_{ENV_ID}_{success_count}_{ts}.npz")
    
    # Convert to numpy arrays
    obs_observation_array = np.asarray(all_obs_observation, dtype=np.float32)
    obs_achieved_goal_array = np.asarray(all_obs_achieved_goal, dtype=np.float32)
    obs_desired_goal_array = np.asarray(all_obs_desired_goal, dtype=np.float32)
    action_id_array = np.asarray(all_action_ids, dtype=np.int64)
    action_params_array = np.asarray(all_action_params, dtype=np.float32)
    rewards_array = np.asarray(all_rewards, dtype=np.float32)
    dones_array = np.asarray(all_dones, dtype=np.bool_)
    episode_id_array = np.asarray(all_episode_ids, dtype=np.int64)
    info_array = np.asarray(all_infos, dtype=object)
    meta_array = np.asarray(all_meta, dtype=object)
    
    # Compute success for each step (success at end of successful episodes)
    success_array = np.zeros(len(all_obs_observation), dtype=np.bool_)
    for i, (done, ep_id) in enumerate(zip(all_dones, all_episode_ids)):
        if done and ep_id < success_count:  # all collected episodes are successful
            success_array[i] = True
    
    np.savez_compressed(
        out_name,
        obs_observation=obs_observation_array,
        obs_achieved_goal=obs_achieved_goal_array,
        obs_desired_goal=obs_desired_goal_array,
        action_id=action_id_array,
        action_params=action_params_array,
        rewards=rewards_array,
        dones=dones_array,
        success=success_array,
        episode_id=episode_id_array,
        info=info_array,
        meta=meta_array,
    )
    
    print(f"\n✓ Saved {success_count} expert demos to {out_name}")
    print(f"  Total steps: {len(all_obs_observation)}")
    print(f"  Success rate: {success_count}/{attempts} ({100*success_count/attempts:.1f}%)")


if __name__ == "__main__":
    main()
