import os
import sys

import numpy as np

import gymnasium as gym
import argparse
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gym_hybrid


ENV_ID = "Moving-v0"
DEFAULT_DEMO_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "demo_vis.gif")
MAX_FRAMES = 200


def make_action_from_params(action_id, action_params, action_space):
    """
    Reconstruct action dict from action_id and normalized action_params.
    action_params is in pmax format (padded to max dimension, normalized to [-1, 1]).
    """
    action = {"id": int(action_id)}
    
    # Initialize all parameter keys with zeros
    for key, space in action_space.spaces.items():
        if key == "id":
            continue
        if space.shape[0] == 0:
            action[key] = np.zeros((0,), dtype=space.dtype)
        else:
            action[key] = np.zeros(space.shape, dtype=space.dtype)
    
    # Denormalize and assign the relevant parameter
    param_key = f"params{action_id}"
    if param_key in action and action[param_key].shape[0] > 0:
        param_space = action_space.spaces[param_key]
        param_low = param_space.low
        param_high = param_space.high
        
        # Extract the relevant part of action_params
        param_dim = action[param_key].shape[0]
        normalized_param = action_params[:param_dim]
        
        # Denormalize from [-1, 1] to [low, high]
        if len(param_low) > 0 and len(param_high) > 0:
            denormalized = (normalized_param + 1.0) / 2.0 * (param_high - param_low) + param_low
            action[param_key] = denormalized.astype(np.float32)
        else:
            action[param_key] = normalized_param.astype(np.float32)
    
    return action


def visualize_episode(env, ids, params, start, target, render_mode='gif', output_path=None, episode_num=0):
    """
    Visualize a single episode.
    
    Args:
        env: Gym environment
        ids: Action IDs for the episode
        params: Action parameters for the episode
        start: Starting position (x, y, theta)
        target: Target position (x, y)
        render_mode: 'gif' to save as GIF, 'human' to display on screen
        output_path: Path to save GIF (only used if render_mode='gif')
        episode_num: Episode number for display
    """
    frames = [] if render_mode == 'gif' else None
    
    obs_dict, _ = env.reset()
    
    # Set initial state if available
    if start is not None and start.size >= 3:
        env.unwrapped.agent.reset(float(start[0]), float(start[1]), float(start[2]))
    if target is not None and target.size >= 2:
        env.unwrapped.target = env.unwrapped.target._replace(x=float(target[0]), y=float(target[1]))
    env.unwrapped.current_step = 0

    if render_mode == 'gif':
        frames.append(env.render())
    else:
        env.render()

    steps = min(len(ids), MAX_FRAMES)
    for i in range(steps):
        action = make_action_from_params(ids[i], params[i], env.action_space)
        obs_dict, _, terminated, truncated, _ = env.step(action)
        
        if render_mode == 'gif':
            frames.append(env.render())
        else:
            env.render()
            
        if terminated or truncated:
            print(f"Episode {episode_num} ended at step {i+1}")
            break
    
    if render_mode == 'gif' and output_path:
        import imageio.v2 as imageio
        imageio.mimsave(output_path, frames, fps=20)
        print(f"Saved episode {episode_num} to: {output_path}")
    
    return len(frames) if frames else steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", "-d", type=str, default=None, help="path to demo .npz file (default: latest expert_demos_*.npz)")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT, help="output GIF path (only used with --mode gif)")
    parser.add_argument("--episode", "-e", type=int, default=None, help="which episode to visualize (default: None for all episodes, or 0 for first)")
    parser.add_argument("--mode", "-m", type=str, default='gif', choices=['gif', 'human'], 
                        help="render mode: 'gif' to save as file, 'human' to display on screen (default: gif)")
    parser.add_argument("--all", "-a", action='store_true', help="visualize all episodes (same as --episode=-1)")
    args = parser.parse_args()

    demo_path = args.demo
    if demo_path is None:
        # Look for new format first (expert_demos_*.npz)
        files = sorted(glob.glob(os.path.join(DEFAULT_DEMO_DIR, "expert_demos_*.npz")), key=os.path.getmtime)
        if not files:
            # Fallback to old format
            files = sorted(glob.glob(os.path.join(DEFAULT_DEMO_DIR, "expert_demo_*.npz")), key=os.path.getmtime)
        if files:
            demo_path = files[-1]
        else:
            raise FileNotFoundError("No demo file found in demo dir")

    print(f"Loading demo from: {demo_path}")
    data = np.load(demo_path, allow_pickle=True)
    
    # Determine which episodes to visualize
    visualize_all = args.all or args.episode == -1
    if args.episode is None and not visualize_all:
        # Default: visualize first episode only
        episodes_to_viz = [0]
    elif visualize_all:
        episodes_to_viz = None  # Will be determined from data
    else:
        episodes_to_viz = [args.episode]
    
    # Set SDL video driver based on mode
    if args.mode == 'human':
        # Remove dummy driver to allow screen display
        if 'SDL_VIDEODRIVER' in os.environ:
            del os.environ['SDL_VIDEODRIVER']
        if 'SDL_AUDIODRIVER' in os.environ:
            del os.environ['SDL_AUDIODRIVER']
    else:
        # Ensure dummy driver for GIF mode
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    
    # Create environment with appropriate render mode
    render_mode = "human" if args.mode == "human" else "rgb_array"
    env = gym.make(ENV_ID, render_mode=render_mode)
    
    # Check if it's the new aggregated format
    if "episode_id" in data:
        # New format: aggregated episodes
        action_ids = data["action_id"]
        action_params = data["action_params"]
        episode_ids = data["episode_id"]
        meta = data.get("meta", None)
        
        max_episode = int(episode_ids.max())
        
        if episodes_to_viz is None:
            # Visualize all episodes
            episodes_to_viz = list(range(max_episode + 1))
            print(f"Visualizing all {len(episodes_to_viz)} episodes")
        
        # Validate episode numbers
        for ep_num in episodes_to_viz:
            if ep_num < 0 or ep_num > max_episode:
                raise ValueError(f"Episode {ep_num} not found. Available episodes: 0-{max_episode}")
        
        # Visualize each requested episode
        for ep_num in episodes_to_viz:
            episode_mask = episode_ids == ep_num
            if not np.any(episode_mask):
                print(f"Warning: Episode {ep_num} has no data, skipping")
                continue
            
            ids = action_ids[episode_mask]
            params = action_params[episode_mask]
            
            # Get start and target from meta
            if meta is not None and len(meta) > ep_num:
                episode_meta = meta[ep_num]
                start = episode_meta.get('start', None)
                target = episode_meta.get('target', None)
            else:
                start = None
                target = None
            
            print(f"Visualizing episode {ep_num}/{max_episode} with {len(ids)} steps")
            
            # Determine output path for this episode
            if args.mode == 'gif':
                if len(episodes_to_viz) > 1:
                    base, ext = os.path.splitext(args.output)
                    output_path = f"{base}_ep{ep_num}{ext}"
                else:
                    output_path = args.output
            else:
                output_path = None
            
            visualize_episode(env, ids, params, start, target, 
                            render_mode=args.mode, output_path=output_path, episode_num=ep_num)
    else:
        # Old format: single episode
        ids = data["action_id"]
        if "params0" in data:
            print("Warning: Using old format. Consider regenerating demos with new format.")
            raise NotImplementedError("Old format not supported. Please regenerate demos with generate_expert_demos.py")
        else:
            params = data["action_params"]
        start = data.get("start", None)
        target = data.get("target", None)
        
        print(f"Visualizing single episode with {len(ids)} steps")
        
        output_path = args.output if args.mode == 'gif' else None
        visualize_episode(env, ids, params, start, target, 
                        render_mode=args.mode, output_path=output_path, episode_num=0)

    env.close()
    
    if args.mode == 'gif':
        print(f"\n✓ Visualization complete!")
    else:
        print(f"\n✓ Display complete!")


if __name__ == "__main__":
    main()
