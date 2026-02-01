import os
import sys

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import imageio.v2 as imageio
import argparse
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gym_hybrid


ENV_ID = "Moving-v0"
DEFAULT_DEMO_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "demo_vis.gif")
MAX_FRAMES = 80


def make_action(action_id, params0, params1, params2):
    return {
        "id": int(action_id),
        "params0": np.asarray(params0, dtype=np.float32),
        "params1": np.asarray(params1, dtype=np.float32),
        "params2": np.asarray(params2, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", "-d", type=str, default=None, help="path to demo .npz file (default: latest expert_demo_*.npz or expert_demo.npz)")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT, help="output GIF path")
    args = parser.parse_args()

    demo_path = args.demo
    if demo_path is None:
        files = sorted(glob.glob(os.path.join(DEFAULT_DEMO_DIR, "expert_demo_*.npz")), key=os.path.getmtime)
        if files:
            demo_path = files[-1]
        elif os.path.exists(os.path.join(DEFAULT_DEMO_DIR, "expert_demo.npz")):
            demo_path = os.path.join(DEFAULT_DEMO_DIR, "expert_demo.npz")
        else:
            raise FileNotFoundError("No demo file found in demo dir")

    data = np.load(demo_path)
    ids = data["action_id"]
    p0s = data["params0"]
    p1s = data["params1"]
    p2s = data["params2"]
    start = data.get("start", None)
    target = data.get("target", None)

    env = gym.make(ENV_ID, render_mode="rgb_array")
    frames = []

    obs, _ = env.reset()
    if start is not None and start.size:
        env.unwrapped.agent.reset(float(start[0]), float(start[1]), float(start[2]))
    if target is not None and target.size:
        env.unwrapped.target = env.unwrapped.target._replace(x=float(target[0]), y=float(target[1]))
    env.unwrapped.current_step = 0

    frames.append(env.render())

    steps = min(len(ids), MAX_FRAMES)
    for i in range(steps):
        action = make_action(ids[i], p0s[i], p1s[i], p2s[i])
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(args.output, frames, fps=20)


if __name__ == "__main__":
    main()
