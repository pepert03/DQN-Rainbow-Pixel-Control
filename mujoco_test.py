import argparse
import os
import sys

import numpy as np
import torch
import yaml


def _add_src_to_path() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _load_env_id_from_config(hyperparameter_set: str) -> str | None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "hyperparameters.yml"
    )
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        all_config = yaml.safe_load(f) or {}
    cfg = all_config.get(hyperparameter_set)
    if not isinstance(cfg, dict):
        return None
    return cfg.get("env_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained MuJoCo policy with human rendering (infinite episodes).\n"
            "Example: uv run python mujoco_test.py walker2d"
        )
    )
    parser.add_argument(
        "x",
        help=(
            "Model name or path. If you pass 'walker2d', it loads runs/walker2d.pt. "
            "If you pass a path ending with .pt, it loads that exact file."
        ),
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help=(
            "Gymnasium env id (overrides configs/hyperparameters.yml). "
            "If omitted, tries to read env_id from config using x as key; otherwise defaults to Walker2d-v5."
        ),
    )
    args = parser.parse_args()

    _add_src_to_path()
    from dqn import DQN  # noqa: E402
    from wrappers import make_env  # noqa: E402

    # Resolve model path
    if (
        args.x.lower().endswith(".pt")
        or os.path.sep in args.x
        or os.path.exists(args.x)
    ):
        model_path = args.x
    else:
        model_path = os.path.join(os.path.dirname(__file__), "runs", f"{args.x}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Resolve env id
    env_id = args.env_id or _load_env_id_from_config(args.x) or "Walker2d-v5"

    # Create env with the same wrappers as training, but enable human rendering.
    env = make_env(env_id, render=True, seed=42)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DQN(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    print(f"Loaded: {model_path}")
    print(f"Env: {env_id}")
    print(f"Device: {device}")

    episode = 0
    try:
        while True:
            obs, _ = env.reset()
            terminated = truncated = False
            ep_return = 0.0
            steps = 0

            while not (terminated or truncated):
                # IMPORTANT: HumanRendering needs env.render() every step.
                env.render()

                with torch.no_grad():
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action = policy(obs_tensor).squeeze(0).argmax().item()

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                steps += 1

            episode += 1
            print(f"Episode {episode}: return={ep_return:.2f} steps={steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
