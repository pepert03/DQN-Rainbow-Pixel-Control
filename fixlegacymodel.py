import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any

import torch
import yaml


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# Allow importing from ./src
_SRC_DIR = os.path.join(_repo_root(), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dqn import DQN, Pixel_DQN  # noqa: E402  # type: ignore[import-not-found]
from wrappers import make_env  # noqa: E402  # type: ignore[import-not-found]


def _torch_load_any(path: str) -> Any:
    """Load *anything* a legacy .pt might contain.

    - If the file is a plain state_dict, this is safe and works.
    - If the file contains a pickled nn.Module, this requires unpickling.
      We try weights_only=False on PyTorch 2.6+.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    # Heuristic: state_dict keys are usually strings, values are tensors.
    k0 = next(iter(obj.keys()))
    v0 = next(iter(obj.values()))
    return isinstance(k0, str) and torch.is_tensor(v0)


def _safe_torch_save(obj: Any, path: str, retries: int = 6) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_exc: Exception | None = None

    for attempt in range(retries):
        tmp_path = f"{path}.tmp.{os.getpid()}.{attempt}"
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as exc:
            last_exc = exc
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            time.sleep(0.15 * (2**attempt))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_path = f"{path}.{ts}"
    torch.save(obj, fallback_path)
    print(f"WARNING: Could not overwrite {path}; saved to {fallback_path} instead.")
    if last_exc is not None:
        print(f"Last save error: {last_exc}")


def _load_preset_config(preset: str) -> dict:
    runs_cfg = os.path.join(_repo_root(), "runs", preset, "config.yml")
    if os.path.exists(runs_cfg):
        with open(runs_cfg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config_path = os.path.join(_repo_root(), "configs", "hyperparameters.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)

    if preset not in all_cfg:
        available = ", ".join(sorted(all_cfg.keys()))
        raise KeyError(f"Unknown preset '{preset}'. Available presets: {available}")

    return all_cfg[preset]


def _build_model(env_id: str, obs_type: str) -> tuple[torch.nn.Module, int, int]:
    env = make_env(env_id, obs_type, render=False)
    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    finally:
        env.close()

    if obs_type == "pixel":
        model = Pixel_DQN(state_dim, action_dim)
    else:
        model = DQN(state_dim, action_dim)

    return model, state_dim, action_dim


def _extract_policy_state_dict(legacy_obj: Any) -> dict:
    """Accept common legacy formats and return a policy state_dict."""
    if isinstance(legacy_obj, dict) and "model_state_dict" in legacy_obj:
        msd = legacy_obj["model_state_dict"]
        if not isinstance(msd, dict):
            raise TypeError("legacy['model_state_dict'] is not a dict")
        return msd

    if _looks_like_state_dict(legacy_obj):
        return legacy_obj

    if hasattr(legacy_obj, "state_dict"):
        return legacy_obj.state_dict()

    raise TypeError(
        "Unsupported legacy file format. Expected a state_dict dict, a dict with "
        "'model_state_dict', or a saved nn.Module."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy policy-only model.pt into the files expected by the current agent.py. "
            "Writes runs/<preset>/best_model.pt (and optionally a minimal checkpoint.pt)."
        )
    )
    parser.add_argument(
        "preset", help="Hyperparameter preset name (e.g., walker2d, humanoid, cartpole)"
    )
    parser.add_argument(
        "--legacy",
        default=None,
        help=(
            "Path to legacy model file (.pt). If omitted, tries runs/<preset>/model.pt then runs/<preset>/best_model.pt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: runs/<preset>",
    )
    parser.add_argument(
        "--also-checkpoint",
        action="store_true",
        help="Also write runs/<preset>/checkpoint.pt with minimal metadata and an empty replay buffer.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Load weights with strict=False (useful only if you know shapes roughly match).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing best_model.pt/checkpoint.pt.",
    )

    args = parser.parse_args()

    cfg = _load_preset_config(args.preset)
    env_id = cfg["env_id"]
    obs_type = cfg["obs"]

    out_dir = args.out_dir or os.path.join(_repo_root(), "runs", args.preset)
    os.makedirs(out_dir, exist_ok=True)

    legacy_path = args.legacy
    if legacy_path is None:
        c1 = os.path.join(out_dir, "model.pt")
        c2 = os.path.join(out_dir, "best_model.pt")
        legacy_path = c1 if os.path.exists(c1) else c2

    if legacy_path is None or not os.path.exists(legacy_path):
        raise FileNotFoundError(
            f"Legacy model not found. Provide --legacy PATH. Tried: {legacy_path}"
        )

    print(f"Preset: {args.preset} | env_id={env_id} | obs={obs_type}")
    print(f"Loading legacy model: {legacy_path}")

    model, state_dim, action_dim = _build_model(env_id, obs_type)
    print(f"Built model for state_dim={state_dim} action_dim={action_dim}")

    legacy_obj = _torch_load_any(legacy_path)
    state_dict = _extract_policy_state_dict(legacy_obj)

    strict = not args.non_strict
    load_result = model.load_state_dict(state_dict, strict=strict)
    if (not strict) and (load_result.missing_keys or load_result.unexpected_keys):
        print("WARNING: Non-strict load had mismatches:")
        if load_result.missing_keys:
            print(f"  missing_keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  unexpected_keys: {load_result.unexpected_keys}")

    best_model_path = os.path.join(out_dir, "best_model.pt")
    if (not args.overwrite) and os.path.exists(best_model_path):
        raise FileExistsError(
            f"{best_model_path} already exists. Use --overwrite to replace it."
        )

    _safe_torch_save(model.state_dict(), best_model_path)
    print(f"Wrote: {best_model_path}")

    if args.also_checkpoint:
        checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
        if (not args.overwrite) and os.path.exists(checkpoint_path):
            raise FileExistsError(
                f"{checkpoint_path} already exists. Use --overwrite to replace it."
            )

        # Minimal checkpoint that current agent.py can load without custom pickles.
        # NOTE: some versions of agent.py expect optimizer_state_dict to exist.
        learning_rate = float(cfg.get("learning_rate", 1e-4))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "target_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_buffer": {
                "capacity": int(cfg.get("replay_memory_size", 10000)),
                "data": [],
            },
            "epsilon": float(cfg.get("epsilon_init", 1.0)),
            "episode": 0,
            "best_reward": float("-inf"),
            "rewards_per_episode": [],
            "epsilon_history": [],
            "step_count": 0,
        }
        _safe_torch_save(checkpoint, checkpoint_path)
        print(f"Wrote: {checkpoint_path}")

    print("Done.")
    print(f"You can now evaluate with: uv run python .\\src\\agent.py {args.preset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
