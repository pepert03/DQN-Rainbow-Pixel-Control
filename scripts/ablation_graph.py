"""Ablation comparison graph for Hopper (Time vs Reward) with borders."""

import glob
import re
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

LABEL_MAP = {
    "rainbow_pixel_hopper": "Rainbow",
    "pixel_hopper": "DQN",
    "no_double_pixel_hopper": "- Double DQN",
    "no_distributional_pixel_hopper": "- Distributional",
    "no_dueling_pixel_hopper": "- Dueling",
    "no_noisy_pixel_hopper": "- NoisyNets",
    "no_n_step_pixel_hopper": "- N-step",
    "no_prioritized_pixel_hopper": "- Prioritized",
}

COLOR_MAP = {
    "Rainbow": "#e6194b",
    "DQN": "#3cb44b",
    "- Double DQN": "#4363d8",
    "- Distributional": "#f58231",
    "- Dueling": "#911eb4",
    "- NoisyNets": "#42d4f4",
    "- N-step": "#f032e6",
    "- Prioritized": "#bfef45",
}

# Regex for parsing logs
TIMESTAMP_RE = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
CHECKPOINT_RE = re.compile(
    r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Checkpoint saved at episode (\d+)"
)


def parse_episode_time_map(log_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Map episodes to training hours from log file."""
    fmt = "%m-%d %H:%M:%S"
    t0 = None
    eps_list = [0]
    hrs_list = [0.0]

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                # Find initial timestamp
                if t0 is None:
                    m = TIMESTAMP_RE.search(line)
                    if m:
                        t0 = datetime.strptime(m.group(1), fmt)
                
                # Parse checkpoint timestamps
                m = CHECKPOINT_RE.search(line)
                if m and t0 is not None:
                    ts = datetime.strptime(m.group(1), fmt)
                    ep = int(m.group(2))
                    hours = (ts - t0).total_seconds() / 3600.0
                    eps_list.append(ep)
                    hrs_list.append(hours)
    except Exception as e:
        warnings.warn(f"Could not parse {log_path}: {e}")
        return None

    if len(eps_list) < 2:
        return None
    return np.array(eps_list), np.array(hrs_list)


def rolling_mean(data: list[float], window: int = 100) -> np.ndarray:
    """Calculate rolling mean."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) < window:
        return np.convolve(arr, np.ones(len(arr)) / len(arr), mode="valid")
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def load_runs():
    """Load runs and prep data."""
    order = list(LABEL_MAP.values())
    entries = []

    for ckpt_path in sorted(glob.glob("runs/*hopper*/checkpoint.pt")):
        run_dir = ckpt_path.replace("\\", "/").split("/")[-2]
        label = LABEL_MAP.get(run_dir)
        if label is None:
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            warnings.warn(f"Could not load {ckpt_path}: {e}")
            continue

        rewards = ckpt.get("rewards_per_episode")
        if rewards is None or len(rewards) == 0:
            continue

        # Map episodes to time
        log_path = ckpt_path.replace("checkpoint.pt", "training.log")
        time_map = parse_episode_time_map(log_path)

        mean100 = rolling_mean(rewards, 100)
        episodes = np.arange(len(mean100)) + 100

        time_axis = None
        if time_map is not None:
            ep_knots, hr_knots = time_map
            time_axis = np.interp(episodes, ep_knots, hr_knots)

        color = COLOR_MAP.get(label)
        # Highlight Rainbow line width
        lw = 3.5 if label == "Rainbow" else 1.5

        entries.append({
            "label": label,
            "mean100": mean100,
            "time_axis": time_axis,
            "color": color,
            "lw": lw,
        })

    # Sort to match predefined order
    entries.sort(key=lambda x: order.index(x["label"]) if x["label"] in order else 999)
    return entries


def main():
    """Plotting function with repeating rainbow gradient and borders."""
    entries = load_runs()
    if not entries:
        print("No data found.")
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.85)

    target_time = 1.3
    rainbow_cmap = "hsv"

    for e in entries:
        if e["time_axis"] is None or len(e["time_axis"]) == 0:
            continue

        x = e["time_axis"]
        y = e["mean100"]

        if e["label"] == "Rainbow":
            # Draw black background line for border effect
            border_lw = e["lw"] + 2.0
            ax.plot(x, y, color="black", linewidth=border_lw, zorder=4)

            # Format points for segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create multi-colored line on top of black border
            lc = LineCollection(segments, cmap=rainbow_cmap, linewidth=e["lw"], zorder=5)
            
            # Cycle color spectrum every 0.5 units of time
            color_values = (x % 0.1) / 0.1
            lc.set_array(color_values)
            ax.add_collection(lc)
            
            # Label color
            text_color = "#e6194b" 
        
        elif e["label"] == "DQN":
            # Draw black background line for border effect
            border_lw = e["lw"] + 2.0
            ax.plot(x, y, color="black", linewidth=border_lw, zorder=4)
            
            # Draw actual colored line on top
            ax.plot(x, y, label=e["label"], color=e["color"], linewidth=e["lw"], zorder=5)
            text_color = e["color"]
            
        else:
            # Standard plot for other lines (lower zorder so borders don't cover them)
            ax.plot(x, y, label=e["label"], color=e["color"], linewidth=e["lw"], zorder=2)
            text_color = e["color"]

        # Add label at the end
        y_at_target = np.interp(target_time, x, y)
        ax.text(
            target_time + 0.02, y_at_target, e["label"], 
            color=text_color, fontsize=10, va="center", weight="bold",
            clip_on=False
        )

    # Set visuals
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Mean100 Reward")
    ax.set_title("Ablation: Mean100 Reward vs Time")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, target_time)
    ax.set_ylim(0, 2400)

    # Save and show
    out_path = "runs/ablation_hopper_rainbow.png"
    fig.savefig(out_path, dpi=150)
    print(f"Image saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()