"""Interactive ablation comparison graphs for Hopper with toggleable runs."""

import glob
import re
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
import torch

LABEL_MAP = {
    "rainbow_pixel_hopper": "Rainbow (full)",
    "pixel_hopper": "Vanilla DQN",
    "no_double_pixel_hopper": "- Double DQN",
    "no_distributional_pixel_hopper": "- Distributional",
    "no_dueling_pixel_hopper": "- Dueling",
    "no_noisy_pixel_hopper": "- NoisyNets",
    "no_n_step_pixel_hopper2": "- N-step",
    "no_prioritized_pixel_hopper": "- Prioritized",
}

COLOR_MAP = {
    "Rainbow (full)": "#e6194b",
    "Vanilla DQN": "#3cb44b",
    "- Double DQN": "#4363d8",
    "- Distributional": "#f58231",
    "- Dueling": "#911eb4",
    "- NoisyNets": "#42d4f4",
    "- N-step": "#f032e6",
    "- Prioritized": "#bfef45",
}

TIMESTAMP_RE = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
CHECKPOINT_RE = re.compile(
    r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Checkpoint saved at episode (\d+)"
)


def parse_episode_time_map(log_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse training.log and return (episodes, hours) arrays from checkpoint lines."""
    fmt = "%m-%d %H:%M:%S"
    t0 = None
    eps_list = [0]
    hrs_list = [0.0]

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if t0 is None:
                    m = TIMESTAMP_RE.search(line)
                    if m:
                        t0 = datetime.strptime(m.group(1), fmt)
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
    arr = np.array(data, dtype=np.float64)
    if len(arr) < window:
        return np.convolve(arr, np.ones(len(arr)) / len(arr), mode="valid")
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def load_runs():
    """Load all hopper runs and return list of dicts with plot data."""
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
            warnings.warn(f"No rewards_per_episode in {ckpt_path}")
            continue

        log_path = ckpt_path.replace("checkpoint.pt", "training.log")
        time_map = parse_episode_time_map(log_path)

        mean100 = rolling_mean(rewards, 100)
        episodes = np.arange(len(mean100)) + 100

        time_axis = None
        if time_map is not None:
            ep_knots, hr_knots = time_map
            time_axis = np.interp(episodes, ep_knots, hr_knots)

        color = COLOR_MAP.get(label)
        lw = 2.5 if label == "Rainbow (full)" else 1.5

        entries.append({
            "label": label,
            "episodes": episodes,
            "mean100": mean100,
            "time_axis": time_axis,
            "color": color,
            "lw": lw,
        })

    entries.sort(key=lambda x: order.index(x["label"]) if x["label"] in order else 999)
    return entries


def main():
    entries = load_runs()
    if not entries:
        print("No checkpoint.pt found in runs/*hopper*/. Nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.18)

    # Draw all lines and keep references
    lines_ep = []
    lines_time = []
    labels = []

    for e in entries:
        line_ep, = ax1.plot(
            e["episodes"], e["mean100"],
            label=e["label"], color=e["color"], linewidth=e["lw"],
        )
        lines_ep.append(line_ep)

        if e["time_axis"] is not None:
            line_t, = ax2.plot(
                e["time_axis"], e["mean100"],
                label=e["label"], color=e["color"], linewidth=e["lw"],
            )
        else:
            # Invisible placeholder so indices stay aligned
            line_t, = ax2.plot([], [], color=e["color"], linewidth=e["lw"])
        lines_time.append(line_t)

        labels.append(e["label"])

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean100 Reward")
    ax1.set_title("Ablation: Mean100 Reward vs Episode")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Mean100 Reward")
    ax2.set_title("Ablation: Mean100 Reward vs Time")
    ax2.grid(True, alpha=0.3)

    # Checkboxes panel on the left
    colors = [e["color"] for e in entries]
    initial_state = [True] * len(labels)

    ax_check = fig.add_axes([0.01, 0.15, 0.13, 0.7])
    ax_check.set_frame_on(False)
    check = CheckButtons(ax_check, labels, initial_state)

    # Style checkbox labels with matching colors
    for i, lbl_text in enumerate(check.labels):
        lbl_text.set_color(colors[i])
        lbl_text.set_fontsize(9)

    def toggle(label):
        idx = labels.index(label)
        visible = not lines_ep[idx].get_visible()
        lines_ep[idx].set_visible(visible)
        lines_time[idx].set_visible(visible)
        fig.canvas.draw_idle()

    check.on_clicked(toggle)

    # Save static version too
    out_path = "runs/ablation_hopper.png"
    fig.savefig(out_path, dpi=150)
    print(f"Static image saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
