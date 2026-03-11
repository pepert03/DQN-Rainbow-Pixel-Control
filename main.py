import argparse

from src.config import load_config


def main():
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")

    args = parser.parse_args()

    config = load_config(args.hyperparameters)

    # Auto-detect whether to use Rainbow or vanilla DQN
    rainbow_flags = [
        "enable_double_dqn",
        "enable_dueling_dqn",
        "enable_prioritized_replay",
        "enable_noisy_nets",
        "enable_distributional",
        "enable_n_step",
    ]
    is_rainbow = any(config.get(flag, False) for flag in rainbow_flags)

    if is_rainbow:
        from src.rainbow.train import train, evaluate
    else:
        from src.dqn.train import train, evaluate

    if args.train:
        train(args.hyperparameters)
    else:
        evaluate(args.hyperparameters, render=True)


if __name__ == "__main__":
    main()
