import traceback, sys

try:
    from src.rainbow.train import RainbowAgent

    a = RainbowAgent("rainbow_states_walker2d")
    print(f"num_envs={a.num_envs}", flush=True)
    a.run(is_training=True)
except KeyboardInterrupt:
    print("Stopped")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
