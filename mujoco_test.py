import mujoco
import mujoco.viewer
import time
import gymnasium as gym

# 1. Load the MuJoCo model and create data
# Walker2d
# gymnasium.make("Walker2d-v5")
# Hopper
# gymnasium.make("Hopper-v5")
# HalfCheetah
# gymnasium.make("HalfCheetah-v5")
# Humanoid
# gymnasium.make("Humanoid-v5")

# model = gym.make("Humanoid-v5").unwrapped.model
model = gym.make("HalfCheetah-v5").unwrapped.model
data = mujoco.MjData(model)

# 2. Launch the viewer and run the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
