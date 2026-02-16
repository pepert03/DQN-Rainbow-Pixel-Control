import gymnasium as gym


env = gym.make("HalfCheetah-v5", render_mode="human")

try:
    observation, info = env.reset()

    for _ in range(1000):
        # Sample from the action space
        action = env.action_space.sample()
        # Take a step using the sampled action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
finally:
    env.close()
