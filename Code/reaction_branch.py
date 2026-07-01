import gymnasium as gym
import chemistrylab
import numpy as np
import pandas as pd

from stable_baselines3 import SAC, A2C, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import time


def evaluate_by_target(model, episodes_per_target=100):
    # get model name
    algo_name = model.__class__.__name__
    results = []

    # 7 targets
    for target_idx in range(7):

        rewards = []
        env = gym.make("GenWurtzReact-v2")
        for _ in range(episodes_per_target):
            
            obs, info = env.reset()
            # overwrite target vector
            target = env.unwrapped.targets[target_idx]
            obs = env.unwrapped._reset(target)

            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)

        results.append(
            {   
                "algo": algo_name,
                "target": target_idx,
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "episodes": episodes_per_target
            }
        )
        env.close()
    return results


def run_experiment(algo, seed, n_envs=10, total_timesteps=1000):

    algo_name = algo.__name__

    # run n_envs environment in parellel.
    env = make_vec_env(
        "GenWurtzReact-v2",
        n_envs=n_envs,
        seed=seed,
        monitor_dir=f"./monitor_logs/{algo_name}/seed_{seed}",
        vec_env_cls=SubprocVecEnv,
    )

    model = algo(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0
    )

    model.learn(
        total_timesteps=total_timesteps
    )

    results = evaluate_by_target(model)

    # add seed information
    for r in results:
        r["seed"] = seed

    return results

if __name__ == "__main__":

    all_results = []

    algos = [
        A2C,
        PPO,
        SAC,
        TD3
    ]

    for algo in algos:
        for seed in range(1):
            print("Running seed:", seed)
            start = time.time()
            result = run_experiment(algo,
                                    seed,
                                    n_envs=4,
                                    total_timesteps=1000)
            all_results.extend(result)
            end = time.time()
            
            print(f"Time: {end-start:.2f} seconds")


    df = pd.DataFrame(all_results)


    df.to_csv(
        "react_wurtz_results.csv",
        index=False
    )


    print(df)