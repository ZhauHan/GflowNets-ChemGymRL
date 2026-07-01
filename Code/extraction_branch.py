import gymnasium as gym
import chemistrylab
import numpy as np
import pandas as pd
import time
from pathlib import Path
import pickle

from stable_baselines3 import SAC, A2C, DQN, PPO 
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


def get_best_trajectory(model, episodes=100):
    best_reward = -np.inf
    best_trajectory = []

    env = gym.make("GenWurtzExtract-v2")
    # repeat since environment might not be deterministic
    for episode in range(episodes):
       
        obs, info = env.reset()
        # choose dodecane as the target to match paper result
        target = env.unwrapped.targets[0]
        # work around to reset target since step only allowed after env.reset()
        obs = env.unwrapped._reset(target)
        
        
        
        done = False
        total_reward = 0

        trajectory = []
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            next_obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append({
                "obs": obs,
                "action":action,
                "reward": reward,
                "next_obs": next_obs
            })

            total_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_trajectory = trajectory
            
    env.close()

    return best_reward, best_trajectory

def save_trajectory(algo_name, seed, trajectory):
    save_dir = (
        Path("results")
        / "trajectories"
        / algo_name
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    file = save_dir / f"seed_{seed}.pkl"

    with open(file, "wb") as f:
        pickle.dump(trajectory, f)

def run_experiment(algo, seed, n_envs=10, total_timesteps=1000):

    algo_name = algo.__name__

    # run n_envs environment in parellel.
    env = make_vec_env(
        "GenWurtzExtract-v2",
        n_envs=n_envs,
        seed=seed,
        monitor_dir=f"./monitor_logs/extract/{algo_name}/seed_{seed}",
        vec_env_cls=SubprocVecEnv,
    )

    model = algo(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0
    )

    # train
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps
    )
    end = time.time()
    print(f"Train Time: {end-start:.2f} seconds")
    
    # evaluate best trajectory and get best_reward 
    start = time.time()
    best_reward, best_trajectory = get_best_trajectory(model)
    end = time.time()
    print(f"Eval Time: {end-start:.2f} seconds")

    # save best trajectory for dodecane
    save_trajectory(algo_name, seed, best_trajectory)

    results = {
        "reward": best_reward,
        "algo": algo_name,
        "seed": seed
    }

    env.close()

    return results

if __name__ == "__main__":

    all_results = []

    algos = [
        A2C,
        DQN,
        PPO
    ]

    for algo in algos:
        for seed in range(1):
            print("Running seed:", seed)
            
            result = run_experiment(algo,
                                    seed,
                                    n_envs=2,
                                    total_timesteps=10000)
            
            all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(
        "extract_wurtz_results.csv",
        index=False
    )
    print(df)