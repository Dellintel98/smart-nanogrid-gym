import gym
import smart_nanogrid_gym
import numpy as np
import os
import argparse

from stable_baselines3 import DDPG, PPO
import time
import matplotlib.pyplot as plt

from smart_nanogrid_gym.utils.config import solvers_files_directory_path


def predict_single_day(current_model, env, kwargs):
    rewards_list = []

    obs, _ = env.reset(**kwargs)
    done = False
    while not done:
        action, _states = current_model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards_list.append(reward)

    return rewards_list


config_info = {
    'basic': {'vehicle_to_everything': False, 'pv_system_available_in_model': False, 'battery_system_available_in_model': False},
    'b-pv': {'vehicle_to_everything': False, 'pv_system_available_in_model': True, 'battery_system_available_in_model': True},
    'v2x': {'vehicle_to_everything': True, 'pv_system_available_in_model': False, 'battery_system_available_in_model': False},
    'v2x-b-pv': {'vehicle_to_everything': True, 'pv_system_available_in_model': True, 'battery_system_available_in_model': True}
}

envs = {
    'basic': gym.make('SmartNanogridEnv-v0', **config_info['basic']),
    'b-pv': gym.make('SmartNanogridEnv-v0', **config_info['b-pv']),
    'v2x': gym.make('SmartNanogridEnv-v0', **config_info['v2x']),
    'v2x-b-pv': gym.make('SmartNanogridEnv-v0', **config_info['v2x-b-pv'])
}

names = os.listdir('RL\\models')
names = [name for name in names if name != '.gitignore']

models = []
for name in names:
    model_dir = f"{solvers_files_directory_path}\\RL\\models\\{name}"
    model_path = f"{model_dir}\\980000"

    lowercase_name = name.lower()
    if 'v2x-b-pv' in lowercase_name:
        env_variant_name = 'v2x-b-pv'
    elif 'v2x' in lowercase_name:
        env_variant_name = 'v2x'
    elif 'b-pv' in lowercase_name:
        env_variant_name = 'b-pv'
    elif 'basic' in lowercase_name:
        env_variant_name = 'basic'
    else:
        raise ValueError(f"{name} should be a variant of a nanogrid model and have it specified in it's file name, "
                         f"i.e. should be one of the following: [basic, b-pv, v2x, v2x-b-pv], but it is not!")
    current_env = envs[env_variant_name]

    uppercase_name = name.upper()
    if 'DDPG' in uppercase_name:
        new_model = DDPG.load(model_path, env=current_env)
        models.append({'name': name, 'model': new_model, 'env_name': env_variant_name, 'info': {'algorithm': 'DDPG'}})
    elif 'PPO' in uppercase_name:
        new_model = PPO.load(model_path, env=current_env)
        models.append({'name': name, 'model': new_model, 'env_name': env_variant_name, 'info': {'algorithm': 'PPO'}})
    else:
        raise ValueError(f"{name} nanogrid model variant should in it's name have specified which algorithm it used "
                         f"during model training, e.g. DDPG or PPO or ddpg or Ppo, etc. "
                         f"Currently accepted algorithms are: DDPG and PPO!")

final_rewards = {}
mean_rewards = {}
for name in names:
    final_rewards[name] = 0
    mean_rewards[name] = 0

reset_config = {'generate_new_initial_values': False}
# Todo: Feat: Specify reset path to initial_values for prediction
for model in models:
    env_variant_name = model['env_name']
    reset_config['algorithm_used'] = model['info']['algorithm']
    reset_config['environment_mode'] = 'prediction'
    rewards = predict_single_day(model['model'], envs[env_variant_name], reset_config)

    model_name = model['name']
    final_rewards[model_name] = sum(rewards)

for name in names:
    mean_rewards[name] = np.mean(final_rewards[name])

envs['basic'].close()
envs['b-pv'].close()
envs['v2x'].close()
envs['v2x-b-pv'].close()

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams.update({'font.size': 18})

for i, name in enumerate(names):
    plt.bar(i, final_rewards[name])

plt.xlabel('Prediction model')
plt.ylabel('Total reward')

plt.legend([name for name in names])
plt.grid()

file_time = time.time()
# plt.savefig(f"saved_figures\\figure_final_rewards_{int(file_time)}.png")
# plt.savefig(f"saved_figures\\figure_final_rewards_{int(file_time)}.png", dpi=300)

plt.show()
