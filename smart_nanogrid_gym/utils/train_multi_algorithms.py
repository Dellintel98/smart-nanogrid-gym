import numpy as np

import smart_nanogrid_gym
import argparse

import gym
import os

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import time


def set_environment_variants(amount_of_chargers, algorithm=''):
    environment_variants = [
        {
            'variant_name': 'basic',
            'config': {
                'vehicle_to_everything': False,
                'pv_system_available_in_model': False,
                'battery_system_available_in_model': False,
                'environment_mode': 'training',
                'algorithm_used': f'{algorithm.upper()}',
                'number_of_chargers': amount_of_chargers
            }},
        {
            'variant_name': 'b-pv',
            'config': {
                'vehicle_to_everything': False,
                'pv_system_available_in_model': True,
                'battery_system_available_in_model': True,
                'environment_mode': 'training',
                'algorithm_used': f'{algorithm.upper()}',
                'number_of_chargers': amount_of_chargers
            }},
        {
            'variant_name': 'v2x',
            'config': {
                'vehicle_to_everything': True,
                'pv_system_available_in_model': False,
                'battery_system_available_in_model': False,
                'environment_mode': 'training',
                'algorithm_used': f'{algorithm.upper()}',
                'number_of_chargers': amount_of_chargers
            }},
        {
            'variant_name': 'v2x-b-pv',
            'config': {
                'vehicle_to_everything': True,
                'pv_system_available_in_model': True,
                'battery_system_available_in_model': True,
                'environment_mode': 'training',
                'algorithm_used': f'{algorithm.upper()}',
                'number_of_chargers': amount_of_chargers
            }}
    ]

    return environment_variants


def train_model(epochs, total_timesteps):
    for epoch in range(epochs):
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name="DDPG")
        model.save(f"{models_dir}/{total_timesteps * epoch}")
    env.close()


def time_training(start, end):
    seconds = end - start
    minutes = seconds / 60
    hours = minutes // 60
    minutes = (minutes / 60 - hours) * 60

    print(f'Training started: {start}\nTraining ended: {end}\nTraining lasted: {hours} h and {minutes} min')


# Todo: Add argument to set reward model to below choice through env parameters or reset method
model_rewards_to_train = ['departure-reward', 'no-reward', 'sparse-reward', 'dense-reward']
number_of_chargers = 4
algorithm = 'DDPG'
env_variants = set_environment_variants(amount_of_chargers=number_of_chargers, algorithm=algorithm)

current_env = env_variants[0]
current_env_name = current_env['variant_name']

device = 'cuda' if number_of_chargers > 8 else 'cpu'

number_of_episodes = 850
timesteps_per_episode = 24
timesteps = number_of_episodes * timesteps_per_episode
training_epochs = 50

start_time = time.time()

for reward_model in model_rewards_to_train:
    s1 = time.time()
    model_name = f'{algorithm}-{current_env_name}-{number_of_chargers}-{reward_model}'
    models_dir = f"models/{model_name}"
    logdir = f"logs/{model_name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    current_env_configuration = current_env['config']
    env = gym.make('SmartNanogridEnv-v0', **current_env_configuration)
    check_env(env)

    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log=logdir, device=device)

    s2 = time.time()
    train_model(training_epochs, timesteps)
    e2 = time.time()

    time_training(s2, e2)
    time_training(s1, e2)

end_time = time.time()
time_training(start_time, end_time)


