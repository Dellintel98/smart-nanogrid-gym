import smart_nanogrid_gym
import argparse

import gym
import os

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

env_variants = [
    {
        'variant_name': 'basic-',
        'config': {
            'vehicle_to_everything': False,
            'pv_system_available_in_model': False,
            'battery_system_available_in_model': False,
            'environment_mode': 'training'
        }},
    {
        'variant_name': 'b-pv-',
        'config': {
            'vehicle_to_everything': False,
            'pv_system_available_in_model': True,
            'battery_system_available_in_model': True,
            'environment_mode': 'training'
        }},
    {
        'variant_name': 'v2x-',
        'config': {
            'vehicle_to_everything': True,
            'pv_system_available_in_model': False,
            'battery_system_available_in_model': False,
            'environment_mode': 'training'
        }},
    {
        'variant_name': 'v2x-b-pv-',
        'config': {
            'vehicle_to_everything': True,
            'pv_system_available_in_model': True,
            'battery_system_available_in_model': True,
            'environment_mode': 'training'
        }}
]
current_env = env_variants[3]
current_env_name = current_env['variant_name']

models_dir = f"models/PPO-{current_env_name}"
logdir = f"logs/PPO-{current_env_name}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

current_env_configuration = current_env['config']
env = gym.make('SmartNanogridEnv-v0', **current_env_configuration)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 20000
# TIMESTEPS = 200
start = time.time()
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

env.close()

end = time.time()
seconds = end - start
minutes = seconds / 60
hours = minutes // 60
minutes = (minutes/60 - hours) * 60

print(f'Training started: {start}\nTraining ended: {end}\nTraining lasted: {hours} h and {minutes} min')
# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_gym", env=env)
#
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
# # Enjoy trained agent
# obs = env.reset()
# for i in range(24):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     # env.render(
