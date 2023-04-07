import smart_nanogrid_gym

import gym
import os

from stable_baselines3 import PPO
import time

number_of_chargers = 4
vehicle_charging_modes = ['controlled', 'bounded']
# charging modes: controlled [with remaining capacity and soc], bounded [-1, 1]*max_power;
#        {add later maybe: max_bounded[-22, 22], unbounded[-inf, inf]}
charging_mode = vehicle_charging_modes[0]

vehicle_uncharged_penalty_modes = ['no_penalty', 'on_departure', 'sparse', 'dense']
# penalty modes: no_penalty, on_departure, sparse, dense
penalty_mode = vehicle_uncharged_penalty_modes[3]

time_intervals = ['15min', '30min', '45min', '1h', '2h']
requested_time_interval = time_intervals[3]

env_variants = [
    {
        'variant_name': 'basic',
        'config': {
            'vehicle_to_everything': False,
            'pv_system_available_in_model': False,
            'battery_system_available_in_model': False,
            'environment_mode': 'training',
            'algorithm_used': 'PPO',
            'number_of_chargers': number_of_chargers,
            'charging_mode': charging_mode,
            'vehicle_uncharged_penalty_mode': penalty_mode,
            'time_interval': requested_time_interval
        }},
    {
        'variant_name': 'b-pv',
        'config': {
            'vehicle_to_everything': False,
            'pv_system_available_in_model': True,
            'battery_system_available_in_model': True,
            'environment_mode': 'training',
            'algorithm_used': 'PPO',
            'number_of_chargers': number_of_chargers,
            'charging_mode': charging_mode,
            'vehicle_uncharged_penalty_mode': penalty_mode,
            'time_interval': requested_time_interval
        }},
    {
        'variant_name': 'v2x',
        'config': {
            'vehicle_to_everything': True,
            'pv_system_available_in_model': False,
            'battery_system_available_in_model': False,
            'environment_mode': 'training',
            'algorithm_used': 'PPO',
            'number_of_chargers': number_of_chargers,
            'charging_mode': charging_mode,
            'vehicle_uncharged_penalty_mode': penalty_mode,
            'time_interval': requested_time_interval
        }},
    {
        'variant_name': 'v2x-b-pv',
        'config': {
            'vehicle_to_everything': True,
            'pv_system_available_in_model': True,
            'battery_system_available_in_model': True,
            'environment_mode': 'training',
            'algorithm_used': 'PPO',
            'number_of_chargers': number_of_chargers,
            'charging_mode': charging_mode,
            'vehicle_uncharged_penalty_mode': penalty_mode,
            'time_interval': requested_time_interval
        }}
]
current_env = env_variants[1]
current_env_name = current_env['variant_name']

models_dir = f"models/PPO-{current_env_name}-{charging_mode}-{penalty_mode}-{number_of_chargers}ch-{requested_time_interval}"
logdir = f"logs/PPO-{current_env_name}-{charging_mode}-{penalty_mode}-{number_of_chargers}ch-{requested_time_interval}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

current_env_configuration = current_env['config']
env = gym.make('SmartNanogridEnv-v0', **current_env_configuration)

device = 'cuda' if number_of_chargers > 8 else 'cpu'
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device=device)

number_of_episodes = 850
timesteps_per_episode = 24
timesteps = number_of_episodes * timesteps_per_episode
training_epochs = 50

start = time.time()
for epoch in range(training_epochs):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{timesteps * epoch}")

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
