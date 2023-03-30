import gym
import smart_nanogrid_gym
import numpy as np
import os
import argparse

from stable_baselines3 import DDPG, PPO
import time
import matplotlib.pyplot as plt

from smart_nanogrid_gym.utils.config import solvers_files_directory_path


class Predictor:
    def __init__(self):
        self.number_of_chargers = 2
        device = 'cuda' if self.number_of_chargers > 8 else 'cpu'

    def load_prediction_model(self):
        pass

    def configure_predictor(self):
        config_info = {
            'basic': {'vehicle_to_everything': False, 'pv_system_available_in_model': False,
                      'battery_system_available_in_model': False, 'number_of_chargers': self.number_of_chargers},
            'b-pv': {'vehicle_to_everything': False, 'pv_system_available_in_model': True,
                     'battery_system_available_in_model': True, 'number_of_chargers': self.number_of_chargers},
            'v2x': {'vehicle_to_everything': True, 'pv_system_available_in_model': False,
                    'battery_system_available_in_model': False, 'number_of_chargers': self.number_of_chargers},
            'v2x-b-pv': {'vehicle_to_everything': True, 'pv_system_available_in_model': True,
                         'battery_system_available_in_model': True, 'number_of_chargers': self.number_of_chargers}
        }

    def create_environment_for_testing_model(self):
        pass

    def predict_single_day(self, current_model, env, kwargs):
        rewards_list = []

        obs, _ = env.reset(**kwargs)
        done = False
        while not done:
            action, _states = current_model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards_list.append(reward)

        return rewards_list
