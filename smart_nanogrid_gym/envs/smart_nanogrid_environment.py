import numpy as np
import os
import sys
import gym
import pathlib
from gym import spaces
from gym.utils import seeding
from scipy.io import loadmat, savemat
from smart_nanogrid_gym.utils import energy_calculations
from smart_nanogrid_gym.utils import station_simulation
from smart_nanogrid_gym.utils import initial_values_generator
from smart_nanogrid_gym.utils import actions_simulation
import time


class SmartNanogridEnv(gym.Env):
    def __init__(self, price_model=1, pv_system_available_in_model=1):
        self.NUMBER_OF_CHARGERS = 10
        self.NUMBER_OF_DAYS_TO_PREDICT = 1
        self.NUMBER_OF_HOURS_AHEAD = 3
        self.CURRENT_PRICE_MODEL = price_model
        self.PV_SYSTEM_AVAILABLE_IN_MODEL = pv_system_available_in_model

        self.EV_PARAMETERS = {
            'CAPACITY': 30,
            'CHARGING EFFICIENCY': 0.91,
            'DISCHARGING EFFICIENCY': 0.91,
            'MAX CHARGING POWER': 11,
            'MAX DISCHARGING POWER': 11
        }

        self.BATTERY_PARAMETERS = {
            'CAPACITY': 20,
            'CHARGING EFFICIENCY': 0.91,
            'DISCHARGING EFFICIENCY': 0.91,
            'MAX CHARGING POWER': 11,
            'MAX DISCHARGING POWER': 11
        }

        self.PV_SYSTEM_PARAMETERS = {
            'LENGTH IN METERS': 2.279,
            'WIDTH IN METERS': 1.134,
            'DEPTH IN MILLIMETERS': 20,
            'TOTAL DIMENSIONS': 2.279 * 1.134 * 20,
            'EFFICIENCY': 0.21
        }

        self.timestep, self.day = None, None
        self.initial_simulation_values = None
        self.info = None

        self.energy, self.ev_state_of_charge = None, None
        self.departing_vehicles = None
        self.grid_energy_per_timestep, self.renewable_energy_utilization_per_timestep = None, None
        self.total_cost_per_timestep, self.penalty_per_timestep = None, None

        self.simulated_single_day = False
        self.file_directory_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')) + '\\files\\'

        amount_of_observed_variables = 2
        number_of_observed_charger_values = 2

        amount_of_charger_predictions = self.NUMBER_OF_CHARGERS * number_of_observed_charger_values
        amount_of_states = amount_of_observed_variables + (self.NUMBER_OF_HOURS_AHEAD * amount_of_observed_variables)

        self.total_amount_of_states = amount_of_states + amount_of_charger_predictions

        low = np.array(np.zeros(self.total_amount_of_states), dtype=np.float32)
        high = np.array(np.ones(self.total_amount_of_states), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1,
            high=1, shape=(self.NUMBER_OF_CHARGERS,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def step(self, actions):
        results = actions_simulation.simulate_central_management_system(self, actions)

        self.total_cost_per_timestep.append(results['Total cost'])
        self.grid_energy_per_timestep.append(results['Grid energy'])
        self.renewable_energy_utilization_per_timestep.append(results['Utilized renewable energy'])
        self.penalty_per_timestep.append(results['Insufficiently charged vehicles penalty'])
        self.ev_state_of_charge = results['EV state of charge']

        self.timestep = self.timestep + 1
        observations = self.__get_observations()

        if self.timestep == 24:
            self.simulated_single_day = True
            self.timestep = 0
            prediction_results = {
                'SOC': self.ev_state_of_charge,
                'Grid energy': self.grid_energy_per_timestep,
                'Utilized renewable energy': self.renewable_energy_utilization_per_timestep,
                'Penalties': self.penalty_per_timestep,
                'Available renewable energy': self.energy['Renewable'],
                'Total cost': self.total_cost_per_timestep
            }
            savemat(self.file_directory_path + '\\prediction_results.mat', {'Prediction results': prediction_results})

        reward = -results['Total cost']
        self.info = {}

        return observations, reward, self.simulated_single_day, self.info

    def reset(self, generate_new_initial_values=True):
        self.timestep = 0
        self.day = 1
        self.simulated_single_day = False

        self.energy = energy_calculations.get_energy(self)

        self.__set_initial_simulation_values(generate_new_initial_values)
        self.__reset_variables_after_completed_day()

        return self.__get_observations()

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def close(self):
        # return 0
        pass

    def __set_initial_simulation_values(self, generate_new_initial_values):
        if generate_new_initial_values:
            self.initial_simulation_values = initial_values_generator.generate_new_values(self.file_directory_path,
                                                                                          self.NUMBER_OF_CHARGERS)
        else:
            self.initial_simulation_values = initial_values_generator.load_initial_values(self.file_directory_path,
                                                                                          self.NUMBER_OF_CHARGERS)

    def __reset_variables_after_completed_day(self):
        if self.timestep != 0:
            raise ValueError('Value of timestep attribute should be 0 during configuration at the program start or '
                             'when trying to reset simulation variables after a completed single day simulation')

        self.total_cost_per_timestep = []
        self.grid_energy_per_timestep = []
        self.renewable_energy_utilization_per_timestep = []
        self.penalty_per_timestep = []
        self.ev_state_of_charge = self.initial_simulation_values["SOC"]

    def __get_observations(self):
        [self.departing_vehicles, departure_times, vehicles_state_of_charge] = station_simulation.simulate_ev_charging_station(self)

        normalized_disturbances_observation_at_current_timestep = np.array([
            self.energy["Radiation"][0, self.timestep] / 1000,
            self.energy["Price"][0, self.timestep] / 0.1
        ])

        min_timesteps_ahead = self.timestep + 1
        max_timesteps_ahead = min_timesteps_ahead + self.NUMBER_OF_HOURS_AHEAD

        normalized_predictions = np.concatenate((
            np.array([self.energy["Radiation"][0, min_timesteps_ahead:max_timesteps_ahead] / 1000]),
            np.array([self.energy["Price"][0, min_timesteps_ahead:max_timesteps_ahead] / 0.1])),
            axis=None
        )

        normalized_states = np.concatenate((
            np.array(vehicles_state_of_charge),
            np.array(departure_times)/24),
            axis=None
        )

        observations = np.concatenate((
            normalized_disturbances_observation_at_current_timestep,
            normalized_predictions,
            normalized_states),
            axis=None
        )

        return observations


