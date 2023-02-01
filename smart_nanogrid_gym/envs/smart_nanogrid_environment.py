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

        low = np.array(np.zeros(8 + 2 * self.NUMBER_OF_CHARGERS), dtype=np.float32)
        high = np.array(np.ones(8 + 2 * self.NUMBER_OF_CHARGERS), dtype=np.float32)
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

        if generate_new_initial_values:
            self.initial_simulation_values = initial_values_generator.generate_new_values(self)
            savemat(self.file_directory_path + '\\initial_values.mat', self.initial_simulation_values)
        else:
            initial_values = loadmat(self.file_directory_path + '\\initial_values.mat')

            arrival_times = initial_values['Arrivals']
            departure_times = initial_values['Departures']

            self.initial_simulation_values = {
                'SOC': initial_values['SOC'],
                'Arrivals': [],
                'Departures': [],
                'Total vehicles charging': initial_values['Total vehicles charging'],
                'Charger occupancy': initial_values['Charger occupancy']
            }

            for charger in range(self.NUMBER_OF_CHARGERS):
                if arrival_times.shape == (1, 10):
                    arrivals = arrival_times[0][charger][0]
                    departures = departure_times[0][charger][0]
                elif arrival_times.shape == (10, 3):
                    arrivals = arrival_times[charger]
                    departures = departure_times[charger]
                else:
                    raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

                self.initial_simulation_values['Arrivals'].append(arrivals.tolist())
                self.initial_simulation_values['Departures'].append(departures.tolist())

        return self.__get_observations()

    def __get_observations(self):
        if self.timestep == 0:
            self.total_cost_per_timestep = []
            self.grid_energy_per_timestep = []
            self.renewable_energy_utilization_per_timestep = []
            self.penalty_per_timestep = []
            self.ev_state_of_charge = self.initial_simulation_values["SOC"]

        [self.departing_vehicles, departure_times, vehicles_state_of_charge] = station_simulation.simulate_ev_charging_station(self)

        disturbances_observation_at_current_timestep = np.array([
            self.energy["Radiation"][0, self.timestep] / 1000,
            self.energy["Price"][0, self.timestep] / 0.1
        ])

        predictions = np.concatenate((
            np.array([self.energy["Radiation"][0, self.timestep + 1:self.timestep + 4] / 1000]),
            np.array([self.energy["Price"][0, self.timestep + 1:self.timestep + 4] / 0.1])),
            axis=None
        )

        states = np.concatenate((
            np.array(vehicles_state_of_charge),
            np.array(departure_times)/24),
            axis=None
        )

        observations = np.concatenate((
            disturbances_observation_at_current_timestep,
            predictions,
            states),
            axis=None
        )

        return observations

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def close(self):
        # return 0
        pass
