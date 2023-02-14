import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from scipy.io import savemat
import time

from smart_nanogrid_gym.utils.central_management_system import CentralManagementSystem
from smart_nanogrid_gym.utils.charging_station import ChargingStation
from smart_nanogrid_gym.utils.pv_system_manager import PVSystemManager
from ..utils.config import data_files_directory_path


class SmartNanogridEnv(gym.Env):
    def __init__(self, price_model=1, pv_system_available_in_model=True):
        self.NUMBER_OF_CHARGERS = 10
        self.NUMBER_OF_DAYS_TO_PREDICT = 1
        self.NUMBER_OF_HOURS_AHEAD = 3
        self.NUMBER_OF_DAYS_AHEAD = 1
        self.CURRENT_PRICE_MODEL = price_model
        self.PV_SYSTEM_AVAILABLE_IN_MODEL = pv_system_available_in_model

        self.charging_station = ChargingStation(self.NUMBER_OF_CHARGERS)
        self.central_management_system = CentralManagementSystem()
        if pv_system_available_in_model:
            self.pv_system_manager = PVSystemManager(self.NUMBER_OF_DAYS_TO_PREDICT, self.NUMBER_OF_DAYS_AHEAD)

        self.timestep = None
        self.info = None

        self.energy = None
        self.energy_price = None
        self.grid_energy_per_timestep, self.renewable_energy_utilization_per_timestep = None, None
        self.total_cost_per_timestep, self.penalty_per_timestep = None, None

        self.simulated_single_day = False

        amount_of_observed_variables = 1 + int(pv_system_available_in_model)
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
        total_charging_power = self.charging_station.simulate_vehicle_charging(actions, self.timestep)
        results = self.central_management_system.simulate(self.timestep, total_charging_power, self.energy,
                                                          self.energy_price, self.charging_station.departing_vehicles,
                                                          self.charging_station.vehicle_state_of_charge,
                                                          self.PV_SYSTEM_AVAILABLE_IN_MODEL)

        self.total_cost_per_timestep.append(results['Total cost'])
        self.grid_energy_per_timestep.append(results['Grid energy'])
        self.renewable_energy_utilization_per_timestep.append(results['Utilized renewable energy'])
        self.penalty_per_timestep.append(results['Insufficiently charged vehicles penalty'])
        self.charging_station.vehicle_state_of_charge = results['EV state of charge']

        self.timestep = self.timestep + 1
        observations = self.__get_observations()

        self.simulated_single_day = self.__check_is_single_day_simulated()
        if self.simulated_single_day:
            self.timestep = 0
            self.__save_prediction_results()

        reward = -results['Total cost']
        self.info = {}

        return observations, reward, self.simulated_single_day, self.info

    def __get_observations(self):
        [departure_times, vehicles_state_of_charge] = self.charging_station.simulate(self.timestep)

        min_timesteps_ahead = self.timestep + 1
        max_timesteps_ahead = min_timesteps_ahead + self.NUMBER_OF_HOURS_AHEAD

        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            normalized_disturbances_observation_at_current_timestep = np.array([
                self.energy["Solar radiation"][0, self.timestep] / 1000,
                self.energy_price[0, self.timestep] / 0.1
            ])

            normalized_predictions = np.concatenate((
                np.array([self.energy["Solar radiation"][0, min_timesteps_ahead:max_timesteps_ahead] / 1000]),
                np.array([self.energy_price[0, min_timesteps_ahead:max_timesteps_ahead] / 0.1])),
                axis=None
            )
        else:
            normalized_disturbances_observation_at_current_timestep = np.array([
                self.energy_price[0, self.timestep] / 0.1
            ])

            normalized_predictions = np.array([self.energy_price[0, min_timesteps_ahead:max_timesteps_ahead] / 0.1])

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

    def __check_is_single_day_simulated(self):
        if self.timestep == 24:
            return True
        else:
            return False

    def __save_prediction_results(self):
        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            available_renewable_energy = self.energy['Available solar energy']
        else:
            available_renewable_energy = []

        prediction_results = {
            'SOC': self.charging_station.vehicle_state_of_charge,
            'Grid energy': self.grid_energy_per_timestep,
            'Utilized renewable energy': self.renewable_energy_utilization_per_timestep,
            'Penalties': self.penalty_per_timestep,
            'Available renewable energy': available_renewable_energy,
            'Total cost': self.total_cost_per_timestep
        }
        savemat(data_files_directory_path + '\\prediction_results.mat', {'Prediction results': prediction_results})

    def reset(self, generate_new_initial_values=True):
        self.timestep = 0
        self.simulated_single_day = False
        self.total_cost_per_timestep = []
        self.grid_energy_per_timestep = []
        self.renewable_energy_utilization_per_timestep = []
        self.penalty_per_timestep = []

        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            self.energy = self.pv_system_manager.get_solar_energy()

        self.energy_price = self.central_management_system.get_energy_price(self.CURRENT_PRICE_MODEL,
                                                                            self.NUMBER_OF_DAYS_TO_PREDICT)
        self.__load_initial_simulation_values(generate_new_initial_values)

        return self.__get_observations()

    def __load_initial_simulation_values(self, generate_new_initial_values):
        if generate_new_initial_values:
            self.charging_station.load_initial_values()
        else:
            self.charging_station.generate_new_initial_values()

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def close(self):
        # return 0
        pass
