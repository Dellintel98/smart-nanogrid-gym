import json

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from scipy.io import savemat
import time

from smart_nanogrid_gym.utils.central_management_system import CentralManagementSystem
from smart_nanogrid_gym.utils.charging_station import ChargingStation
from smart_nanogrid_gym.utils.pv_system_manager import PVSystemManager
from ..utils.config import data_files_directory_path, solvers_files_directory_path


# Todo: Feat: Set possibility of using different filetypes for saving and loading only for predictions

# Todo: Feat: Add stohasticity in vehicle departures
# Todo: Feat: Add model training visualisation using pygame
# Todo: Feat: Add possibility for Electric Vehicles to have different battery capacities
# Todo: Feat: Add penalty for discharging vehicles (v2v) if it happens except for steps in which some other vehicle is
#             departing or plans to depart in next n steps
# Todo: Feat: Add possibility to load model specifications from json or csv..., e.g. load pricing model for energy
# Todo: Feat: Add possibility for using pv system data from the paper I'm writing

# Todo: Train models in DDPG and PPO for these cases: a) basic, only battery, only PV, battery and PV, only v2x,
#       only v2g, only v2v, v2v and battery, v2v and PV, v2g and battery, v2g and PV, v2v and battery and PV,
#       v2g and battery and PV, v2x and battery and PV

class SmartNanogridEnv(gym.Env):
    def __init__(self, price_model=0, number_of_chargers=8, pv_system_available_in_model=True, battery_system_available_in_model=True,
                 vehicle_to_everything=False, enable_different_vehicle_battery_capacities=True, enable_requested_state_of_charge=False,
                 algorithm_used='', environment_mode='', time_interval='', charging_mode='', vehicle_uncharged_penalty_mode=''):
        # Todo: Feat: Add possibility to specify whether to use same capacity or different ones for vehicle battery
        self.CURRENT_PRICE_MODEL = price_model
        self.NUMBER_OF_CHARGERS = number_of_chargers

        self.PV_SYSTEM_AVAILABLE_IN_MODEL = pv_system_available_in_model
        self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL = battery_system_available_in_model
        self.VEHICLE_TO_EVERYTHING = vehicle_to_everything

        self.ALGORITHM_USED = algorithm_used
        self.ENVIRONMENT_MODE = environment_mode
        self.REQUESTED_TIME_INTERVAL = time_interval
        self.TIME_INTERVAL = self.set_time_interval(time_interval)

        self.CHARGING_MODE = charging_mode
        self.VEHICLE_UNCHARGED_PENALTY_MODE = vehicle_uncharged_penalty_mode

        self.NUMBER_OF_DAYS_TO_PREDICT = 1
        self.NUMBER_OF_HOURS_AHEAD = 3

        self.central_management_system = CentralManagementSystem(self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL,
                                                                 self.PV_SYSTEM_AVAILABLE_IN_MODEL,
                                                                 self.VEHICLE_TO_EVERYTHING, self.CURRENT_PRICE_MODEL,
                                                                 self.NUMBER_OF_DAYS_TO_PREDICT, self.TIME_INTERVAL,
                                                                 self.NUMBER_OF_CHARGERS,
                                                                 enable_different_vehicle_battery_capacities,
                                                                 enable_requested_state_of_charge,
                                                                 self.CHARGING_MODE, self.VEHICLE_UNCHARGED_PENALTY_MODE)

        self.timestep = None
        self.info = None

        self.grid_energy_per_timestep, self.solar_energy_utilization_per_timestep = None, None
        self.total_cost_per_timestep, self.vehicle_penalty_per_timestep = None, None
        self.total_penalty_per_timestep, self.battery_penalty_per_timestep = None, None
        self.battery_per_timestep, self.grid_energy_cost_per_timestep = None, None
        self.grid_power_per_timestep = None

        self.battery_action_per_timestep, self.charger_actions_per_timestep = None, None
        self.total_charging_power_per_timestep, self.total_discharging_power_per_timestep = None, None
        self.charger_power_values_per_timestep, self.battery_power_value_per_timestep = None, None

        self.simulated_single_day = False

        amount_of_observed_variables = 1 + int(self.PV_SYSTEM_AVAILABLE_IN_MODEL)
        number_of_observed_charger_values = 2

        amount_of_charger_predictions = self.NUMBER_OF_CHARGERS * number_of_observed_charger_values
        amount_of_states = amount_of_observed_variables + (self.NUMBER_OF_HOURS_AHEAD * amount_of_observed_variables)

        self.total_amount_of_states = amount_of_states + amount_of_charger_predictions + int(self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL)

        spaces_low = np.array(np.zeros(self.total_amount_of_states), dtype=np.float32)
        spaces_high = np.array(np.ones(self.total_amount_of_states), dtype=np.float32)

        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL:
            if self.VEHICLE_TO_EVERYTHING:
                actions_low = np.array(np.ones(self.NUMBER_OF_CHARGERS + 1), dtype=np.float32) * (-1)
            else:
                actions_low = np.array(np.zeros(self.NUMBER_OF_CHARGERS), dtype=np.float32)
                actions_low = np.insert(actions_low, self.NUMBER_OF_CHARGERS, -1)
            actions_high = np.array(np.ones(self.NUMBER_OF_CHARGERS + 1), dtype=np.float32)

            self.action_space = spaces.Box(low=actions_low, high=actions_high, shape=(self.NUMBER_OF_CHARGERS + 1,),
                                           dtype=np.float32)
        else:
            if self.VEHICLE_TO_EVERYTHING:
                actions_low = -1
            else:
                actions_low = 0
            actions_high = 1
            self.action_space = spaces.Box(low=actions_low, high=actions_high, shape=(self.NUMBER_OF_CHARGERS,),
                                           dtype=np.float32)

        self.observation_space = spaces.Box(low=spaces_low, high=spaces_high, dtype=np.float32)

        # Todo: Add look-ahead action_space for looking at agents planned actions to see will departing vehicles be
        #       charged enough based on current action, and penalize wrong future actions

    def set_time_interval(self, requested_time_interval):
        # Method for setting time_interval by keyword argument from ['1h', '2h'...-> '?h'; '15min'...->'?min']
        # Todo: Feat: Add security check for value provided as an argument
        if requested_time_interval:
            if 'h' in requested_time_interval:
                time_interval = float(requested_time_interval.replace('h', ''))
                return time_interval
            elif 'min' in requested_time_interval:
                time_interval = float(requested_time_interval.replace('min', '')) / 60.0
                return time_interval
            else:
                raise ValueError('Wrong time interval was provided')
        else:
            return float(1)

    def step(self, actions):
        results = self.central_management_system.simulate(self.timestep, actions)

        self.total_cost_per_timestep.append(results['Total cost'])
        self.grid_power_per_timestep.append(results['Grid power'])
        self.grid_energy_per_timestep.append(results['Grid energy'])
        self.solar_energy_utilization_per_timestep.append(results['Utilized solar energy'])
        self.vehicle_penalty_per_timestep.append(results['Insufficiently charged vehicles penalty'])
        self.battery_penalty_per_timestep.append(results['Battery penalty'])
        self.total_penalty_per_timestep.append(results['Total penalty'])
        self.battery_per_timestep.append(results['Battery state of charge'])
        self.grid_energy_cost_per_timestep.append(results['Grid energy cost'])
        self.battery_action_per_timestep.append(results['Battery action'])
        self.charger_actions_per_timestep.append(results['Charger actions'])
        self.total_charging_power_per_timestep.append(results['Total charging power'])
        self.total_discharging_power_per_timestep.append(results['Total discharging power'])
        self.charger_power_values_per_timestep.append(results['Charger power values'])
        self.battery_power_value_per_timestep.append(results['Battery power value'])

        observations = self.__get_observations()
        self.timestep = self.timestep + 1

        self.simulated_single_day = self.__check_is_single_day_simulated()
        if self.simulated_single_day:
            self.timestep = 0
            self.__save_prediction_results()

        reward = -results['Total cost']
        self.info = {}

        out_of_scope = False

        return observations, reward, self.simulated_single_day, out_of_scope, self.info

    def __get_observations(self):
        min_timesteps_ahead = self.timestep + 1
        max_timesteps_ahead = min_timesteps_ahead + self.NUMBER_OF_HOURS_AHEAD

        results = self.central_management_system.observe(self.timestep, min_timesteps_ahead, max_timesteps_ahead)

        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            normalized_disturbances_observation_at_current_timestep = np.array([results['solar_radiation'],
                                                                                results['energy_price']])
            normalized_predictions = np.concatenate((np.array([results['radiation_predictions']]),
                                                     np.array([results['price_predictions']])),
                                                    axis=None)
        else:
            normalized_disturbances_observation_at_current_timestep = np.array([results['energy_price']])
            normalized_predictions = np.array([results['price_predictions']])

        departures_array = np.array(results['departures'])
        normalized_departures = departures_array / 24

        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL:
            normalized_states = np.concatenate((
                np.array(results['vehicles_state_of_charge']),
                normalized_departures,
                np.array(results['battery_soc'])),
                axis=None
            )
        else:
            normalized_states = np.concatenate((
                np.array(results['vehicles_state_of_charge']),
                normalized_departures),
                axis=None
            )

        observations = np.concatenate((
            normalized_disturbances_observation_at_current_timestep,
            normalized_predictions,
            normalized_states),
            axis=None, dtype=np.float32
        )

        return observations

    def __check_is_single_day_simulated(self):
        if self.timestep == (24.0 / self.TIME_INTERVAL):
            return True
        else:
            return False

    def __save_prediction_results(self):
        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            available_solar_energy = self.central_management_system.pv_system_manager.get_available_solar_energy()
            available_solar_energy = available_solar_energy.tolist()
        else:
            available_solar_energy = []

        prediction_results = {
            'SOC': self.central_management_system.charging_station.get_vehicles_state_of_charge().tolist(),
            'Grid_power': self.grid_power_per_timestep,
            'Grid_energy': self.grid_energy_per_timestep,
            'Utilized_solar_energy': self.solar_energy_utilization_per_timestep,
            'Vehicle_penalties': self.vehicle_penalty_per_timestep,
            'Battery_penalties': self.battery_penalty_per_timestep,
            'Total_penalties': self.total_penalty_per_timestep,
            'Available_solar_energy': available_solar_energy,
            'Total_cost': self.total_cost_per_timestep,
            'Battery_state_of_charge': self.battery_per_timestep,
            'Grid_energy_cost': self.grid_energy_cost_per_timestep,
            'Battery_action': self.battery_action_per_timestep,
            'Charger_actions': self.charger_actions_per_timestep,
            'Total_charging_power': self.total_charging_power_per_timestep,
            'Total_discharging_power': self.total_discharging_power_per_timestep,
            'Charger_power_values': self.charger_power_values_per_timestep,
            'Battery_power_value': self.battery_power_value_per_timestep
        }
        # Todo: Change mat to excel
        savemat(data_files_directory_path + '\\last_prediction_results.mat', {'Prediction_results': prediction_results})

        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL and self.PV_SYSTEM_AVAILABLE_IN_MODEL and self.VEHICLE_TO_EVERYTHING:
            model_variant_name = 'v2x-b-pv'
        elif self.VEHICLE_TO_EVERYTHING:
            model_variant_name = 'v2x'
        elif self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL and self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            model_variant_name = 'b-pv'
        else:
            model_variant_name = 'basic'

        if self.ENVIRONMENT_MODE == 'training':
            file_destination = 'training_files'
        elif self.ENVIRONMENT_MODE == 'evaluation':
            file_destination = 'evaluation_files'
        elif self.ENVIRONMENT_MODE == 'prediction':
            file_destination = 'single_prediction_files'
        else:
            file_destination = ''

        saving_directory_path = solvers_files_directory_path + '\\RL\\' + file_destination + '\\'

        file_name_prefix = f'{self.ALGORITHM_USED}'
        file_name_root = f'{model_variant_name}-{self.CHARGING_MODE}-{self.VEHICLE_UNCHARGED_PENALTY_MODE}'
        file_name_suffix = f'{self.NUMBER_OF_CHARGERS}ch-{self.REQUESTED_TIME_INTERVAL}'
        file_name = f'{file_name_prefix}-{file_name_root}-{file_name_suffix}'

        savemat(f'{saving_directory_path}{file_name}-prediction_results.mat', {'Prediction_results': prediction_results})

        with open(saving_directory_path + file_name + "-prediction_results.json", "w") as fp:
            json.dump(prediction_results, fp, indent=4)

        self.central_management_system.charging_station.save_initial_values_to_json_file(saving_directory_path,
                                                                                         filename=file_name)

        self.central_management_system.charging_station.save_initial_values_to_mat_file(saving_directory_path,
                                                                                        filename=file_name)

    def reset(self, generate_new_initial_values=True, algorithm_used='', environment_mode='', **kwargs):
        self.timestep = 0
        self.simulated_single_day = False
        self.total_cost_per_timestep = []
        self.grid_power_per_timestep = []
        self.grid_energy_per_timestep = []
        self.solar_energy_utilization_per_timestep = []
        self.vehicle_penalty_per_timestep = []
        self.battery_penalty_per_timestep = []
        self.total_penalty_per_timestep = []
        self.battery_per_timestep = []
        self.grid_energy_cost_per_timestep = []
        self.battery_action_per_timestep = []
        self.charger_actions_per_timestep = []
        self.total_charging_power_per_timestep = []
        self.total_discharging_power_per_timestep = []
        self.charger_power_values_per_timestep = []
        self.battery_power_value_per_timestep = []

        self.ALGORITHM_USED = algorithm_used if algorithm_used else self.ALGORITHM_USED
        self.ENVIRONMENT_MODE = environment_mode if environment_mode else self.ENVIRONMENT_MODE

        # Todo: Feat: Add reset to all subclasses and to price and pv if different models have different configs for them

        self.__load_initial_simulation_values(generate_new_initial_values)

        return self.__get_observations(), {}

    def __load_initial_simulation_values(self, generate_new_initial_values):
        if generate_new_initial_values:
            self.central_management_system.charging_station.generate_new_initial_values(self.TIME_INTERVAL)
        else:
            self.central_management_system.charging_station.load_initial_values()

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def close(self):
        # return 0
        pass
