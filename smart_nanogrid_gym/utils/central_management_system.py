from numpy import zeros, array, concatenate, random

from smart_nanogrid_gym.utils.accountant import Accountant
from smart_nanogrid_gym.utils.battery_energy_storage_system import BatteryEnergyStorageSystem
from smart_nanogrid_gym.utils.charging_station import ChargingStation
from smart_nanogrid_gym.utils.penaliser import Penaliser
from smart_nanogrid_gym.utils.pv_system_manager import PVSystemManager


class CentralManagementSystem:
    def __init__(self, battery_system_available_in_model, pv_system_available_in_model, vehicle_to_everything,
                 current_price_model, experiment_length_in_days, time_interval, number_of_chargers,
                 enable_different_vehicle_battery_capacities, enable_requested_state_of_charge,
                 charging_mode, vehicle_uncharged_penalty_mode):
        self.TIME_INTERVAL = time_interval
        self.EXPERIMENT_LENGTH_IN_DAYS = experiment_length_in_days
        self.NUMBER_OF_CHARGERS = number_of_chargers

        self.battery_system = self.initialise_battery_system(battery_system_available_in_model, charging_mode)
        self.pv_system_manager = self.initialise_pv_system(pv_system_available_in_model)

        self.vehicle_to_everything = vehicle_to_everything
        self.charging_station = ChargingStation(number_of_chargers, time_interval,
                                                enable_different_vehicle_battery_capacities,
                                                enable_requested_state_of_charge, charging_mode,
                                                vehicle_uncharged_penalty_mode)

        self.accountant = Accountant()
        self.accountant.set_energy_price(current_price_model, experiment_length_in_days, time_interval)

        self.penaliser = Penaliser()

    def initialise_battery_system(self, battery_system_available_in_model, charging_mode):
        if battery_system_available_in_model:
            return BatteryEnergyStorageSystem(charging_mode, 80, 0.5, 44, 44, 0.95, 0.95, 0.15)
        else:
            return None

    def initialise_pv_system(self, pv_system_available_in_model):
        if pv_system_available_in_model:
            return PVSystemManager(self.EXPERIMENT_LENGTH_IN_DAYS, self.TIME_INTERVAL)
        else:
            return None

    def observe(self, timestep, min_timesteps_ahead, max_timesteps_ahead):
        [departure_times, vehicles_state_of_charge] = self.charging_station.simulate(timestep, self.TIME_INTERVAL)

        if self.battery_system:
            battery_soc = self.battery_system.get_state_of_charge()
        else:
            battery_soc = 0.0

        energy_price = self.accountant.get_normalised_energy_price_at_time_t(timestep)
        price_predictions = self.accountant.get_normalised_energy_price_in_range(min_timesteps_ahead,
                                                                                 max_timesteps_ahead)

        if self.pv_system_manager:
            solar_radiation = self.pv_system_manager.get_normalized_solar_radiation_at_timestep_t(timestep)
            radiation_predictions = self.pv_system_manager.get_normalized_solar_predictions_in_range(min_timesteps_ahead,
                                                                                                     max_timesteps_ahead)

            return {
                'departures': departure_times,
                'vehicles_state_of_charge': vehicles_state_of_charge,
                'battery_soc': battery_soc,
                'energy_price': energy_price,
                'price_predictions': price_predictions,
                'solar_radiation': solar_radiation,
                'radiation_predictions': radiation_predictions
            }

        return {
            'departures': departure_times,
            'vehicles_state_of_charge': vehicles_state_of_charge,
            'battery_soc': battery_soc,
            'energy_price': energy_price,
            'price_predictions': price_predictions
        }

    def simulate(self, timestep, actions):
        management_results = self.manage_nanogrid(timestep, actions)
        return management_results

    def manage_nanogrid(self, timestep, actions):
        charger_actions = actions[0:self.NUMBER_OF_CHARGERS]

        if self.battery_system:
            battery_action = actions[-1]
        else:
            battery_action = 0

        [total_charging_power, total_discharging_power] = self.charging_station.simulate_vehicle_charging(charger_actions,
                                                                                                          timestep,
                                                                                                          self.TIME_INTERVAL)
        if self.pv_system_manager:
            available_solar_power = self.pv_system_manager.get_available_solar_produced_power_at_timestep_t(timestep)
        else:
            available_solar_power = 0

        total_power = total_charging_power + total_discharging_power
        grid_power = self.calculate_grid_power(total_power, available_solar_power, battery_action)
        grid_energy = grid_power * self.TIME_INTERVAL

        energy_price = self.accountant.get_energy_price_at_time_t(timestep)
        grid_energy_cost = self.accountant.calculate_grid_energy_cost(grid_energy, energy_price)

        self.penaliser.calculate_insufficiently_charged_penalty(self.charging_station.get_all_departing_vehicles(),
                                                                self.charging_station.get_vehicles_state_of_charge(),
                                                                self.charging_station.get_requested_end_state_of_charge_for_all_chargers(),
                                                                timestep)

        total_penalty = self.penaliser.get_total_penalty()
        total_cost = self.accountant.calculate_total_cost(additional_cost=total_penalty)

        if self.battery_system:
            battery_soc = self.battery_system.current_capacity
        else:
            battery_soc = 0

        return {
            'Total cost': total_cost,
            'Grid power': grid_power,
            'Grid energy': grid_energy,
            'Utilized solar energy': available_solar_power,
            'Insufficiently charged vehicles penalty': self.penaliser.get_insufficiently_charged_vehicles_penalty(),
            'Battery penalty': self.penaliser.total_battery_penalty,
            'Total penalty': total_penalty,
            'Battery state of charge': battery_soc,
            'Grid energy cost': grid_energy_cost
        }

    def calculate_grid_power(self, power_demand, available_solar_power, battery_action):
        remaining_power_demand = power_demand - available_solar_power

        if remaining_power_demand == 0:
            return 0
        elif remaining_power_demand > 0:
            power_from_grid = self.calculate_amount_of_power_supplied_from_grid(remaining_power_demand, battery_action)
            return power_from_grid
        else:
            available_power = available_solar_power - power_demand
            power_to_grid = self.calculate_amount_of_power_supplied_to_grid(available_power, battery_action)
            if power_to_grid != 0:
                breakpoint()
            return power_to_grid

    def calculate_amount_of_power_supplied_from_grid(self, power_demand, battery_action):
        if self.battery_system and battery_action != 0:
            self.penaliser.penalise_battery_discharging(battery_action)
            remaining_power_demand = self.battery_system.discharge(power_demand, battery_action, self.TIME_INTERVAL)

            if self.battery_system.CHARGING_MODE == 'bounded':
                self.penaliser.penalise_discharging_battery_with_power_greater_than_power_demand(remaining_power_demand)
                self.penaliser.penalise_battery_capacity_below_depth_of_discharge(self.battery_system.current_capacity,
                                                                                  self.battery_system.depth_of_discharge)
            return remaining_power_demand
        else:
            return power_demand

    def calculate_amount_of_power_supplied_to_grid(self, available_power, battery_action):
        if self.battery_system and battery_action != 0:
            self.penaliser.penalise_battery_charging(battery_action)
            remaining_available_power = self.battery_system.charge(available_power, battery_action, self.TIME_INTERVAL)

            if self.battery_system.CHARGING_MODE == 'bounded':
                self.penaliser.penalise_charging_battery_with_non_existing_power(remaining_available_power)
                self.penaliser.penalise_battery_capacity_greater_than_100_percent(self.battery_system.current_capacity)
        else:
            remaining_available_power = available_power

        if self.vehicle_to_everything:
            return -remaining_available_power
        else:
            # Todo: Feat: Add penalty for wasted energy/power -> wasted_power = (+||-???)remaining_available_power
            return 0
