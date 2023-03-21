from numpy import zeros, array, concatenate, random

from smart_nanogrid_gym.utils.accountant import Accountant
from smart_nanogrid_gym.utils.battery_energy_storage_system import BatteryEnergyStorageSystem
from smart_nanogrid_gym.utils.penaliser import Penaliser


class CentralManagementSystem:
    def __init__(self, battery_system_available_in_model, pv_system_available_in_model, vehicle_to_everything,
                 current_price_model, experiment_length_in_days, time_interval):
        # To-do: Add building_demand, building_in_nanogrid as init arguments
        self.battery_system = self.initialise_battery_system(battery_system_available_in_model)
        self.pv_system_available = pv_system_available_in_model

        self.vehicle_to_everything = vehicle_to_everything
        self.accountant = Accountant()
        self.accountant.set_energy_price(current_price_model, experiment_length_in_days, time_interval)
        self.penaliser = Penaliser()

    def initialise_battery_system(self, battery_system_available_in_model):
        if battery_system_available_in_model:
            return BatteryEnergyStorageSystem(80, 0.5, 44, 44, 0.95, 0.95, 0.15)
        else:
            return None

    def simulate(self, timestep, charging_station, number_of_chargers, pv_system_manager,
                 actions, time_interval):
        charger_actions = actions[0:number_of_chargers]

        if self.battery_system:
            battery_action = actions[-1]
        else:
            battery_action = 0
        [total_charging_power, total_discharging_power] = charging_station.simulate_vehicle_charging(charger_actions,
                                                                                                     timestep,
                                                                                                     time_interval)
        if self.pv_system_available:
            available_solar_power = pv_system_manager.get_available_solar_produced_power(time_interval)
        else:
            available_solar_power = 0

        management_results = self.manage_nanogrid(timestep, total_charging_power, total_discharging_power,
                                                  available_solar_power,
                                                  charging_station.departing_vehicles,
                                                  charging_station.vehicle_state_of_charge,
                                                  battery_action, time_interval)
        return management_results

    def manage_nanogrid(self, timestep, total_charging_power, total_discharging_power, solar_power,
                        departing_vehicles, soc, battery_action, time_interval):
        available_solar_power = self.get_available_solar_power_at_current_timestep(solar_power, timestep)

        total_power = total_charging_power + total_discharging_power
        grid_power = self.calculate_grid_power(total_power, available_solar_power, battery_action, time_interval)
        grid_energy = grid_power * time_interval

        energy_price = self.accountant.get_energy_price_at_time_t(timestep)
        grid_energy_cost = self.accountant.calculate_grid_energy_cost(grid_energy, energy_price)

        self.penaliser.calculate_insufficiently_charged_penalty(departing_vehicles, soc, timestep)

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
            'Battery state of charge': battery_soc,
            'Grid energy cost': grid_energy_cost
        }

    def get_available_solar_power_at_current_timestep(self, solar_power, current_timestep):
        if self.pv_system_available:
            available_solar_power = solar_power[0, current_timestep]
        else:
            available_solar_power = 0
        return available_solar_power

    def calculate_grid_power(self, power_demand, available_solar_power, battery_action, time_interval):
        # if building_in_nanogrid:
        #     remaining_energy_demand = building_demand + total_power - available_renewable_energy
        remaining_power_demand = power_demand - available_solar_power

        if remaining_power_demand == 0:
            return 0, 0
        elif remaining_power_demand > 0:
            power_from_grid, battery_penalty = self.calculate_amount_of_power_supplied_from_grid(remaining_power_demand, battery_action, time_interval)
            return power_from_grid, battery_penalty
        else:
            available_power = available_solar_power - power_demand
            power_to_grid, battery_penalty = self.calculate_amount_of_power_supplied_to_grid(available_power, battery_action, time_interval)
            return power_to_grid, battery_penalty

    def calculate_amount_of_power_supplied_from_grid(self, power_demand, battery_action, time_interval):
        if self.battery_system and battery_action != 0:
            self.penaliser.penalise_battery_discharging(battery_action)
            remaining_power_demand = self.battery_system.discharge(power_demand, battery_action, time_interval)
            return remaining_power_demand
        else:
            return power_demand

    def calculate_amount_of_power_supplied_to_grid(self, available_power, battery_action, time_interval):
        if self.battery_system and battery_action != 0:
            self.penaliser.penalise_battery_charging(battery_action)
            remaining_available_power = self.battery_system.charge(available_power, battery_action, time_interval)
        else:
            remaining_available_power = available_power

        if self.vehicle_to_everything:
            return -remaining_available_power
        else:
            # Todo: Feat: Add penalty for wasted energy/power -> wasted_power = (+||-???)remaining_available_power
            return 0

    def get_normalised_energy_price_at_timestep_t(self, t):
        return self.accountant.get_normalised_energy_price_at_time_t(t)

    def get_normalised_energy_price_in_range(self, min_timesteps_ahead, max_timesteps_ahead):
        return self.accountant.get_normalised_energy_price_in_range(min_timesteps_ahead, max_timesteps_ahead)
