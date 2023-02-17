from numpy import zeros, array, concatenate, random

from smart_nanogrid_gym.utils.battery_energy_storage_system import BatteryEnergyStorageSystem


class CentralManagementSystem:
    def __init__(self, battery_system_available_in_model, building_demand, building_in_nanogrid):
        self.total_cost = 0
        self.grid_energy_cost = 0
        self.battery_system = self.initialise_battery_system(battery_system_available_in_model)
        self.building_demand = self.initialise_building_demand(building_in_nanogrid)
        self.building_exclusive_demand = self.initialise_building_exclusive_demand(building_demand)

    def initialise_battery_system(self, battery_system_available_in_model):
        if battery_system_available_in_model:
            return BatteryEnergyStorageSystem(80, 0.5, 0.95, 0.95, 44, 44, 0.15)
        else:
            return None

    def initialise_building_demand(self, building_in_nanogrid):
        if building_in_nanogrid:
            demand = array([(random.rand() * 10) for i in range(24)])
            return demand
        else:
            return None

    def initialise_building_exclusive_demand(self, building_demand):
        if building_demand:
            demand = array([round(random.rand() - 0.1) for i in range(24)])
            return demand
        else:
            return None

    def simulate(self, current_timestep, total_charging_power, energy, energy_price, departing_vehicles, soc,
                 pv_system_available, vehicle_to_everything, building_in_nanogrid):
        # hour = self.timestep
        # timestep = self.timestep
        # time_interval = 1
        hour = current_timestep

        if pv_system_available:
            renewable = energy['Available solar energy']
            available_renewable_energy = renewable[hour, 0]
        else:
            available_renewable_energy = 0

        grid_energy = self.calculate_grid_energy(total_charging_power, available_renewable_energy, hour,
                                                 vehicle_to_everything, building_in_nanogrid)

        self.calculate_grid_energy_cost(grid_energy, energy_price[0, hour])
        insufficiently_charged_vehicles_penalty = self.calculate_insufficiently_charged_penalty(departing_vehicles, soc,
                                                                                                hour)

        self.calculate_total_cost(insufficiently_charged_vehicles_penalty)

        if self.battery_system:
            battery_soc = self.battery_system.current_battery_capacity
        else:
            battery_soc = 0

        return {
            'Total cost': self.total_cost,
            'Grid energy': grid_energy,
            'Utilized renewable energy': available_renewable_energy,
            'Insufficiently charged vehicles penalty': insufficiently_charged_vehicles_penalty,
            'Battery state of charge': battery_soc,
            'Grid energy cost': self.grid_energy_cost
        }

    def calculate_grid_energy(self, total_power, available_renewable_energy, hour, vehicle_to_everything,
                              building_in_nanogrid):
        # if building_in_nanogrid:
        #     current_building_exclusive_demand = self.building_exclusive_demand[hour]
        # else:
        #     current_building_exclusive_demand = 0

        # if building_in_nanogrid:
        #     # if current_building_exclusive_demand:
        #     #     grid_energy = 0
        #     #     return grid_energy
        #
        #     building_demand = self.building_demand[hour]
        #     remaining_energy_demand = building_demand + total_power - available_renewable_energy
        # else:
        #     remaining_energy_demand = total_power - available_renewable_energy
        remaining_energy_demand = total_power - available_renewable_energy

        if not self.battery_system:
            if remaining_energy_demand == 0:
                grid_energy = 0
            else:
                if vehicle_to_everything:
                    grid_energy = remaining_energy_demand
                else:
                    grid_energy = max([remaining_energy_demand, 0])
            return grid_energy

        if remaining_energy_demand == 0:
            grid_energy = 0
        elif remaining_energy_demand > 0:
            capacity_available_to_discharge = self.battery_system.current_battery_capacity - self.battery_system.depth_of_discharge
            if capacity_available_to_discharge > 0:
                power_available_for_discharge = capacity_available_to_discharge * self.battery_system.max_battery_capacity
                max_discharging_energy = min([self.battery_system.max_discharging_power, power_available_for_discharge])
                temp_remaining_energy_demand = remaining_energy_demand - max_discharging_energy

                if temp_remaining_energy_demand == 0:
                    grid_energy = 0
                elif temp_remaining_energy_demand > 0:
                    grid_energy = temp_remaining_energy_demand
                else:
                    max_discharging_energy = remaining_energy_demand
                    grid_energy = 0

                self.battery_system.current_battery_capacity = self.battery_system.current_battery_capacity - max_discharging_energy / self.battery_system.max_battery_capacity
            else:
                grid_energy = remaining_energy_demand
        else:
            remaining_available_renewable_energy = available_renewable_energy - total_power
            capacity_available_to_charge = 1 - self.battery_system.current_battery_capacity

            if capacity_available_to_charge > 0:
                power_available_for_charge = capacity_available_to_charge * self.battery_system.max_battery_capacity
                max_charging_energy = min([self.battery_system.max_charging_power, power_available_for_charge])
                temp_remaining_available_renewable_energy = remaining_available_renewable_energy - max_charging_energy

                if temp_remaining_available_renewable_energy == 0:
                    grid_energy = 0
                    temp_bess_cap = self.battery_system.current_battery_capacity + max_charging_energy / self.battery_system.max_battery_capacity
                    # if temp_bess_cap > 1:
                    #     breakpoint()
                elif temp_remaining_available_renewable_energy > 0:
                    temp_bess_cap = self.battery_system.current_battery_capacity + max_charging_energy / self.battery_system.max_battery_capacity
                    # if temp_bess_cap > 1:
                    #     breakpoint()
                    if vehicle_to_everything:
                        grid_energy = -temp_remaining_available_renewable_energy
                    else:
                        grid_energy = 0
                else:
                    # temp_bess_cap = self.battery_system.current_battery_capacity + max_charging_energy / self.battery_system.max_battery_capacity
                    # if temp_bess_cap > 1:
                    #     breakpoint()
                    max_charging_energy = remaining_available_renewable_energy
                    grid_energy = 0

                self.battery_system.current_battery_capacity = self.battery_system.current_battery_capacity + max_charging_energy / self.battery_system.max_battery_capacity
            else:
                if vehicle_to_everything:
                    grid_energy = remaining_energy_demand
                else:
                    grid_energy = 0
        return grid_energy

    def calculate_grid_energy_cost(self, grid_energy, price):
        self.grid_energy_cost = grid_energy * price

    def calculate_insufficiently_charged_penalty(self, departing_vehicles, soc, hour):
        penalties_per_departing_vehicle = []
        for vehicle in range(len(departing_vehicles)):
            penalty = self.calculate_insufficiently_charged_penalty_per_vehicle(departing_vehicles[vehicle], soc, hour)
            penalties_per_departing_vehicle.append(penalty)

        return sum(penalties_per_departing_vehicle)

    def calculate_insufficiently_charged_penalty_per_vehicle(self, vehicle, soc, hour):
        uncharged_capacity = 1 - soc[vehicle, hour - 1]
        penalty = (uncharged_capacity * 2) ** 2
        return penalty

    def calculate_total_cost(self, total_penalty):
        self.total_cost = self.grid_energy_cost + total_penalty

    def get_energy_price(self, current_price_model, experiment_length_in_days):
        grid_tariff_high = 0.028
        grid_tariff_low = 0.013333333
        energy_tariff_high = 0.148933333
        energy_tariff_low = 0.087613333
        res_incentive = 0.014
        high_tariff = grid_tariff_high + energy_tariff_high + res_incentive
        low_tariff = grid_tariff_low + energy_tariff_low + res_incentive

        price_day = self.get_price_day(current_price_model, low_tariff=low_tariff, high_tariff=high_tariff)
        price = zeros((experiment_length_in_days, 2 * 24))
        for day in range(0, experiment_length_in_days):
            price[day, :] = price_day
        return price

    def get_price_day(self, current_price_model, low_tariff=0.0, high_tariff=0.0):
        price_day = []
        if current_price_model == 0:
            price_day = array([low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff,
                               high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
                               high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
                               high_tariff, low_tariff, low_tariff, low_tariff, low_tariff])
        if current_price_model == 1:
            price_day = array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        elif current_price_model == 2:
            price_day = array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06,
                               0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
        elif current_price_model == 3:
            price_day = array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                               0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
        elif current_price_model == 4:
            price_day = array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
        elif current_price_model == 5:
            price_day[1, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
            price_day[2, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05,
                               0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05]
            price_day[3, :] = [0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                               0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070]
            price_day[4, :] = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]

        price_day = concatenate([price_day, price_day], axis=0)
        return price_day
