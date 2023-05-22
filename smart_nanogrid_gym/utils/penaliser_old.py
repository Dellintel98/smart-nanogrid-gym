from numpy import floor, ceil, sign


class PenaliserOld:
    def __init__(self, number_of_chargers):
        self.NUMBER_OF_CHARGERS = number_of_chargers
        self.END_STATE_OF_CHARGE_MARGIN_RATIO = 0.05

        self._insufficiently_charged_vehicle_penalty = 0.0
        self.total_insufficiently_charged_vehicles_penalty = 0.0
        self._needless_vehicle_charging_penalty = 0.0
        self.total_needless_vehicles_charging_penalty = 0.0
        self.vehicles_overcharging_penalty = 0.0
        self.vehicles_over_discharging_penalty = 0.0
        self.total_vehicle_penalty = 0.0
        self.vehicle_reward = 0.0

        self.battery_state_of_charge_below_dod_penalty = 0.0
        self.battery_overcharging_penalty = 0.0
        self.battery_over_discharging_penalty = 0.0
        self.needless_battery_discharging_penalty = 0.0
        self.excess_battery_discharging_penalty = 0.0
        self.total_battery_penalty = 0.0
        self.battery_reward = 0.0

        self.low_resource_utilisation_penalty = 0.0
        self.charging_nonexistent_vehicles_penalty = 0.0

        self.total_penalty = 0.0

    def penalise_charging_station_issues(self, timestep, penalty_check_vehicles, requested_end_states_of_charge, states_of_charge,
                                         overcharging_values, over_discharging_values, vehicle_arrivals,
                                         charging_nonexistent_vehicles):
        # self.penalise_vehicle_overcharging(overcharging_values)
        # self.penalise_vehicle_over_discharging(over_discharging_values)
        self.penalise_charging_vehicles_outside_bounds(timestep, penalty_check_vehicles, requested_end_states_of_charge,
                                                       states_of_charge, vehicle_arrivals, charging_nonexistent_vehicles)

    def penalise_charging_vehicles_outside_bounds(self, timestep, penalty_check_vehicles, requested_end_soc, soc, arrivals,
                                                  charging_nonexistent_vehicles):
        self.charging_nonexistent_vehicles_penalty = sum(charging_nonexistent_vehicles)
        insufficiency_penalties = []
        needless_charging_penalties = []
        self.vehicle_reward = 0.0
        for vehicle in penalty_check_vehicles:
            vehicle_state_of_charge = self.extract_state_of_charge(vehicle, timestep, arrivals, soc)
            requested_vehicle_state_of_charge = self.extract_requested_state_of_charge(vehicle, timestep, arrivals,
                                                                                       requested_end_soc)

            self.penalise_state_of_charge_outside_margin(requested_vehicle_state_of_charge, vehicle_state_of_charge)

            insufficiency_penalties.append(self._insufficiently_charged_vehicle_penalty)
            # needless_charging_penalties.append(self._needless_vehicle_charging_penalty)

        self.total_insufficiently_charged_vehicles_penalty = sum(insufficiency_penalties)
        # self.total_needless_vehicles_charging_penalty = sum(needless_charging_penalties)
        # self.vehicle_reward = self.vehicle_reward * 10

    def extract_state_of_charge(self, vehicle, timestep, arrivals, states_of_charge):
        if timestep in arrivals:
            return states_of_charge[vehicle, timestep]
        else:
            return states_of_charge[vehicle, timestep - 1]

    def extract_requested_state_of_charge(self, vehicle, timestep, arrivals, requested_states_of_charge):
        if timestep in arrivals:
            return requested_states_of_charge[vehicle, timestep]
        else:
            return requested_states_of_charge[vehicle, timestep - 1]

    def penalise_state_of_charge_outside_margin(self, requested_soc, current_soc):
        lower_charged_state_margin = self.END_STATE_OF_CHARGE_MARGIN_RATIO * requested_soc
        upper_charged_state_margin = lower_charged_state_margin

        if requested_soc == 1.0:
            upper_charged_state_margin = 0.0

        if current_soc < requested_soc - lower_charged_state_margin:
            self._insufficiently_charged_vehicle_penalty = ((requested_soc - current_soc) * 10) ** 2
            # self._insufficiently_charged_vehicle_penalty = 10 * (requested_soc - current_soc) ** requested_soc
            self._needless_vehicle_charging_penalty = 0.0
            # self.vehicle_reward = self.vehicle_reward + 0.0
        elif requested_soc + upper_charged_state_margin < current_soc:
            self._insufficiently_charged_vehicle_penalty = 0.0
            self._needless_vehicle_charging_penalty = ((requested_soc - current_soc) * 2) ** 2
            # self.vehicle_reward = self.vehicle_reward + 0.0
        elif requested_soc - lower_charged_state_margin <= current_soc <= requested_soc + upper_charged_state_margin:
            self._insufficiently_charged_vehicle_penalty = 0.0
            self._needless_vehicle_charging_penalty = 0.0
            # self.vehicle_reward = self.vehicle_reward + 5.0

    def penalise_vehicle_overcharging(self, overcharging_powers):
        self.vehicles_overcharging_penalty = sum(overcharging_powers)

    def penalise_vehicle_over_discharging(self, over_discharging_powers):
        self.vehicles_over_discharging_penalty = sum(over_discharging_powers)

    def penalise_nanogrid_resource_issues(self, current_state_of_charge, depth_of_discharge, overcharging_value,
                                          over_discharging_value, solar_power, grid_power, battery_power, vehicle_power_demand):
        # self.penalise_battery_overcharging(overcharging_value)
        # self.penalise_battery_over_discharging(over_discharging_value)
        # self.penalise_unwanted_battery_discharging(initial_power_demand, remaining_power_demand)
        self.penalise_battery_state_below_depth_of_discharge(current_state_of_charge, depth_of_discharge)
        # self.penalise_low_resource_utilisation(grid_power, solar_power, vehicle_power_demand, battery_power)

    def penalise_battery_overcharging(self, overcharging_value):
        self.battery_overcharging_penalty = overcharging_value

    def penalise_battery_over_discharging(self, over_discharging_value):
        self.battery_over_discharging_penalty = over_discharging_value

    # def penalise_unwanted_battery_discharging(self, initial_power, remaining_power, ):
    #     if remaining_power < initial_power < 0:
    #         self.needless_battery_discharging_penalty = round(initial_power - remaining_power, 2) * 10
    #         self.excess_battery_discharging_penalty = 0.0
    #     elif remaining_power < 0 <= initial_power:
    #         self.needless_battery_discharging_penalty = 0.0
    #         self.excess_battery_discharging_penalty = round(-remaining_power, 2) * 10
    #     else:
    #         self.needless_battery_discharging_penalty = 0.0
    #         self.excess_battery_discharging_penalty = 0.0

    def penalise_battery_state_below_depth_of_discharge(self, current_state_of_charge, depth_of_discharge):
        if current_state_of_charge < depth_of_discharge:
            self.battery_state_of_charge_below_dod_penalty = ((depth_of_discharge - current_state_of_charge) * 10) ** 2
            # self.battery_state_of_charge_below_dod_penalty = 10 * self.NUMBER_OF_CHARGERS * (depth_of_discharge ** current_state_of_charge)
            # self.battery_reward = 0.0
        elif depth_of_discharge <= current_state_of_charge <= 1.0:
            self.battery_state_of_charge_below_dod_penalty = 0.0
            # self.battery_reward = 7.0
        else:
            raise ValueError("Error: Battery SOC greater than 1!")

    def penalise_low_resource_utilisation(self, grid_power, available_solar_power, vehicle_power_demand, battery_power):
        total_produced_power = available_solar_power + (abs(battery_power) * 0**(1 + sign(battery_power)))
        total_power_demand = vehicle_power_demand + (battery_power * 0**abs(sign(battery_power) - 1))

        # zero_division_padding = 0**(available_solar_power + floor(-1 * self.unit_step_function(battery_power) + 1))
        zero_division_padding = 1
        total_power_demand = total_power_demand + zero_division_padding
        total_produced_power = total_produced_power + zero_division_padding

        if total_produced_power == 0:
            breakpoint()

        utilisation = total_power_demand / total_produced_power

        unwanted_behaviour_value = 0**ceil(self.unit_step_function(battery_power)) * 0**ceil(self.unit_step_function(grid_power))
        utilisation = utilisation - unwanted_behaviour_value

        self.low_resource_utilisation_penalty = self.unit_step_function(-1 * floor(utilisation)) * abs(grid_power)
        # self.low_resource_utilisation_penalty = floor(self.unit_step_function(-1 * floor(utilisation))) \
        #                                         * self.NUMBER_OF_CHARGERS * 20\
        #                                         + ceil(self.unit_step_function(-1 * floor(utilisation))) \
        #                                         * floor(self.unit_step_function(utilisation)) * abs(grid_power)

    def unit_step_function(self, x):
        return 0.5 * (1 + sign(x))

    def get_insufficiently_charged_vehicles_penalty(self):
        return self.total_insufficiently_charged_vehicles_penalty

    def get_needlessly_charged_vehicles_penalty(self):
        return self.total_needless_vehicles_charging_penalty

    def get_overcharged_vehicles_penalty(self):
        return self.vehicles_overcharging_penalty

    def get_over_discharged_vehicles_penalty(self):
        return self.vehicles_over_discharging_penalty

    def get_battery_state_of_charge_below_dod_penalty(self):
        return self.battery_state_of_charge_below_dod_penalty

    def get_needlessly_discharged_battery_penalty(self):
        return self.needless_battery_discharging_penalty

    def get_battery_overcharging_penalty(self):
        return self.battery_overcharging_penalty

    def get_battery_over_discharging_penalty(self):
        return self.battery_over_discharging_penalty

    def get_excessively_discharged_battery_penalty(self):
        return self.excess_battery_discharging_penalty

    def get_low_resource_utilisation_penalty(self):
        return self.low_resource_utilisation_penalty

    def get_charging_nonexistent_vehicles_penalty(self):
        return self.charging_nonexistent_vehicles_penalty

    def get_total_penalty(self):
        self.calculate_total_penalty()
        return self.total_penalty

    def get_total_battery_penalty(self):
        return self.total_battery_penalty

    def get_total_vehicle_penalty(self):
        return self.total_vehicle_penalty

    def calculate_total_penalty(self):
        self.calculate_total_battery_penalty()
        self.calculate_total_vehicle_penalty()

        self.total_penalty = 0.8*self.total_battery_penalty + 1*self.total_vehicle_penalty
        #                      + self.low_resource_utilisation_penalty \
        #                      + self.charging_nonexistent_vehicles_penalty

    def calculate_total_battery_penalty(self):
        self.total_battery_penalty = self.battery_state_of_charge_below_dod_penalty
                                     # + self.battery_overcharging_penalty \
                                     # + self.battery_over_discharging_penalty \
                                     # - self.battery_reward
                                     # + self.needless_battery_discharging_penalty\
                                     # + self.excess_battery_discharging_penalty\

    def calculate_total_vehicle_penalty(self):
        self.total_vehicle_penalty = self.total_insufficiently_charged_vehicles_penalty
                                     # + self.total_needless_vehicles_charging_penalty\
                                     # + self.vehicles_overcharging_penalty\
                                     # + self.vehicles_over_discharging_penalty\
                                     # - self.vehicle_reward
