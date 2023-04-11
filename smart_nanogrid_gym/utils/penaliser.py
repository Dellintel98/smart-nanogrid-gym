class Penaliser:
    def __init__(self):
        self.REWARD_FOR_ALLOWED_VALUE_OF_STATE_OF_CHARGE = 4.0

        self.insufficiently_charged_vehicles_penalty = 0.0
        self.vehicle_state_of_charge_above_1_penalty = 0.0
        self.vehicle_reward = 0.0
        self.total_vehicle_penalty = 0.0

        self.battery_state_of_charge_below_dod_penalty = 0.0
        self.battery_state_of_charge_above_1_penalty = 0.0
        self.needless_battery_charging_penalty = 0.0
        self.excess_battery_charging_penalty = 0.0
        self.needless_battery_discharging_penalty = 0.0
        self.excess_battery_discharging_penalty = 0.0

        self.battery_reward = 0.0
        self.total_battery_penalty = 0.0

        self.total_penalty = 0.0

    def get_insufficiently_charged_vehicles_penalty(self):
        return self.insufficiently_charged_vehicles_penalty

    def calculate_insufficiently_charged_penalty(self, departing_vehicles, soc, requested_end_soc, timestep):
        penalties_per_departing_vehicle = []
        for vehicle in range(len(departing_vehicles)):
            penalty = self.calculate_insufficiently_charged_penalty_per_vehicle(departing_vehicles[vehicle], soc,
                                                                                requested_end_soc, timestep)
            penalties_per_departing_vehicle.append(penalty)

        self.insufficiently_charged_vehicles_penalty = sum(penalties_per_departing_vehicle)

    def calculate_insufficiently_charged_penalty_per_vehicle(self, vehicle, soc, requested_end_soc, timestep):
        # uncharged_capacity = 1 - soc[vehicle, timestep - 1]
        uncharged_capacity = requested_end_soc[vehicle, timestep - 1] - soc[vehicle, timestep - 1]
        charging_breathing_space = 0.05 * requested_end_soc[vehicle, timestep - 1]
        lower_breathing_space = -charging_breathing_space
        upper_breathing_space = charging_breathing_space
        # Todo: Rename to be readable and easily understandable
        if requested_end_soc[vehicle, timestep - 1] == 1:
            lower_breathing_space = 0.0

        if lower_breathing_space <= uncharged_capacity <= upper_breathing_space:
            penalty = 0
        elif uncharged_capacity < lower_breathing_space and requested_end_soc[vehicle, timestep - 1] == 1:
            penalty = (uncharged_capacity * 4) ** 2
        elif uncharged_capacity < lower_breathing_space:
            penalty = (uncharged_capacity * 2)
        else:
            penalty = (uncharged_capacity * 2) ** 2

        return penalty

    def penalise_vehicle_charging_above_max_state_of_charge(self, vehicle_soc):
        if vehicle_soc > 1.0:
            self.vehicle_state_of_charge_above_1_penalty = (vehicle_soc * 3) ** 2
        else:
            self.vehicle_state_of_charge_above_1_penalty = 0

    def penalise_battery_issues(self, current_state_of_charge, depth_of_discharge, initial_power_demand,
                                remaining_power_demand):
        self.penalise_unwanted_battery_charging(initial_power_demand, remaining_power_demand)
        self.penalise_unwanted_battery_discharging(initial_power_demand, remaining_power_demand)
        self.penalise_state_of_charge_outside_bounds(current_state_of_charge, depth_of_discharge)

    def penalise_unwanted_battery_charging(self, initial_power, remaining_power):
        if 0 <= initial_power < remaining_power:
            self.needless_battery_charging_penalty = remaining_power - initial_power
            self.excess_battery_charging_penalty = 0.0
        elif initial_power <= 0 < remaining_power:
            self.needless_battery_charging_penalty = 0.0
            self.excess_battery_charging_penalty = remaining_power
        else:
            self.needless_battery_charging_penalty = 0.0
            self.excess_battery_charging_penalty = 0.0

    def penalise_unwanted_battery_discharging(self, initial_power, remaining_power):
        if remaining_power < initial_power <= 0:
            self.needless_battery_discharging_penalty = initial_power - remaining_power
            self.excess_battery_discharging_penalty = 0.0
        elif remaining_power < 0 <= initial_power:
            self.needless_battery_discharging_penalty = 0.0
            self.excess_battery_discharging_penalty = -remaining_power
        else:
            self.needless_battery_discharging_penalty = 0.0
            self.excess_battery_discharging_penalty = 0.0

    def penalise_state_of_charge_outside_bounds(self, current_state_of_charge, depth_of_discharge):
        if depth_of_discharge <= current_state_of_charge <= 1.0:
            self.battery_reward = 0.0
            # self.battery_reward = self.REWARD_FOR_ALLOWED_VALUE_OF_STATE_OF_CHARGE
            self.battery_state_of_charge_below_dod_penalty = 0.0
            self.battery_state_of_charge_above_1_penalty = 0.0
        elif current_state_of_charge < depth_of_discharge:
            self.battery_state_of_charge_below_dod_penalty = ((depth_of_discharge - current_state_of_charge) * 5) ** 2
            self.battery_state_of_charge_above_1_penalty = 0.0
            self.battery_reward = 0.0
        elif current_state_of_charge > 1.0:
            self.battery_state_of_charge_above_1_penalty = ((current_state_of_charge - 1.0) * 5) ** 2
            self.battery_state_of_charge_below_dod_penalty = 0.0
            self.battery_reward = 0.0

    def get_total_penalty(self):
        self.calculate_total_penalty()
        return self.total_penalty

    def get_total_battery_penalty(self):
        return self.total_battery_penalty

    def calculate_total_penalty(self):
        self.calculate_total_battery_penalty()
        self.calculate_total_vehicle_penalty()
        self.total_penalty = self.total_battery_penalty + self.total_vehicle_penalty

    def calculate_total_battery_penalty(self):
        self.total_battery_penalty = self.battery_state_of_charge_below_dod_penalty\
                                     + self.battery_state_of_charge_above_1_penalty\
                                     + self.needless_battery_charging_penalty\
                                     + self.excess_battery_charging_penalty\
                                     + self.needless_battery_discharging_penalty\
                                     + self.excess_battery_discharging_penalty\
                                     - self.battery_reward

    def calculate_total_vehicle_penalty(self):
        self.total_vehicle_penalty = self.insufficiently_charged_vehicles_penalty\
                                     + self.vehicle_state_of_charge_above_1_penalty\
                                     - self.vehicle_reward
