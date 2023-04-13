class Penaliser:
    def __init__(self):
        self.END_STATE_OF_CHARGE_MARGIN_RATIO = 0.05

        self._insufficiently_charged_vehicle_penalty = 0.0
        self.total_insufficiently_charged_vehicles_penalty = 0.0
        self._needless_vehicle_charging_penalty = 0.0
        self.total_needless_vehicles_charging_penalty = 0.0
        self.excess_vehicles_charging_penalty = 0.0
        self.excess_vehicles_discharging_penalty = 0.0
        self.total_vehicle_penalty = 0.0

        self.battery_state_of_charge_below_dod_penalty = 0.0
        self.needless_battery_charging_penalty = 0.0
        self.excess_battery_charging_penalty = 0.0
        self.needless_battery_discharging_penalty = 0.0
        self.excess_battery_discharging_penalty = 0.0
        self.total_battery_penalty = 0.0

        self.total_penalty = 0.0

    def penalise_charging_station_issues(self, timestep, penalty_check_vehicles, requested_end_states_of_charge, states_of_charge,
                                         excess_charging_powers, excess_discharging_powers, vehicle_arrivals):
        self.penalise_unwanted_vehicle_charging(excess_charging_powers)
        self.penalise_unwanted_vehicle_discharging(excess_discharging_powers)
        self.penalise_charging_vehicles_outside_bounds(timestep, penalty_check_vehicles, requested_end_states_of_charge,
                                                       states_of_charge, vehicle_arrivals)

    def penalise_charging_vehicles_outside_bounds(self, timestep, penalty_check_vehicles, requested_end_soc, soc, arrivals):
        insufficiency_penalties = []
        needless_charging_penalties = []
        for vehicle in penalty_check_vehicles:
            vehicle_state_of_charge = self.extract_state_of_charge(vehicle, timestep, arrivals, soc)
            requested_vehicle_state_of_charge = self.extract_requested_state_of_charge(vehicle, timestep, arrivals,
                                                                                       requested_end_soc)

            self.penalise_state_of_charge_outside_margin(requested_vehicle_state_of_charge, vehicle_state_of_charge)

            insufficiency_penalties.append(self._insufficiently_charged_vehicle_penalty)
            needless_charging_penalties.append(self._needless_vehicle_charging_penalty)

        self.total_insufficiently_charged_vehicles_penalty = sum(insufficiency_penalties)
        self.total_needless_vehicles_charging_penalty = sum(needless_charging_penalties)

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
            self._insufficiently_charged_vehicle_penalty = ((requested_soc - current_soc) * 4) ** 2
            self._needless_vehicle_charging_penalty = 0.0
        elif requested_soc + upper_charged_state_margin < current_soc:
            self._insufficiently_charged_vehicle_penalty = 0.0
            self._needless_vehicle_charging_penalty = (requested_soc - current_soc) ** 2
        else:
            self._insufficiently_charged_vehicle_penalty = 0.0
            self._needless_vehicle_charging_penalty = 0.0

    def penalise_unwanted_vehicle_charging(self, excess_charging_powers):
        self.excess_vehicles_charging_penalty = sum(excess_charging_powers)

    def penalise_unwanted_vehicle_discharging(self, excess_discharging_powers):
        self.excess_vehicles_discharging_penalty = sum(excess_discharging_powers)

    def penalise_battery_issues(self, current_state_of_charge, depth_of_discharge, initial_power_demand,
                                remaining_power_demand, excess_charging_power, excess_discharging_power):
        self.penalise_unwanted_battery_charging(initial_power_demand, remaining_power_demand, excess_charging_power)
        self.penalise_unwanted_battery_discharging(initial_power_demand, remaining_power_demand, excess_discharging_power)
        self.penalise_battery_state_below_depth_of_discharge(current_state_of_charge, depth_of_discharge)

    def penalise_unwanted_battery_charging(self, initial_power, remaining_power, excess_charging_power):
        if 0 <= initial_power < remaining_power:
            self.needless_battery_charging_penalty = remaining_power - initial_power
            self.excess_battery_charging_penalty = 0.0
        elif initial_power <= 0 < remaining_power:
            self.needless_battery_charging_penalty = 0.0
            self.excess_battery_charging_penalty = remaining_power
        else:
            self.needless_battery_charging_penalty = 0.0
            self.excess_battery_charging_penalty = 0.0

        self.excess_battery_charging_penalty = self.excess_battery_charging_penalty + excess_charging_power

    def penalise_unwanted_battery_discharging(self, initial_power, remaining_power, excess_discharging_power):
        if remaining_power < initial_power <= 0:
            self.needless_battery_discharging_penalty = initial_power - remaining_power
            self.excess_battery_discharging_penalty = 0.0
        elif remaining_power < 0 <= initial_power:
            self.needless_battery_discharging_penalty = 0.0
            self.excess_battery_discharging_penalty = -remaining_power
        else:
            self.needless_battery_discharging_penalty = 0.0
            self.excess_battery_discharging_penalty = 0.0

        self.excess_battery_discharging_penalty = self.excess_battery_discharging_penalty + excess_discharging_power

    def penalise_battery_state_below_depth_of_discharge(self, current_state_of_charge, depth_of_discharge):
        if current_state_of_charge < depth_of_discharge:
            self.battery_state_of_charge_below_dod_penalty = ((depth_of_discharge - current_state_of_charge) * 2) ** 2
        else:
            self.battery_state_of_charge_below_dod_penalty = 0.0

    def get_insufficiently_charged_vehicles_penalty(self):
        return self.insufficiently_charged_vehicles_penalty

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
                                     + self.needless_battery_charging_penalty\
                                     + self.excess_battery_charging_penalty\
                                     + self.needless_battery_discharging_penalty\
                                     + self.excess_battery_discharging_penalty

    def calculate_total_vehicle_penalty(self):
        self.total_vehicle_penalty = self.total_insufficiently_charged_vehicles_penalty\
                                     + self.total_needless_vehicles_charging_penalty\
                                     + self.excess_vehicles_charging_penalty\
                                     + self.excess_vehicles_discharging_penalty
