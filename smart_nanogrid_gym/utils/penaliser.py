class Penaliser:
    def __init__(self):
        self.insufficiently_charged_vehicles_penalty = 0
        self.state_of_charge_above_1_penalty = 0
        self.battery_penalty = 0
        self.total_penalty = 0

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

    def penalise_vehicle_charging_max_state_of_charge(self, vehicle_soc):
        if vehicle_soc > 1.0:
            self.state_of_charge_above_1_penalty = (vehicle_soc * 3) ** 2
        else:
            self.state_of_charge_above_1_penalty = 0

    def penalise_battery_charging(self, positive_battery_action):
        if positive_battery_action < 0:
            self.battery_penalty = -positive_battery_action
        else:
            self.battery_penalty = 0
            # Todo: Feat: Add battery_penalty = battery_action or different penalising strategy

    def penalise_battery_discharging(self, negative_battery_action):
        if negative_battery_action > 0:
            self.battery_penalty = negative_battery_action
        else:
            self.battery_penalty = 0
            # Todo: Feat: Add battery_penalty = battery_action or different penalising strategy

    def get_total_penalty(self):
        self.calculate_total_penalty()
        return self.total_penalty

    def calculate_total_penalty(self):
        self.total_penalty = self.insufficiently_charged_vehicles_penalty + self.battery_penalty