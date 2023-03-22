class Penaliser:
    def __init__(self):
        self.insufficiently_charged_vehicles_penalty = 0
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
        uncharged_capacity = requested_end_soc - soc[vehicle, timestep - 1]
        charging_breathing_space = 0.05 * requested_end_soc

        if -charging_breathing_space <= uncharged_capacity <= charging_breathing_space:
            penalty = 0
        elif uncharged_capacity < -charging_breathing_space:
            penalty = (uncharged_capacity * 2)
        else:
            penalty = (uncharged_capacity * 2) ** 2

        return penalty

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
