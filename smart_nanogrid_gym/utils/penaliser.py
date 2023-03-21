class Penaliser:
    def __init__(self):
        self.insufficiently_charged_vehicles_penalty = 0
        self.battery_penalty = 0
        self.total_penalty = 0

    def calculate_insufficiently_charged_penalty(self, departing_vehicles, soc, timestep):
        penalties_per_departing_vehicle = []
        for vehicle in range(len(departing_vehicles)):
            penalty = self.calculate_insufficiently_charged_penalty_per_vehicle(departing_vehicles[vehicle], soc, timestep)
            penalties_per_departing_vehicle.append(penalty)

        self.insufficiently_charged_vehicles_penalty = sum(penalties_per_departing_vehicle)

    def calculate_insufficiently_charged_penalty_per_vehicle(self, vehicle, soc, timestep):
        uncharged_capacity = 1 - soc[vehicle, timestep - 1]
        penalty = (uncharged_capacity * 2) ** 2
        return penalty

    def penalise_battery_charging(self):
        pass

    def penalise_battery_discharging(self, battery_action):
        # Todo: Feat: Add penalty for positive action when trying to discharge battery
        if battery_action > 0:
            self.battery_penalty = battery_action
        else:
            self.battery_penalty = 0
            # Todo: Feat: Add battery_penalty = battery_action or different penalising strategy

    def get_total_penalty(self):
        self.calculate_total_penalty()
        return self.total_penalty

    def calculate_total_penalty(self):
        self.total_penalty = self.insufficiently_charged_vehicles_penalty + self.battery_penalty
