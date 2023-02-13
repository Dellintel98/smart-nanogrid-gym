class CentralManagementSystem:
    def __init__(self):
        self.total_cost = 0
        self.grid_energy_cost = 0

    def calculate_available_renewable_energy(self, renewable, consumed):
        available_energy = renewable - consumed
        return max([0, available_energy])

    def calculate_grid_energy(self, total_power, available_renewable_energy):
        grid_energy = total_power - available_renewable_energy
        return max([grid_energy, 0])

    def calculate_grid_energy_cost(self, grid_energy, price):
        self.grid_energy_cost = grid_energy * price

    def calculate_insufficiently_charged_penalty_per_vehicle(self, vehicle, soc, hour):
        uncharged_capacity = 1 - soc[vehicle, hour + 1]
        penalty = (uncharged_capacity * 2) ** 2
        return penalty

    def calculate_insufficiently_charged_penalty(self, departing_vehicles, soc, hour):
        penalties_per_departing_vehicle = []
        for vehicle in range(len(departing_vehicles)):
            penalty = self.calculate_insufficiently_charged_penalty_per_vehicle(departing_vehicles[vehicle], soc, hour)
            penalties_per_departing_vehicle.append(penalty)

        return sum(penalties_per_departing_vehicle)

    def calculate_total_cost(self, total_penalty):
        self.total_cost = self.grid_energy_cost + total_penalty

    def simulate(self, current_timestep, total_charging_power, energy,
                 departing_vehicles, soc):
        # hour = self.timestep
        # timestep = self.timestep
        # time_interval = 1
        hour = current_timestep
        consumed = energy['Consumed']
        renewable = energy['Available renewable']

        available_renewable_energy = self.calculate_available_renewable_energy(renewable[0, hour], consumed[0, hour])
        grid_energy = self.calculate_grid_energy(total_charging_power, available_renewable_energy)

        self.calculate_grid_energy_cost(grid_energy, energy["Price"][0, hour])
        insufficiently_charged_vehicles_penalty = self.calculate_insufficiently_charged_penalty(departing_vehicles, soc,
                                                                                                hour)

        self.calculate_total_cost(insufficiently_charged_vehicles_penalty)

        return {
            'Total cost': self.total_cost,
            'Grid energy': grid_energy,
            'Utilized renewable energy': available_renewable_energy,
            'Insufficiently charged vehicles penalty': insufficiently_charged_vehicles_penalty,
            'EV state of charge': soc
        }
