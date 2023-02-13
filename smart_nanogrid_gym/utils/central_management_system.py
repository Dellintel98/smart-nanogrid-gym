from numpy import zeros, array, concatenate


class CentralManagementSystem:
    def __init__(self):
        self.total_cost = 0
        self.grid_energy_cost = 0

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

    def simulate(self, current_timestep, total_charging_power, energy, energy_price,
                 departing_vehicles, soc):
        # hour = self.timestep
        # timestep = self.timestep
        # time_interval = 1
        hour = current_timestep
        renewable = energy['Available renewable']

        available_renewable_energy = renewable[0, hour]
        grid_energy = self.calculate_grid_energy(total_charging_power, available_renewable_energy)

        self.calculate_grid_energy_cost(grid_energy, energy_price[0, hour])
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

    def get_price_day(self, current_price_model):
        # high_tariff = 0.028 + 0.148933
        # low_tariff = 0.013333 + 0.087613
        price_day = []
        # if current_price_model == 0:
        #     price_day = np.array([low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff,
        #                           high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
        #                           high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
        #                           low_tariff, low_tariff, low_tariff, low_tariff])
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

    def get_energy_price(self, current_price_model, experiment_length_in_days):
        price_day = self.get_price_day(current_price_model)
        price = zeros((experiment_length_in_days, 2 * 24))
        for day in range(0, experiment_length_in_days):
            price[day, :] = price_day
        return price
