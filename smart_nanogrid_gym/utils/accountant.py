from numpy import zeros, array, concatenate


class Accountant:
    def __init__(self):
        self.SELLING_ENERGY_TO_GRID_PRICE_COEFFICIENT = 0.8
        self.total_cost = 0.0
        self.grid_energy_cost = 0.0

        self.high_tariff = 0.0
        self.low_tariff = 0.0
        self.set_grid_tariffs()

        self.energy_price = zeros((1, 2 * 24))
        self.energy_price_max = 0.0

    def set_grid_tariffs(self):
        grid_tariff_high = 0.028
        grid_tariff_low = 0.013333333
        energy_tariff_high = 0.148933333
        energy_tariff_low = 0.087613333
        res_incentive = 0.014
        self.high_tariff = grid_tariff_high + energy_tariff_high + res_incentive
        self.low_tariff = grid_tariff_low + energy_tariff_low + res_incentive

    def calculate_grid_energy_cost(self, energy, price):
        if energy < 0:
            self.grid_energy_cost = energy * self.SELLING_ENERGY_TO_GRID_PRICE_COEFFICIENT * price
        else:
            self.grid_energy_cost = energy * price

        return self.grid_energy_cost

    def calculate_total_cost(self, additional_cost):
        self.total_cost = self.grid_energy_cost + additional_cost
        return self.total_cost

    def get_energy_price_at_time_t(self, t):
        # Todo: Adjust method to include different simulation days
        return self.energy_price[0, t]

    def get_normalised_energy_price_at_time_t(self, t):
        return self.energy_price[0, t] / self.energy_price_max

    def get_normalised_energy_price_in_range(self, min_timesteps_ahead, max_timesteps_ahead):
        return self.energy_price[0, min_timesteps_ahead:max_timesteps_ahead] / self.energy_price_max

    def set_energy_price(self, current_price_model, experiment_length_in_days, time_interval):
        self.energy_price = zeros((experiment_length_in_days, 2 * 24))
        self.initialise_energy_price(current_price_model, experiment_length_in_days, time_interval)
        self.energy_price_max = self.energy_price.max(where=(self.energy_price >= 0), initial=0)

    def initialise_energy_price(self, current_price_model, experiment_length_in_days, time_interval):
        price_day = self.get_price_day(current_price_model, time_interval)
        for day in range(0, experiment_length_in_days):
            self.energy_price[day, :] = price_day

    def get_price_day(self, current_price_model, time_interval):
        price_day = []
        if current_price_model == 0:
            tariffs = []
            for i in range(int(24 / time_interval)):
                if i < 7 / time_interval or i > 19 / time_interval:
                    tariffs.append(self.low_tariff)
                else:
                    tariffs.append(self.high_tariff)

            day_tariffs = array(tariffs)
            price_day = array([self.low_tariff, self.low_tariff, self.low_tariff, self.low_tariff, self.low_tariff,
                               self.low_tariff, self.low_tariff, self.high_tariff, self.high_tariff, self.high_tariff,
                               self.high_tariff, self.high_tariff, self.high_tariff, self.high_tariff, self.high_tariff,
                               self.high_tariff, self.high_tariff, self.high_tariff, self.high_tariff, self.high_tariff,
                               self.low_tariff, self.low_tariff, self.low_tariff, self.low_tariff])
        if current_price_model == 1:
            price_day = array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            # low high
        elif current_price_model == 2:
            price_day = array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06,
                               0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
            # dynamic
        elif current_price_model == 3:
            price_day = array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                               0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
            # dynamic
        elif current_price_model == 4:
            price_day = array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
            # dynamic
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
