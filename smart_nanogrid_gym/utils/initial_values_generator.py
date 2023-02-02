import numpy as np
from numpy import random
import scipy.io


def generate_new_values(self):
    number_of_chargers = self.NUMBER_OF_CHARGERS
    arrivals = []
    departures = []

    vehicle_state_of_charge = np.zeros([number_of_chargers, 25])
    charger_occupancy = np.zeros([number_of_chargers, 25])

    # initial state stochastic creation
    for charger in range(number_of_chargers):
        is_vehicle_present = 0
        total_occupancy_timesteps_per_charger = 0
        arrival = 0

        vehicle_arrivals = []
        vehicle_departures = []

        for hour in range(24):
            if is_vehicle_present == 0:
                arrival = round(random.rand()-0.1)
                if arrival == 1 and hour <= 20:
                    random_integer = random.randint(20, 50)
                    vehicle_state_of_charge[charger, hour] = random_integer / 100

                    total_occupancy_timesteps_per_charger = total_occupancy_timesteps_per_charger+1

                    vehicle_arrivals.append(hour)
                    upper_limit = min(hour + 10, 25)
                    vehicle_departures.append(random.randint(hour+4, int(upper_limit)))

            if arrival == 1 and total_occupancy_timesteps_per_charger > 0:
                if hour < vehicle_departures[total_occupancy_timesteps_per_charger - 1]:
                    is_vehicle_present = 1
                    charger_occupancy[charger, hour] = 1
                else:
                    is_vehicle_present = 0
                    charger_occupancy[charger, hour] = 0
            else:
                is_vehicle_present = 0
                charger_occupancy[charger, hour] = 0

        arrivals.append(vehicle_arrivals)
        departures.append(vehicle_departures)

    # information vector creator
    total_vehicles_charging = np.zeros([24])
    for hour in range(24):
        total_vehicles_charging[hour] = np.sum(charger_occupancy[:, hour])

    return {
        'SOC': vehicle_state_of_charge,
        'Arrivals': arrivals,
        'Departures': departures,
        'Total vehicles charging': total_vehicles_charging,
        'Charger occupancy': charger_occupancy
    }
