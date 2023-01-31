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
        pointer = 0

        arrival_car = []
        departure_car = []

        for hour in range(24):
            if is_vehicle_present == 0:
                arrival = round(random.rand()-0.1)
                if arrival == 1 and hour <= 20:
                    ran = random.randint(20, 50)
                    vehicle_state_of_charge[charger, hour] = ran / 100
                    pointer = pointer+1
                    arrival_car.append(hour)
                    upper_limit = min(hour + 10, 25)
                    departure_car.append(random.randint(hour+4, int(upper_limit)))

            if arrival == 1 and pointer > 0:
                if hour < departure_car[pointer - 1]:
                    is_vehicle_present = 1
                    charger_occupancy[charger, hour] = 1
                else:
                    is_vehicle_present = 0
                    charger_occupancy[charger, hour] = 0
            else:
                is_vehicle_present = 0
                charger_occupancy[charger, hour] = 0

        arrivals.append(arrival_car)
        departures.append(departure_car)

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
