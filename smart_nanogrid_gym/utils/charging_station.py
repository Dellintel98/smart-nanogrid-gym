import numpy as np
from numpy import random
from scipy.io import loadmat, savemat


class ChargingStation:
    def __init__(self, number_of_chargers):
        self.NUMBER_OF_CHARGERS = number_of_chargers
        self.vehicle_state_of_charge = None
        self.arrivals = None
        self.departures = None
        self.charger_occupancy = None
        self.total_vehicles_charging = None
        self.departing_vehicles = []
        self.departure_times = []
        self.vehicle_state_of_charge_at_current_timestep = []

    def simulate(self, current_timestep):
        self.find_departing_vehicles(current_timestep)
        self.calculate_departure_times(current_timestep)
        self.calculate_state_of_charge_for_each_vehicle(current_timestep)

        return self.departing_vehicles, self.departure_times, self.vehicle_state_of_charge_at_current_timestep

    def find_departing_vehicles(self, hour):
        if hour >= 24:
            return []

        self.departing_vehicles.clear()
        for charger in range(self.NUMBER_OF_CHARGERS):
            charger_occupied = self.check_charger_occupancy(self.charger_occupancy[charger, hour])
            vehicle_departing = self.check_is_vehicle_departing(self.departures[charger], hour)

            if charger_occupied and vehicle_departing:
                self.departing_vehicles.append(charger)

    def check_charger_occupancy(self, charger_occupancy):
        if charger_occupancy == 1:
            return True
        else:
            return False

    def check_is_vehicle_departing(self, vehicle_departure, hour):
        if hour + 1 in vehicle_departure:
            return True
        else:
            return False

    def calculate_departure_times(self, hour):
        self.departure_times.clear()
        for charger in range(self.NUMBER_OF_CHARGERS):
            charger_occupied = self.check_charger_occupancy(self.charger_occupancy[charger, hour])

            if charger_occupied:
                departure_time = self.calculate_next_departure_time(self.departures[charger], hour)
                self.departure_times.append(departure_time)
            else:
                self.departure_times.append(0)

    def calculate_next_departure_time(self, charger_departures, hour):
        for vehicle in range(len(charger_departures)):
            if hour <= charger_departures[vehicle]:
                return charger_departures[vehicle] - hour
        return []

    def calculate_state_of_charge_for_each_vehicle(self, hour):
        self.vehicle_state_of_charge_at_current_timestep.clear()
        for charger in range(self.NUMBER_OF_CHARGERS):
            self.vehicle_state_of_charge_at_current_timestep.append(self.vehicle_state_of_charge[charger, hour])

    def load_initial_values(self, file_directory_path):
        initial_values = loadmat(file_directory_path + '\\initial_values.mat')

        arrival_times = initial_values['Arrivals']
        departure_times = initial_values['Departures']

        reformatted_arrivals = []
        reformatted_departures = []

        for charger in range(self.NUMBER_OF_CHARGERS):
            if arrival_times.shape == (1, 10):
                arrivals = arrival_times[0][charger][0]
                departures = departure_times[0][charger][0]
            elif arrival_times.shape == (10, 3):
                arrivals = arrival_times[charger]
                departures = departure_times[charger]
            else:
                raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

            reformatted_arrivals.append(arrivals.tolist())
            reformatted_departures.append(departures.tolist())

        self.vehicle_state_of_charge = initial_values['SOC']
        self.arrivals = reformatted_arrivals
        self.departures = reformatted_departures
        self.charger_occupancy = initial_values['Charger occupancy']
        self.total_vehicles_charging = initial_values['Total vehicles charging']

    def generate_new_values(self, file_directory_path):
        arrivals = []
        departures = []

        vehicle_state_of_charge = np.zeros([self.NUMBER_OF_CHARGERS, 25])
        charger_occupancy = np.zeros([self.NUMBER_OF_CHARGERS, 25])

        # initial state stochastic creation
        for charger in range(self.NUMBER_OF_CHARGERS):
            is_vehicle_present = 0
            total_occupancy_timesteps_per_charger = 0
            arrival = 0

            vehicle_arrivals = []
            vehicle_departures = []

            for hour in range(24):
                if is_vehicle_present == 0:
                    arrival = round(random.rand() - 0.1)
                    if arrival == 1 and hour <= 20:
                        random_integer = random.randint(20, 50)
                        vehicle_state_of_charge[charger, hour] = random_integer / 100

                        total_occupancy_timesteps_per_charger = total_occupancy_timesteps_per_charger + 1

                        vehicle_arrivals.append(hour)
                        upper_limit = min(hour + 10, 25)
                        vehicle_departures.append(random.randint(hour + 4, int(upper_limit)))

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

        self.vehicle_state_of_charge = vehicle_state_of_charge
        self.arrivals = arrivals
        self.departures = departures
        self.charger_occupancy = charger_occupancy
        self.total_vehicles_charging = total_vehicles_charging

        generated_initial_values = {
            'SOC': vehicle_state_of_charge,
            'Arrivals': arrivals,
            'Departures': departures,
            'Total vehicles charging': total_vehicles_charging,
            'Charger occupancy': charger_occupancy
        }

        savemat(file_directory_path + '\\initial_values.mat', generated_initial_values)
