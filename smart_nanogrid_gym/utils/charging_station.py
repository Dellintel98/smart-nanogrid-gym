import time

import numpy as np
from numpy import random, arange
from scipy.io import loadmat, savemat

from smart_nanogrid_gym.utils.charger import Charger
from smart_nanogrid_gym.utils.config import data_files_directory_path
from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle


class ChargingStation:
    def __init__(self, number_of_chargers):
        self.NUMBER_OF_CHARGERS = number_of_chargers
        self.chargers = [Charger() for _ in range(self.NUMBER_OF_CHARGERS)]
        self.vehicle_state_of_charge = np.zeros([self.NUMBER_OF_CHARGERS, 25])
        self.charger_occupancy = np.zeros([self.NUMBER_OF_CHARGERS, 25])
        self.arrivals = []
        self.departures = []
        self.total_vehicles_charging = np.zeros(24)
        self.departing_vehicles = []
        self.departure_times = []
        self.vehicle_state_of_charge_at_current_timestep = []

        self.electric_vehicle_info = ElectricVehicle(battery_capacity=40, current_capacity=0, charging_efficiency=0.95,
                                                     discharging_efficiency=0.95, max_charging_power=22,
                                                     max_discharging_power=22)

    def simulate(self, current_timestep):
        self.find_departing_vehicles(current_timestep)
        self.calculate_departure_times(current_timestep)
        self.calculate_state_of_charge_for_each_vehicle(current_timestep)

        return self.departure_times, self.vehicle_state_of_charge_at_current_timestep

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

    def load_initial_values(self):
        self.clear_initialisation_variables()

        initial_values = loadmat(data_files_directory_path + '\\initial_values.mat')

        arrival_times = initial_values['Arrivals']
        departure_times = initial_values['Departures']

        self.vehicle_state_of_charge = initial_values['SOC']
        self.charger_occupancy = initial_values['Charger_occupancy']
        self.total_vehicles_charging = initial_values['Total_vehicles_charging']

        for charger in range(self.NUMBER_OF_CHARGERS):
            if arrival_times.shape == (1, self.NUMBER_OF_CHARGERS):
                arrivals = arrival_times[0][charger][0]
                departures = departure_times[0][charger][0]
            elif arrival_times.shape == (self.NUMBER_OF_CHARGERS, 3):
                arrivals = arrival_times[charger]
                departures = departure_times[charger]
            else:
                raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

            self.arrivals.append(arrivals.tolist())
            self.departures.append(departures.tolist())
            self.chargers[charger].vehicle_arrivals = self.arrivals[charger]
            self.chargers[charger].vehicle_state_of_charge = self.vehicle_state_of_charge[charger, :]
            self.chargers[charger].occupancy = initial_values['Charger_occupancy'][charger, :]

    def clear_initialisation_variables(self):
        try:
            self.arrivals.clear()
            self.departures.clear()
            self.charger_occupancy.fill(0)
            self.vehicle_state_of_charge.fill(0)
            self.total_vehicles_charging = np.zeros(24)
            return True
        except ValueError:
            return False

    def generate_new_initial_values(self):
        initial_variables_cleared = self.clear_initialisation_variables()
        initial_vehicle_presence_generated = self.generate_initial_vehicle_presence(initial_variables_cleared)
        self.calculate_initial_total_vehicles_charging(initial_vehicle_presence_generated)

        generated_initial_values = {
            'SOC': self.vehicle_state_of_charge,
            'Arrivals': self.arrivals,
            'Departures': self.departures,
            'Charger_occupancy': self.charger_occupancy,
            'Total_vehicles_charging': self.total_vehicles_charging
        }

        savemat(data_files_directory_path + '\\initial_values.mat', generated_initial_values)

    def generate_initial_vehicle_presence(self, initial_variables_cleared):
        if initial_variables_cleared:
            for charger in range(self.NUMBER_OF_CHARGERS):
                self.generate_initial_vehicle_presence_per_charger(charger)
            return True
        return False

    def generate_initial_vehicle_presence_per_charger(self, charger):
        vehicle_arrivals = []
        vehicle_departures = []

        vehicle_present = False
        current_departure_time = 0

        for hour in range(24):
            if not vehicle_present:
                arrival = round(random.rand() - 0.1)
                if arrival == 1 and hour <= 20:
                    vehicle_present = True

                    self.generate_random_arrival_vehicle_state_of_charge(charger, hour)
                    vehicle_arrivals.append(hour)

                    current_departure_time = self.generate_random_vehicle_departure_time(hour)
                    vehicle_departures.append(current_departure_time)

            if vehicle_present and hour < current_departure_time:
                self.charger_occupancy[charger, hour] = 1
                self.chargers[charger].occupancy[hour] = 1
            else:
                vehicle_present = False
                self.charger_occupancy[charger, hour] = 0
                self.chargers[charger].occupancy[hour] = 0

        self.arrivals.append(vehicle_arrivals)
        self.departures.append(vehicle_departures)
        self.chargers[charger].vehicle_arrivals.extend(vehicle_arrivals)

    def generate_random_arrival_vehicle_state_of_charge(self, charger, hour):
        random_integer = random.randint(20, 50)
        self.vehicle_state_of_charge[charger, hour] = random_integer / 100
        self.chargers[charger].vehicle_state_of_charge[hour] = self.vehicle_state_of_charge[charger, hour]

    def generate_random_vehicle_departure_time(self, hour):
        upper_limit = min(hour + 10, 25)
        return random.randint(hour + 4, int(upper_limit))

    def calculate_initial_total_vehicles_charging(self, initial_vehicle_presence_generated):
        if initial_vehicle_presence_generated:
            for hour in range(24):
                hour_occupancy = self.charger_occupancy[:, hour]
                occupancy_total = hour_occupancy.sum()
                self.total_vehicles_charging[hour] = occupancy_total

    def simulate_vehicle_charging(self, actions, current_timestep):
        # hour = self.timestep
        # timestep = self.timestep
        # time_interval = 1
        hour = current_timestep

        charger_power_values = np.zeros(self.NUMBER_OF_CHARGERS)

        for index, charger in enumerate(self.chargers):
            # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
            if charger.occupancy[hour] == 1:
                charger_power_values[index] = charger.charge_or_discharge_vehicle(actions[index], hour)
            else:
                charger_power_values[index] = 0

        total_discharging_power = charger_power_values[charger_power_values < 0].sum()
        total_charging_power = charger_power_values[charger_power_values > 0].sum()

        return total_charging_power, total_discharging_power
