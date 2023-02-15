import numpy as np
from numpy import random
from scipy.io import loadmat, savemat
from .config import data_files_directory_path
from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle


class ChargingStation:
    def __init__(self, number_of_chargers):
        self.NUMBER_OF_CHARGERS = number_of_chargers
        self.vehicle_state_of_charge = np.zeros([self.NUMBER_OF_CHARGERS, 25])
        self.charger_occupancy = np.zeros([self.NUMBER_OF_CHARGERS, 25])
        self.arrivals = []
        self.departures = []
        self.total_vehicles_charging = np.zeros([24])
        self.departing_vehicles = []
        self.departure_times = []
        self.vehicle_state_of_charge_at_current_timestep = []
        self.charger_power_values = np.zeros(self.NUMBER_OF_CHARGERS)

        self.electric_vehicle_info = ElectricVehicle(battery_capacity=30, charging_efficiency=0.95,
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

        for charger in range(self.NUMBER_OF_CHARGERS):
            if arrival_times.shape == (1, 10):
                arrivals = arrival_times[0][charger][0]
                departures = departure_times[0][charger][0]
            elif arrival_times.shape == (10, 3):
                arrivals = arrival_times[charger]
                departures = departure_times[charger]
            else:
                raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

            self.arrivals.append(arrivals.tolist())
            self.departures.append(departures.tolist())

        self.vehicle_state_of_charge = initial_values['SOC']
        self.charger_occupancy = initial_values['Charger occupancy']
        self.total_vehicles_charging = initial_values['Total vehicles charging']

    def clear_initialisation_variables(self):
        try:
            self.arrivals.clear()
            self.departures.clear()
            self.charger_occupancy.fill(0)
            self.vehicle_state_of_charge.fill(0)
            self.total_vehicles_charging.fill(0)
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
            'Charger occupancy': self.charger_occupancy,
            'Total vehicles charging': self.total_vehicles_charging
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
            else:
                vehicle_present = False
                self.charger_occupancy[charger, hour] = 0

        self.arrivals.append(vehicle_arrivals)
        self.departures.append(vehicle_departures)

    def generate_random_arrival_vehicle_state_of_charge(self, charger, hour):
        random_integer = random.randint(20, 50)
        self.vehicle_state_of_charge[charger, hour] = random_integer / 100

    def generate_random_vehicle_departure_time(self, hour):
        upper_limit = min(hour + 10, 25)
        return random.randint(hour + 4, int(upper_limit))

    def calculate_initial_total_vehicles_charging(self, initial_vehicle_presence_generated):
        if initial_vehicle_presence_generated:
            for hour in range(24):
                self.total_vehicles_charging[hour] = np.sum(self.charger_occupancy[:, hour])

    def simulate_vehicle_charging(self, actions, current_timestep):
        # hour = self.timestep
        # timestep = self.timestep
        # time_interval = 1
        hour = current_timestep

        self.charger_power_values = np.zeros(self.NUMBER_OF_CHARGERS)

        for charger in range(self.NUMBER_OF_CHARGERS):
            # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
            if self.charger_occupancy[charger, hour] == 1:
                self.charger_power_values[charger] = self.charge_or_discharge_vehicle(actions[charger], hour, charger)
            else:
                self.charger_power_values[charger] = 0

        total_charging_power = self.charger_power_values.sum()
        # total_charging_power = sum(charger_power_values)

        return total_charging_power

    def charge_or_discharge_vehicle(self, action, hour, charger):
        if action >= 0:
            charger_power_value = self.charge_vehicle(action, hour, charger)
        else:
            charger_power_value = self.discharge_vehicle(action, hour, charger)

        return charger_power_value

    def charge_vehicle(self, action, hour, charger):
        max_charging_power = self.calculate_max_charging_power(hour, charger)
        charging_power = self.calculate_charging_or_discharging_power(max_charging_power, action)
        self.calculate_next_vehicle_state_of_charge(charging_power, hour, charger)
        return charging_power

    def calculate_max_charging_power(self, hour, charger):
        # max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'],
        #                           soc[charger, timestep] * self.EV_PARAMETERS['CAPACITY'] / time_interval])
        if hour in self.arrivals[charger]:
            remaining_uncharged_capacity = 1 - self.vehicle_state_of_charge[charger, hour]
            power_left_to_charge = remaining_uncharged_capacity * self.electric_vehicle_info.battery_capacity
        else:
            remaining_uncharged_capacity = 1 - self.vehicle_state_of_charge[charger, hour - 1]
            power_left_to_charge = remaining_uncharged_capacity * self.electric_vehicle_info.battery_capacity

        max_charging_energy = min([self.electric_vehicle_info.max_charging_power, power_left_to_charge])
        return max_charging_energy

    def calculate_charging_or_discharging_power(self, max_charging_power, action):
        return action * max_charging_power

    def calculate_next_vehicle_state_of_charge(self, power_value, hour, charger):
        if hour in self.arrivals[charger]:
            self.vehicle_state_of_charge[charger, hour] = self.vehicle_state_of_charge[
                                                              charger, hour] + power_value / self.electric_vehicle_info.battery_capacity
        else:
            self.vehicle_state_of_charge[charger, hour] = self.vehicle_state_of_charge[
                                                              charger, hour - 1] + power_value / self.electric_vehicle_info.battery_capacity
            # soc[charger, timestep] = soc[charger, timestep - 1] + (charging_power[charger] * time_interval) / \
            #                          self.EV_PARAMETERS['CAPACITY']

    def discharge_vehicle(self, action, hour, charger):
        max_discharging_power = self.calculate_max_discharging_power(hour, charger)
        discharging_power = self.calculate_charging_or_discharging_power(max_discharging_power, action)
        self.calculate_next_vehicle_state_of_charge(discharging_power, hour, charger)
        return discharging_power

    def calculate_max_discharging_power(self, hour, charger):
        if hour in self.arrivals[charger]:
            power_left_to_discharge = self.vehicle_state_of_charge[
                                          charger, hour] * self.electric_vehicle_info.battery_capacity
        else:
            power_left_to_discharge = self.vehicle_state_of_charge[
                                          charger, hour - 1] * self.electric_vehicle_info.battery_capacity

        max_discharging_energy = min([self.electric_vehicle_info.max_discharging_power, power_left_to_discharge])
        return max_discharging_energy
