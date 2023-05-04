import json

from numpy import random, zeros, asarray, ndarray, array, vstack
from scipy.io import loadmat, savemat

from smart_nanogrid_gym.utils.charger import Charger
from smart_nanogrid_gym.utils.config import data_files_directory_path
from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle


class ChargingStation:
    def __init__(self, number_of_chargers, time_interval, enable_different_vehicle_battery_capacities,
                 enable_requested_state_of_charge, charging_mode, vehicle_uncharged_penalty_mode):
        self.NUMBER_OF_CHARGERS = number_of_chargers
        self.array_columns = int(25 / time_interval)
        self.enable_different_vehicle_battery_capacities = enable_different_vehicle_battery_capacities
        self.enable_requested_state_of_charge = enable_requested_state_of_charge
        self.chargers = [Charger(charging_mode) for _ in range(self.NUMBER_OF_CHARGERS)]
        self.UNCHARGED_PENALTY_MODE = vehicle_uncharged_penalty_mode

        self.arrivals = []
        self.departures = []
        self.departure_times = []
        self.vehicle_state_of_charge_at_current_timestep = []
        self._departing_vehicles = []
        self._penalty_check_vehicles = []

        self.electric_vehicle_info = ElectricVehicle(battery_capacity=40, current_capacity=0, charging_efficiency=0.95,
                                                     discharging_efficiency=0.95, max_charging_power=22,
                                                     max_discharging_power=22, requested_end_capacity=0.8)
        self.generated_initial_values = {}
        self.generated_initial_values_json = {}

    def simulate(self, current_timestep, time_interval):
        self.find_vehicles_for_penalty_check(current_timestep, time_interval)
        # self.find_departing_vehicles(current_timestep, time_interval)  # Keep this to use if needed
        self.calculate_departure_times(current_timestep)
        self.extract_current_state_of_charge_per_vehicle(current_timestep)

        return self.departure_times, self.vehicle_state_of_charge_at_current_timestep

    def find_vehicles_for_penalty_check(self, timestep, time_interval):
        if timestep >= (24 / time_interval):
            return []

        self._penalty_check_vehicles.clear()
        for index, charger in enumerate(self.chargers):
            charger_occupied = charger.occupancy[timestep]

            if self.UNCHARGED_PENALTY_MODE == 'no_penalty':
                penalty_check_allowed = False
            elif self.UNCHARGED_PENALTY_MODE == 'on_departure':
                penalty_check_allowed = self.check_is_vehicle_departing(self.departures[index], timestep)
            elif self.UNCHARGED_PENALTY_MODE == 'sparse':
                penalty_check_allowed = self.check_is_vehicle_departing_in_next_n_timesteps(self.departures[index],
                                                                                            timestep, n=3)
            elif self.UNCHARGED_PENALTY_MODE == 'dense':
                penalty_check_allowed = True
            else:
                raise ValueError("Error: Wrong vehicle uncharged - penalty mode provided!")

            if charger_occupied and penalty_check_allowed:
                self._penalty_check_vehicles.append(index)

    def find_departing_vehicles(self, timestep, time_interval):
        if timestep >= (24 / time_interval):
            return []

        self._departing_vehicles.clear()
        for index, charger in enumerate(self.chargers):
            charger_occupied = charger.occupancy[timestep]

            vehicle_departing = self.check_is_vehicle_departing(self.departures[index], timestep)
            # vehicle_departing = self.check_is_vehicle_departing_in_next_n_timesteps(self.departures[index], timestep, n=3)

            if charger_occupied and vehicle_departing:
                self._departing_vehicles.append(index)

    def check_is_vehicle_departing(self, vehicle_departure, timestep):
        # Todo: Combine this and below method and include possibility for checking n timesteps ahead
        if timestep + 1 in vehicle_departure:
            return True
        else:
            return False

    def check_is_vehicle_departing_in_next_n_timesteps(self, vehicle_departure, timestep, n):
        if (timestep + 1 in vehicle_departure) or (timestep + 2 in vehicle_departure) or (timestep + 3 in vehicle_departure):
            return True
        else:
            return False

    def calculate_departure_times(self, timestep):
        self.departure_times.clear()
        for index, charger in enumerate(self.chargers):
            charger_occupied = charger.occupancy[timestep]

            if charger_occupied:
                departure_time = self.calculate_next_departure_time(self.departures[index], timestep)
                if isinstance(departure_time, ndarray):
                    breakpoint()
                self.departure_times.append(departure_time)
            else:
                self.departure_times.append(0)

    def calculate_next_departure_time(self, charger_departures, timestep):
        for departure_time in charger_departures:
            if timestep <= departure_time:
                a = departure_time - timestep
                if isinstance(a, ndarray):
                    breakpoint()
                return a
        return []

    def extract_current_state_of_charge_per_vehicle(self, timestep):
        self.vehicle_state_of_charge_at_current_timestep.clear()
        for charger in self.chargers:
            self.vehicle_state_of_charge_at_current_timestep.append(charger.vehicle_state_of_charge[timestep])

    def load_initial_values(self):
        self.clear_initialisation_variables()

        with open(data_files_directory_path + "\\initial_values.json", "r") as fp:
            initials = json.load(fp)

        self.arrivals = initials['Arrivals']
        self.departures = initials['Departures']

        vehicle_state_of_charge = array(initials['SOC'])
        charger_occupancy = array(initials['Charger_occupancy'])
        vehicle_capacities = array(initials['Vehicle_capacities'])

        for index, charger in enumerate(self.chargers):
            charger.vehicle_arrivals = self.arrivals[index]
            charger.vehicle_state_of_charge = vehicle_state_of_charge[index, :]
            charger.occupancy = charger_occupancy[index, :]
            charger.vehicle_capacities = vehicle_capacities[index, :]

    def clear_initialisation_variables(self):
        try:
            self.arrivals.clear()
            self.departures.clear()
            for charger in self.chargers:
                charger.vehicle_arrivals.clear()
                charger.vehicle_state_of_charge.fill(0)
                charger.vehicle_capacities.fill(0)
                charger.occupancy.fill(0)
                charger.requested_end_state_of_charge.fill(0)
            return True
        except ValueError:
            return False

    def generate_new_initial_values(self, time_interval):
        initial_variables_cleared = self.clear_initialisation_variables()
        initial_vehicle_presence_generated = self.generate_initial_vehicle_presence(initial_variables_cleared, time_interval)

        arrivals_array = asarray(self.arrivals, dtype=object)
        departures_array = asarray(self.departures, dtype=object)

        vehicles_state_of_charge = self.get_vehicles_state_of_charge()
        charger_occupancy = self.get_occupancy_for_all_chargers()
        vehicle_capacities = self.get_vehicle_capacities_for_all_chargers()
        requested_vehicle_state_of_charge = self.get_requested_end_state_of_charge_for_all_chargers()

        generated_initial_values = {
            'SOC': vehicles_state_of_charge,
            'Arrivals': arrivals_array,
            'Departures': departures_array,
            'Charger_occupancy': charger_occupancy,
            'Vehicle_capacities': vehicle_capacities,
            'Requested_SOC': requested_vehicle_state_of_charge
        } if initial_vehicle_presence_generated else {}

        generated_initial_values_json = {
            'SOC': vehicles_state_of_charge.tolist(),
            'Arrivals': self.arrivals,
            'Departures': self.departures,
            'Charger_occupancy': charger_occupancy.tolist(),
            'Vehicle_capacities': vehicle_capacities.tolist(),
            'Requested_SOC': requested_vehicle_state_of_charge.tolist()
        } if initial_vehicle_presence_generated else {}

        self.generated_initial_values = generated_initial_values
        self.generated_initial_values_json = generated_initial_values_json

        with open(data_files_directory_path + "\\initial_values.json", "w") as fp:
            json.dump(generated_initial_values_json, fp, indent=4)

    def save_initial_values_to_json_file(self, path, filename):
        prefix = f'\\{filename}-' if filename else ''
        with open(path + f'\\{prefix}initial_values.json', "w") as fp:
            json.dump(self.generated_initial_values_json, fp, indent=4)

    def generate_initial_vehicle_presence(self, initial_variables_cleared, time_interval):
        if initial_variables_cleared:
            for charger in self.chargers:
                self.generate_initial_vehicle_presence_per_charger(charger, time_interval)
            return True
        return False

    def generate_initial_vehicle_presence_per_charger(self, charger, time_interval):
        vehicle_arrivals = []
        vehicle_departures = []

        vehicle_present = False
        current_departure_time = 0
        current_vehicle_capacity = 0
        vehicle_capacity_generated = False
        current_requested_end_of_charge = 0
        requested_soc_generated = False

        total_timesteps = int(24 / time_interval)
        for timestep in range(total_timesteps):
            if not vehicle_present:
                arrival = round(random.rand() - 0.1)
                if arrival == 1 and timestep < total_timesteps:
                    vehicle_present = True

                    self.generate_random_arrival_vehicle_state_of_charge(charger, timestep)
                    self.generate_random_requested_end_vehicle_state_of_charge(charger, timestep)
                    if self.enable_different_vehicle_battery_capacities and not vehicle_capacity_generated:
                        current_vehicle_capacity = self.generate_random_vehicle_battery_capacity()
                        vehicle_capacity_generated = True
                    elif not self.enable_different_vehicle_battery_capacities and not vehicle_capacity_generated:
                        current_vehicle_capacity = 40
                        vehicle_capacity_generated = True

                    if self.enable_requested_state_of_charge and not requested_soc_generated:
                        current_requested_end_of_charge = self.generate_random_requested_end_vehicle_state_of_charge(charger, timestep)
                        requested_soc_generated = True
                    elif not self.enable_requested_state_of_charge and not requested_soc_generated:
                        current_requested_end_of_charge = 1.0
                        requested_soc_generated = True

                    vehicle_arrivals.append(timestep)

                    current_departure_time = self.generate_random_vehicle_departure_time(timestep, time_interval, total_timesteps)
                    vehicle_departures.append(current_departure_time)

            if vehicle_present and timestep < current_departure_time:
                charger.occupancy[timestep] = 1
                charger.vehicle_capacities[timestep] = current_vehicle_capacity
                charger.requested_end_state_of_charge[timestep] = current_requested_end_of_charge
            else:
                vehicle_present = False
                charger.occupancy[timestep] = 0
                charger.vehicle_capacities[timestep] = 0
                vehicle_capacity_generated = False
                current_vehicle_capacity = 0.0
                charger.requested_end_state_of_charge[timestep] = 0
                current_requested_end_of_charge = 0
                requested_soc_generated = False

        self.arrivals.append(vehicle_arrivals)
        self.departures.append(vehicle_departures)
        charger.vehicle_arrivals.extend(vehicle_arrivals)

    def generate_random_arrival_vehicle_state_of_charge(self, charger, timestep):
        random_state_of_charge = random.uniform(0.1, 0.9)
        charger.vehicle_state_of_charge[timestep] = random_state_of_charge

    def generate_random_requested_end_vehicle_state_of_charge(self, charger, timestep):
        arrival_state_of_charge = charger.vehicle_state_of_charge[timestep]
        arrival_state_of_charge = arrival_state_of_charge + 0.1 if arrival_state_of_charge <= 0.9 else 1.0
        random_state_of_charge = random.uniform(arrival_state_of_charge, 1.0)
        return random_state_of_charge

    def generate_random_vehicle_battery_capacity(self):
        random_vehicle_capacity = random.randint(15, 120)
        return random_vehicle_capacity

    def generate_random_vehicle_departure_time(self, timestep, time_interval, total_timesteps):
        max_charging_time = timestep + int(10 / time_interval)
        max_departing_time = total_timesteps + int(1 / time_interval)
        upper_limit = min(max_charging_time, max_departing_time)
        low = timestep + int(4 / time_interval)
        high = int(upper_limit)
        if low >= high:
            return int(low)
        return random.randint(low, high)

    def simulate_vehicle_charging(self, actions, current_timestep, time_interval):
        charger_power_values = zeros(self.NUMBER_OF_CHARGERS)

        for index, charger in enumerate(self.chargers):
            action = actions[index]
            # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
            if charger.occupancy[current_timestep] == 1:
                charger_power_values[index] = charger.charge_or_discharge_vehicle(action, current_timestep, time_interval)
            else:
                charger_power_values[index] = 0
                charger.reset_info_values(action)

        total_discharging_power = charger_power_values[charger_power_values < 0].sum()
        total_charging_power = charger_power_values[charger_power_values > 0].sum()

        return {
            'Total charging power': total_charging_power,
            'Total discharging power': total_discharging_power,
            'Charger power values': charger_power_values.tolist()
        }

    def get_vehicles_state_of_charge(self):
        vehicle_state_of_charge = vstack([charger.vehicle_state_of_charge for charger in self.chargers])
        return vehicle_state_of_charge

    def get_occupancy_for_all_chargers(self):
        occupancy = vstack([charger.occupancy for charger in self.chargers])
        return occupancy

    def get_vehicle_capacities_for_all_chargers(self):
        capacities = vstack([charger.vehicle_capacities for charger in self.chargers])
        return capacities

    def get_requested_end_state_of_charge_for_all_chargers(self):
        requested_soc = vstack([charger.requested_end_state_of_charge for charger in self.chargers])
        return requested_soc

    def get_all_departing_vehicles(self):
        return self._departing_vehicles

    def get_info_for_penalisation(self):
        vehicles_for_penalty_check = self._penalty_check_vehicles
        vehicles_state_of_charge = self.get_vehicles_state_of_charge()
        requested_end_state_of_charge_per_charger = self.get_requested_end_state_of_charge_for_all_chargers()
        arrivals = self.arrivals
        excess_charging_powers = self.get_excess_vehicles_charging_power_per_charger()
        excess_discharging_powers = self.get_excess_vehicles_discharging_power_per_charger()
        charging_nonexistent_vehicles = self.get_charging_non_existent_vehicles()

        return {
            'penalty_check_vehicles': vehicles_for_penalty_check,
            'states_of_charge': vehicles_state_of_charge,
            'requested_end_states_of_charge': requested_end_state_of_charge_per_charger,
            'vehicle_arrivals': arrivals,
            'excess_charging_powers': excess_charging_powers,
            'excess_discharging_powers': excess_discharging_powers,
            'charging_nonexistent_vehicles': charging_nonexistent_vehicles
        }

    def get_excess_vehicles_charging_power_per_charger(self):
        return [charger.excess_vehicle_charging_power for charger in self.chargers]

    def get_excess_vehicles_discharging_power_per_charger(self):
        return [charger.excess_vehicle_discharging_power for charger in self.chargers]

    def get_charging_non_existent_vehicles(self):
        return [charger.charging_non_existent_vehicle for charger in self.chargers]
