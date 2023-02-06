import numpy as np
import time


def calculate_charging_or_discharging_power(max_charging_power, action):
    return action * max_charging_power


def calculate_next_vehicle_state_of_charge(power_value, self, arrival, hour, soc):
    if hour in arrival:
        soc[hour] = soc[hour] + power_value / self.EV_PARAMETERS['CAPACITY']
    else:
        soc[hour] = soc[hour - 1] + power_value / self.EV_PARAMETERS['CAPACITY']
        # soc[charger, timestep] = soc[charger, timestep - 1] + (charging_power[charger] * time_interval) / \
        #                          self.EV_PARAMETERS['CAPACITY']


def calculate_max_charging_power(self, arrival, hour, soc):
    # max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'],
    #                           soc[charger, timestep] * self.EV_PARAMETERS['CAPACITY'] / time_interval])
    if hour in arrival:
        remaining_uncharged_capacity = 1 - soc[hour]
        power_left_to_charge = remaining_uncharged_capacity * self.EV_PARAMETERS['CAPACITY']
    else:
        remaining_uncharged_capacity = 1 - soc[hour - 1]
        power_left_to_charge = remaining_uncharged_capacity * self.EV_PARAMETERS['CAPACITY']

    max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'], power_left_to_charge])
    return max_charging_energy


def charge_vehicle(self, action, arrival, hour, vehicle_soc):
    max_charging_power = calculate_max_charging_power(self, arrival, hour, vehicle_soc)
    charging_power = calculate_charging_or_discharging_power(max_charging_power, action)
    calculate_next_vehicle_state_of_charge(charging_power, self, arrival, hour, vehicle_soc)
    return charging_power


def calculate_max_discharging_power(self, arrival, hour, soc):
    if hour in arrival:
        power_left_to_discharge = soc[hour] * self.EV_PARAMETERS['CAPACITY']
    else:
        power_left_to_discharge = soc[hour - 1] * self.EV_PARAMETERS['CAPACITY']

    max_discharging_energy = min([self.EV_PARAMETERS['MAX DISCHARGING POWER'], power_left_to_discharge])
    return max_discharging_energy


def discharge_vehicle(self, action, arrival, hour, vehicle_soc):
    max_discharging_power = calculate_max_discharging_power(self, arrival, hour, vehicle_soc)
    discharging_power = calculate_charging_or_discharging_power(max_discharging_power, action)
    calculate_next_vehicle_state_of_charge(discharging_power, self, arrival, hour, vehicle_soc)
    return discharging_power


def charge_or_discharge_vehicle(self, action, arrival, hour, vehicle_soc):
    if action >= 0:
        charger_power_value = charge_vehicle(self, action, arrival, hour, vehicle_soc)
    else:
        charger_power_value = discharge_vehicle(self, action, arrival, hour, vehicle_soc)

    return charger_power_value


def simulate_central_management_system(self, actions):
    # hour = self.timestep
    # timestep = self.timestep
    # time_interval = 1
    hour = self.timestep
    consumed = self.energy['Consumed']
    renewable = self.energy['Available renewable']
    charger_occupancy = self.initial_simulation_values['Charger occupancy']
    arrivals = self.initial_simulation_values['Arrivals']

    departing_vehicles = self.departing_vehicles
    soc = self.ev_state_of_charge

    charger_power_values = np.zeros(self.NUMBER_OF_CHARGERS)

    for charger in range(self.NUMBER_OF_CHARGERS):
        # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
        if charger_occupancy[charger, hour] == 1:
            charger_power_values[charger] = charge_or_discharge_vehicle(self, actions[charger], arrivals[charger],
                                                                        hour, soc[charger])
        else:
            charger_power_values[charger] = 0

    # ----------------------------------------------------------------------------
    # Calculation of energy utilization from the PV
    # Calculation of energy coming from Grid
    available_renewable_energy = max([0, renewable[0, hour] - consumed[0, hour]])
    total_charging_power = sum(charger_power_values)

    # ----------------------------------------------------------------------------
    # First Cost index
    grid_energy = max([total_charging_power - available_renewable_energy, 0])
    grid_energy_cost = grid_energy*self.energy["Price"][0, hour]

    # ----------------------------------------------------------------------------
    # Second Cost index
    # Penalty of wasted RES energy
    # This is not used in this environment version
    # RES_avail = max([RES_avail-Total_charging, 0])
    # Cost_2 = -RES_avail * (self.Energy["Price"][0, hour]/2)

    # ----------------------------------------------------------------------------
    # Third Cost index
    # Penalty of not fully charging the cars that leave
    penalties_per_departing_vehicle = []
    for vehicle in range(len(departing_vehicles)):
        penalties_per_departing_vehicle.append(((1-soc[departing_vehicles[vehicle], hour+1])*2)**2)

    insufficiently_charged_vehicles_penalty = sum(penalties_per_departing_vehicle)

    total_cost = grid_energy_cost + insufficiently_charged_vehicles_penalty

    return {
        'Total cost': total_cost,
        'Grid energy': grid_energy,
        'Utilized renewable energy': available_renewable_energy,
        'Insufficiently charged vehicles penalty': insufficiently_charged_vehicles_penalty,
        'EV state of charge': soc
    }
