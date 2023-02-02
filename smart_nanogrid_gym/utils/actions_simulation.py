import numpy as np
import time


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

    charging_power = np.zeros(self.NUMBER_OF_CHARGERS)

    # ----------------------------------------------------------------------------
    # Calculation of demand based on actions
    # Calculation of actions for cars
    for charger in range(self.NUMBER_OF_CHARGERS):
        # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
        if charger_occupancy[charger, hour] != 0:
            if actions[charger] >= 0:
                if hour in arrivals[charger]:
                    max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'], (1 - soc[charger, hour]) * self.EV_PARAMETERS['CAPACITY']])
                else:
                    max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'], (1 - soc[charger, hour-1]) * self.EV_PARAMETERS['CAPACITY']])
            else:
                max_charging_energy = min([self.EV_PARAMETERS['MAX DISCHARGING POWER'], soc[charger, hour] * self.EV_PARAMETERS['CAPACITY']])
                # max_charging_power = min([self.EV_PARAMETERS['MAX CHARGING POWER'],
                #                           soc[charger, timestep] * self.EV_PARAMETERS['CAPACITY'] / time_interval])
        else:
            max_charging_energy = 0

        charging_power[charger] = actions[charger]*max_charging_energy

    # ----------------------------------------------------------------------------
    # Calculation of next state of Battery based on actions
    for charger in range(self.NUMBER_OF_CHARGERS):
        if charger_occupancy[charger, hour] == 1 and hour in arrivals[charger]:
            soc[charger, hour] = soc[charger, hour] + charging_power[charger]/self.EV_PARAMETERS['CAPACITY']
        elif charger_occupancy[charger, hour] == 1 and hour not in arrivals[charger]:
            soc[charger, hour] = soc[charger, hour-1] + charging_power[charger]/self.EV_PARAMETERS['CAPACITY']
            # soc[charger, timestep] = soc[charger, timestep - 1] + (charging_power[charger] * time_interval) / \
            #                          self.EV_PARAMETERS['CAPACITY']

    # ----------------------------------------------------------------------------
    # Calculation of energy utilization from the PV
    # Calculation of energy coming from Grid
    available_renewable_energy = max([0, renewable[0, hour] - consumed[0, hour]])
    total_charging_power = sum(charging_power)

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
