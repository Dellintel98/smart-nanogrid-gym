import numpy as np
import time


def simulate_central_management_system(self, actions):
    hour = self.timestep
    consumed = self.energy['Consumed']
    renewable = self.energy['Available renewable']
    charger_occupancy = self.initial_simulation_values['Charger occupancy']

    departing_vehicles = self.departing_vehicles
    soc = self.ev_state_of_charge

    charging_power = np.zeros(self.NUMBER_OF_CHARGERS)

    # ----------------------------------------------------------------------------
    # Calculation of demand based on actions
    # Calculation of actions for cars
    for charger in range(self.NUMBER_OF_CHARGERS):
        if actions[charger] >= 0:
            max_charging_energy = min([10, (1-soc[charger, hour]) * self.EV_PARAMETERS['CAPACITY']])
        else:
            max_charging_energy = min([10, soc[charger, hour] * self.EV_PARAMETERS['CAPACITY']])

        charging_power[charger] = 100*actions[charger]/100*max_charging_energy

    # ----------------------------------------------------------------------------
    # Calculation of next state of Battery based on actions
    for charger in range(self.NUMBER_OF_CHARGERS):
        if charger_occupancy[charger, hour] == 1:
            soc[charger, hour+1] = soc[charger, hour] + charging_power[charger]/self.EV_PARAMETERS['CAPACITY']

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
