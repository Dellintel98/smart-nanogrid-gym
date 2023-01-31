import numpy as np


def simulate_ev_charging_station(self):
    vehicle_soc = self.ev_state_of_charge
    arrival = self.initial_simulation_values['Arrivals']
    departure = self.initial_simulation_values['Departures']
    charger_occupancy = self.initial_simulation_values['Charger occupancy']
    number_of_chargers = self.NUMBER_OF_CHARGERS
    day = self.day
    hour = self.timestep

    # calculation of which cars depart now
    departing_vehicles = []
    if hour < 24:
        for charger in range(number_of_chargers):
            departure_vehicle = departure[charger]
            if charger_occupancy[charger, hour] == 1 and (hour+1 in departure_vehicle):
                departing_vehicles.append(charger)

    # calculation of the hour each car is leaving
    departure_times = []
    for charger in range(number_of_chargers):
        if charger_occupancy[charger, hour] == 0:
            departure_times.append(0)
        else:
            for ii in range(len(departure[charger])):
                if hour < departure[charger][ii]:
                    departure_times.append(departure[charger][ii]-hour)
                    break

    # calculation of the BOC of each car
    vehicles_state_of_charge = []
    for charger in range(number_of_chargers):
        vehicles_state_of_charge.append(vehicle_soc[charger, hour])

    return departing_vehicles, departure_times, vehicles_state_of_charge
