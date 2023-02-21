from typing import Optional
from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle
from numpy import array, zeros


class Charger:
    def __init__(self):
        self.occupied: bool = False
        self.connected_electric_vehicle: Optional[ElectricVehicle] = None
        self.power_value: float = 0.0
        self.vehicle_arrivals: [] = []
        self.vehicle_state_of_charge: array = zeros(25)
        self.occupancy: array = zeros(25)
        self.connected_electric_vehicle = ElectricVehicle(battery_capacity=40,
                                                          current_capacity=0,
                                                          charging_efficiency=0.95, discharging_efficiency=0.95,
                                                          max_charging_power=22, max_discharging_power=22)

    def connect_vehicle(self, hour):
        self.connected_electric_vehicle = ElectricVehicle(battery_capacity=40,
                                                          current_capacity=self.vehicle_state_of_charge[hour],
                                                          charging_efficiency=0.95, discharging_efficiency=0.95,
                                                          max_charging_power=22, max_discharging_power=22)
        self.occupied = True

    def disconnect_vehicle(self):
        # save data about departed vehicle or return it to be saved
        self.connected_electric_vehicle = None
        self.occupied = False

    def charge_or_discharge_vehicle(self, action, hour):
        if action >= 0:
            self.power_value = self.charge_vehicle(action, hour)
        else:
            self.power_value = self.discharge_vehicle(action, hour)

        return self.power_value

    def charge_vehicle(self, action, hour):
        max_charging_power = self.calculate_max_charging_power(hour)
        charging_power = self.calculate_charging_or_discharging_power(max_charging_power, action)
        self.calculate_next_vehicle_state_of_charge(charging_power, hour)
        return charging_power

    def calculate_max_charging_power(self, hour):
        # max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'],
        #                           soc[timestep] * self.EV_PARAMETERS['CAPACITY'] / time_interval])
        if hour in self.vehicle_arrivals:
            remaining_uncharged_capacity = 1 - self.vehicle_state_of_charge[hour]
            power_left_to_charge = remaining_uncharged_capacity * self.connected_electric_vehicle.battery_capacity
        else:
            remaining_uncharged_capacity = 1 - self.vehicle_state_of_charge[hour - 1]
            power_left_to_charge = remaining_uncharged_capacity * self.connected_electric_vehicle.battery_capacity

        max_charging_energy = min([self.connected_electric_vehicle.max_charging_power, power_left_to_charge])
        return max_charging_energy

    def calculate_charging_or_discharging_power(self, max_power, action):
        return action * max_power

    def calculate_next_vehicle_state_of_charge(self, power_value, hour):
        state_of_charge_value_change = power_value / self.connected_electric_vehicle.battery_capacity
        if hour in self.vehicle_arrivals:
            self.vehicle_state_of_charge[hour] = self.vehicle_state_of_charge[hour] + state_of_charge_value_change
        else:
            self.vehicle_state_of_charge[hour] = self.vehicle_state_of_charge[hour - 1] + state_of_charge_value_change
            # soc[timestep] = soc[timestep - 1] + (charging_power * time_interval) / \
            #                          self.EV_PARAMETERS['CAPACITY']

    def discharge_vehicle(self, action, hour):
        max_discharging_power = self.calculate_max_discharging_power(hour)
        discharging_power = self.calculate_charging_or_discharging_power(max_discharging_power, action)
        self.calculate_next_vehicle_state_of_charge(discharging_power, hour)
        return discharging_power

    def calculate_max_discharging_power(self, hour):
        if hour in self.vehicle_arrivals:
            power_left_to_discharge = self.vehicle_state_of_charge[hour]\
                                      * self.connected_electric_vehicle.battery_capacity
        else:
            power_left_to_discharge = self.vehicle_state_of_charge[hour - 1]\
                                      * self.connected_electric_vehicle.battery_capacity

        max_discharging_energy = min([self.connected_electric_vehicle.max_discharging_power, power_left_to_discharge])
        return max_discharging_energy
