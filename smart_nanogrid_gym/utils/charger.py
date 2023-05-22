from typing import Optional
from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle
from numpy import array, zeros, floor, ceil, sign


class Charger:
    def __init__(self, charging_mode):
        self.vehicle_overcharging_value = 0.0
        self.vehicle_over_discharging_value = 0.0
        self.charging_non_existent_vehicle = 0.0
        self.CHARGING_MODE = charging_mode
        self.occupied: bool = False
        self.connected_electric_vehicle: Optional[ElectricVehicle] = None
        self.power_value: float = 0.0
        self.vehicle_arrivals: [] = []
        self.vehicle_state_of_charge: array = zeros(25)
        self.vehicle_capacities: array = zeros(25)
        self.occupancy: array = zeros(25)
        self.requested_end_state_of_charge: array = zeros(25)
        self.connected_electric_vehicle = ElectricVehicle(battery_capacity=40,
                                                          current_capacity=0, requested_end_capacity=1.0,
                                                          charging_efficiency=0.95, discharging_efficiency=0.95,
                                                          max_charging_power=22, max_discharging_power=22)

    # def connect_vehicle(self, hour):
    #     self.connected_electric_vehicle = ElectricVehicle(battery_capacity=40, requested_end_capacity=0.8,
    #                                                       current_capacity=self.vehicle_state_of_charge[hour],
    #                                                       charging_efficiency=0.95, discharging_efficiency=0.95,
    #                                                       max_charging_power=22, max_discharging_power=22)
    #     self.occupied = True

    # def disconnect_vehicle(self):
    #     # save data about departed vehicle or return it to be saved
    #     self.connected_electric_vehicle = None
    #     self.occupied = False

    def charge_or_discharge_vehicle(self, action, timestep, time_interval):
        if action == 0:
            self.power_value = 0.0
            self.vehicle_overcharging_value = 0.0
            self.vehicle_over_discharging_value = 0.0
            if timestep in self.vehicle_arrivals:
                self.vehicle_state_of_charge[timestep] = self.vehicle_state_of_charge[timestep]
            else:
                self.vehicle_state_of_charge[timestep] = self.vehicle_state_of_charge[timestep - 1]
        elif action > 0:
            self.power_value = self.charge_vehicle(action, timestep, time_interval)
            self.vehicle_over_discharging_value = 0.0
        else:
            self.power_value = self.discharge_vehicle(action, timestep, time_interval)
            self.vehicle_overcharging_value = 0.0

        # self.calculate_next_vehicle_state_of_charge(timestep, time_interval)
        self.charging_non_existent_vehicle = 0.0

        return self.power_value

    def charge_vehicle(self, action, timestep, time_interval):
        if self.CHARGING_MODE == 'bounded':
            charging_power = self.calculate_charging_power(action)

            if timestep in self.vehicle_arrivals:
                vehicle_capacity = self.vehicle_capacities[timestep]
                vehicle_state_of_charge = self.vehicle_state_of_charge[timestep]
            else:
                vehicle_capacity = self.vehicle_capacities[timestep - 1]
                vehicle_state_of_charge = self.vehicle_state_of_charge[timestep - 1]

            state_of_charge_value_change = (charging_power * time_interval) / vehicle_capacity

            calculated_state_of_charge = vehicle_state_of_charge + state_of_charge_value_change

            overcharging_flag = floor(0.5 * (1 + sign(calculated_state_of_charge - 1)))
            self.vehicle_overcharging_value = overcharging_flag * self.connected_electric_vehicle.max_charging_power
            # NON-CONSTANT PENALTIES TEND TO LEAD TO ALGORITHM STILL BEING IN UNWANTED AREA BUT LOWERING ACTIONS TO MIN
            # SO THAT THE PENALTY IS MINIMAL!!!
            # if calculated_state_of_charge > 1.0:
            #     # FULL EV BATTERY CAN OVERCHARGE BUT SOC STAYS THE SAME,
            #     # EXCESS ENERGY TRANSFORMS TO HEAT
            #     possible_charging_power = ((1.0 - vehicle_state_of_charge) * vehicle_capacity) / time_interval
            #     # self.excess_vehicle_charging_power = (charging_power - possible_charging_power)
            #     self.vehicle_overcharging_power = round(charging_power - possible_charging_power, 2) * 10
            # else:
            #     self.vehicle_overcharging_power = 0.0

            self.vehicle_state_of_charge[timestep] = min(calculated_state_of_charge, 1.0)
        else:
            raise ValueError("Error: Wrong charging mode provided!")

        return charging_power

    def calculate_charging_power(self, action):
        charging_power = action * self.connected_electric_vehicle.max_charging_power * self.connected_electric_vehicle.charging_efficiency
        return charging_power

    # def calculate_next_vehicle_state_of_charge(self, timestep, time_interval):
    #     if self.power_value != 0:
    #         state_of_charge_value_change = (self.power_value * time_interval) / self.vehicle_capacities[timestep]
    #     else:
    #         state_of_charge_value_change = 0.0
    #     # Todo: Add alpha limit for CC-CV charging switch here
    #
    #     if timestep in self.vehicle_arrivals:
    #         self.vehicle_state_of_charge[timestep] = self.vehicle_state_of_charge[timestep] + state_of_charge_value_change
    #     else:
    #         self.vehicle_state_of_charge[timestep] = self.vehicle_state_of_charge[timestep - 1] + state_of_charge_value_change

    def discharge_vehicle(self, action, timestep, time_interval):
        if self.CHARGING_MODE == 'bounded':
            discharging_power = self.calculate_discharging_power(action)

            if timestep in self.vehicle_arrivals:
                vehicle_capacity = self.vehicle_capacities[timestep]
                vehicle_state_of_charge = self.vehicle_state_of_charge[timestep]
            else:
                vehicle_capacity = self.vehicle_capacities[timestep - 1]
                vehicle_state_of_charge = self.vehicle_state_of_charge[timestep - 1]

            state_of_charge_value_change = (discharging_power * time_interval) / vehicle_capacity
            calculated_state_of_charge = vehicle_state_of_charge + state_of_charge_value_change

            over_discharging_flag = ceil(0.5 * (1 + sign(calculated_state_of_charge)))
            self.vehicle_over_discharging_value = over_discharging_flag * self.connected_electric_vehicle.max_discharging_power
            # NON-CONSTANT PENALTIES TEND TO LEAD TO ALGORITHM STILL BEING IN UNWANTED AREA BUT LOWERING ACTIONS TO MIN
            # SO THAT THE PENALTY IS MINIMAL!!!
            # if calculated_state_of_charge < 0:
            #     # EMPTY EV BATTERY CANNOT BE DISCHARGED
            if self.vehicle_over_discharging_value:
                possible_discharging_power = (vehicle_state_of_charge * vehicle_capacity) / time_interval
            #     # self.vehicle_over_discharging_power = (abs(discharging_power) - possible_discharging_power)
            #     # self.vehicle_over_discharging_power = round(abs(discharging_power) - possible_discharging_power, 2) * 10
                discharging_power = -possible_discharging_power
            # else:
            #     self.vehicle_over_discharging_power = 0.0

            self.vehicle_state_of_charge[timestep] = max(0.0, calculated_state_of_charge)
        else:
            raise ValueError("Error: Wrong discharging mode provided!")
        # Todo: Add returning calculated power value like in battery_system
        return discharging_power

    def calculate_discharging_power(self, action):
        discharging_power = action * self.connected_electric_vehicle.max_discharging_power * self.connected_electric_vehicle.discharging_efficiency
        return discharging_power

    def reset_info_values(self, action):
        self.power_value = 0.0
        self.vehicle_overcharging_value = 0.0
        self.vehicle_over_discharging_value = 0.0

        # NON-CONSTANT PENALTIES TEND TO LEAD TO ALGORITHM STILL BEING IN UNWANTED AREA BUT LOWERING ACTIONS TO MIN
        # SO THAT THE PENALTY IS MINIMAL!!!
        if action:
            self.charging_non_existent_vehicle = 100
        else:
            self.charging_non_existent_vehicle = 0.0
