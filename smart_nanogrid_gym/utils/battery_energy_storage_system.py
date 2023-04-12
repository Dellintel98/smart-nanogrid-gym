# from dataclasses import dataclass
from numpy import ndarray


class BatteryEnergyStorageSystem:
    def __init__(self, charging_mode, max_capacity: int, current_state_of_charge: float,
                 max_charging_power: int, max_discharging_power: int,
                 charging_efficiency: float = 0.95, discharging_efficiency: float = 0.95,
                 depth_of_discharge: float = 0.15):
        self.CHARGING_MODE = charging_mode
        self.max_capacity: int = max_capacity
        self.current_state_of_charge: float = current_state_of_charge
        self.charging_efficiency: float = charging_efficiency
        self.discharging_efficiency: float = discharging_efficiency
        self.max_charging_power: int = max_charging_power
        self.max_discharging_power: int = max_discharging_power
        self.depth_of_discharge: float = depth_of_discharge
        self.current_power_value: float = 0.0
        self.excess_charging_power: float = 0.0
        self.excess_discharging_power: float = 0.0

    def charge_or_discharge(self, action, power_demand, time_interval):
        if action == 0:
            self.current_power_value = 0.0
            self.excess_charging_power = 0.0
            self.excess_discharging_power = 0.0
        elif action > 0:
            available_power_for_charging = -power_demand
            power_demand = self.charge(available_power_for_charging, action, time_interval)
            self.excess_discharging_power = 0.0
        else:
            power_demand = self.discharge(power_demand, action, time_interval)
            self.excess_charging_power = 0.0

        return power_demand

    def charge(self, available_power, positive_action, time_interval):
        if self.CHARGING_MODE == 'bounded':
            charging_power = positive_action * self.max_charging_power * self.charging_efficiency
            current_state_of_charge = self.current_state_of_charge + (charging_power * time_interval) / self.max_capacity

            if current_state_of_charge > 1:
                # FULL BATTERY CAN OVERCHARGE BUT SOC STAYS THE SAME,
                # EXCESS ENERGY TRANSFORMS TO HEAT
                possible_charging_power = ((1.0 - self.current_state_of_charge) * self.max_capacity) / time_interval
                self.excess_charging_power = charging_power - possible_charging_power
            else:
                self.excess_charging_power = 0.0

            self.current_state_of_charge = min(current_state_of_charge, 1.0)
            self.current_power_value = charging_power

            remaining_available_power = available_power - charging_power

            return -remaining_available_power
        else:
            raise ValueError("Error: Wrong battery charging mode provided!")

    def discharge(self, power_demand, negative_action, time_interval):
        if self.CHARGING_MODE == 'bounded':
            discharging_power = negative_action * self.max_discharging_power * self.discharging_efficiency
            current_state_of_charge = self.current_state_of_charge + (discharging_power * time_interval) / self.max_capacity

            if current_state_of_charge < 0:
                # EMPTY BATTERY CANNOT BE DISCHARGED, THEREFORE REMAINING POWER CANNOT BE CHANGED
                # FOR THE VALUE LARGER THAN THE AMOUNT BATTERY HAS
                possible_discharging_power = (self.current_state_of_charge * self.max_capacity) / time_interval
                self.excess_discharging_power = abs(discharging_power) - possible_discharging_power
                discharging_power = -possible_discharging_power
            else:
                self.excess_discharging_power = 0.0

            self.current_state_of_charge = max(0.0, current_state_of_charge)
            self.current_power_value = discharging_power
            # Todo: Add calculated_power_value to show wrong initial discharging power value for wrong action

            remaining_demand = power_demand + discharging_power

            return remaining_demand
        else:
            raise ValueError("Error: Wrong battery discharging mode provided!")

    def get_state_of_charge(self):
        return self.current_state_of_charge

    def get_used_power_value(self):
        return self.current_power_value

    def get_system_info(self):
        info = {
            'current_state_of_charge': self.current_state_of_charge,
            'depth_of_discharge': self.depth_of_discharge,
            'excess_charging_power': self.excess_charging_power,
            'excess_discharging_power': self.excess_discharging_power
        }
        return info
