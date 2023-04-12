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
        self.excess_charging_amount: float = 0.0
        self.excess_discharging_amount: float = 0.0

    def charge_or_discharge(self, action, power_demand, time_interval):
        if action == 0:
            self.current_power_value = 0.0
            self.excess_charging_amount = 0.0
            self.excess_discharging_amount = 0.0
        elif action > 0:
            available_power_for_charging = -power_demand
            power_demand = self.charge(available_power_for_charging, action, time_interval)
            self.excess_discharging_amount = 0.0
        else:
            power_demand = self.discharge(power_demand, action, time_interval)
            self.excess_charging_amount = 0.0

        return power_demand

    def charge(self, available_power, positive_action, time_interval):
        if self.CHARGING_MODE == 'bounded':
            charging_power = positive_action * self.max_charging_power * self.charging_efficiency
            current_state_of_charge = self.current_state_of_charge + (charging_power * time_interval) / self.max_capacity
            self.current_state_of_charge = min(current_state_of_charge, 1.0)
            if current_state_of_charge > 1:
                self.excess_charging_amount = current_state_of_charge - 1.0
            else:
                self.excess_charging_amount = 0.0
            self.current_power_value = charging_power

            remaining_available_power = available_power - charging_power

            return -remaining_available_power
        else:
            raise ValueError("Error: Wrong battery charging mode provided!")

    def discharge(self, power_demand, negative_action, time_interval):
        if self.CHARGING_MODE == 'bounded':
            discharging_power = negative_action * self.max_discharging_power * self.discharging_efficiency
            current_state_of_charge = self.current_state_of_charge + (discharging_power * time_interval) / self.max_capacity
            self.current_state_of_charge = max(0.0, current_state_of_charge)
            if current_state_of_charge < 0:
                self.excess_discharging_amount = -current_state_of_charge
            else:
                self.excess_discharging_amount = 0.0
            # Todo: Add penalty for excessive discharging, i.e. for over-discharging
            # Todo: Add return remaining_demand and also difference between 0.0 and soc if soc below 0
            self.current_power_value = discharging_power
            # Todo: Add adjusting remaining demand if current_soc < 0 -> MAYBE
            remaining_demand = power_demand + discharging_power

            return remaining_demand
        else:
            raise ValueError("Error: Wrong battery discharging mode provided!")

    def get_state_of_charge(self):
        return self.current_state_of_charge

    def get_used_power_value(self):
        return self.current_power_value
