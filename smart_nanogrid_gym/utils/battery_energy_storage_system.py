# from dataclasses import dataclass
from numpy import ndarray


class BatteryEnergyStorageSystem:
    def __init__(self, max_capacity: int, current_capacity: float, max_charging_power: int, max_discharging_power: int,
                 charging_efficiency: float = 0.95, discharging_efficiency: float = 0.95,
                 depth_of_discharge: float = 0.15):
        self.max_capacity: int = max_capacity
        self.current_capacity: float = current_capacity
        self.charging_efficiency: float = charging_efficiency
        self.discharging_efficiency: float = discharging_efficiency
        self.max_charging_power: int = max_charging_power
        self.max_discharging_power: int = max_discharging_power
        self.depth_of_discharge: float = depth_of_discharge

    def charge(self, available_energy):
        capacity_available_to_charge = 1 - self.current_capacity

        if capacity_available_to_charge > 0:
            power_available_for_charge = capacity_available_to_charge * self.max_capacity
            max_charging_energy = min([self.max_charging_power, power_available_for_charge])

            remaining_available_energy = available_energy - max_charging_energy

            if remaining_available_energy < 0:
                max_charging_energy = available_energy
                remaining_available_energy = 0

            self.current_capacity = self.current_capacity + max_charging_energy / self.max_capacity

            return remaining_available_energy
        else:
            return available_energy

    def discharge(self, energy_demand):
        capacity_available_to_discharge = self.current_capacity - self.depth_of_discharge

        if capacity_available_to_discharge > 0:
            power_available_for_discharge = capacity_available_to_discharge * self.max_capacity
            max_discharging_energy = min([self.max_discharging_power, power_available_for_discharge])

            remaining_demand = energy_demand - max_discharging_energy

            if remaining_demand < 0:
                max_discharging_energy = energy_demand
                remaining_demand = 0

            self.current_capacity = self.current_capacity - max_discharging_energy / self.max_capacity

            return remaining_demand
        else:
            return energy_demand
