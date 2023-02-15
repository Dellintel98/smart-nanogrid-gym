from dataclasses import dataclass
from numpy import ndarray


@dataclass
class BatteryEnergyStorageSystem:
    max_battery_capacity: int
    current_battery_capacity: float
    charging_efficiency: float
    discharging_efficiency: float
    max_charging_power: int
    max_discharging_power: int
    depth_of_discharge: float
