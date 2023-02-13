from dataclasses import dataclass
from numpy import ndarray


@dataclass
class ElectricVehicle:
    battery_capacity: int
    charging_efficiency: float
    discharging_efficiency: float
    max_charging_power: int
    max_discharging_power: int
