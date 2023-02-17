from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle


class Charger:
    max_battery_capacity: int
    vehicle_connected: bool
    occupied: bool
    connected_electric_vehicle: ElectricVehicle

