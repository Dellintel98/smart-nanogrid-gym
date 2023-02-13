from numpy import mean, reshape, zeros, shape
from scipy.io import loadmat


class PVSystemManager:
    def __init__(self):
        pass

    def calculate_solar_irradiance_mean_per_timestep(self, timestep_in_minutes, experiment_length_in_hours,
                                                     file_directory_path):
        solar_irradiance_forecast = self.load_irradiance_data(file_directory_path, 'solar_irradiance.mat')

        solar_irradiance = zeros([experiment_length_in_hours, 1])
        experiment_length_in_minutes = timestep_in_minutes * experiment_length_in_hours

        count = 0
        for time_interval in range(0, experiment_length_in_minutes, timestep_in_minutes):
            next_time_interval = time_interval + timestep_in_minutes
            solar_irradiance[count, 0] = (mean(solar_irradiance_forecast[time_interval: next_time_interval]))
            count = count + 1
        return solar_irradiance

    def calculate_pv_scaling_coefficient(self, pv_system_total_dimensions, pv_system_efficiency):
        return pv_system_total_dimensions * pv_system_efficiency / 1000

    def calculate_available_solar_energy(self, solar_irradiance, pv_system_total_dimensions, pv_system_efficiency):
        scaling_pv = self.calculate_pv_scaling_coefficient(pv_system_total_dimensions, pv_system_efficiency)
        scaling_sol = 1.5
        return solar_irradiance * scaling_pv * scaling_sol

    def calculate_available_solar_radiation(self, solar_irradiance, experiment_length_in_days, timestep_in_minutes):
        experiment_day_length_in_timesteps = int(60 / timestep_in_minutes) * 24

        reshaped_solar_irradiance = reshape(
            solar_irradiance,
            (experiment_length_in_days, experiment_day_length_in_timesteps * 2)
        )

        return reshaped_solar_irradiance

    def load_irradiance_data(self, file_directory_path, irradiance_data_filename):
        irradiance_data = loadmat(file_directory_path + irradiance_data_filename)
        return irradiance_data['irradiance']

    def get_energy(self, experiment_length_in_days, pv_system_available, file_directory_path,
                   pv_system_total_dimensions, pv_system_efficiency, number_of_days_ahead_for_prediction):
        timestep_in_minutes = 60
        experiment_length_in_hours = 24 * (experiment_length_in_days + number_of_days_ahead_for_prediction)

        solar_irradiance = self.calculate_solar_irradiance_mean_per_timestep(timestep_in_minutes,
                                                                             experiment_length_in_hours,
                                                                             file_directory_path)

        solar_radiation = self.calculate_available_solar_radiation(solar_irradiance, experiment_length_in_days,
                                                                   timestep_in_minutes)

        if pv_system_available:
            available_solar_energy = self.calculate_available_solar_energy(solar_irradiance, pv_system_total_dimensions,
                                                                           pv_system_efficiency)
        else:
            available_solar_energy = zeros(shape(solar_irradiance))

        consumed = zeros(shape(available_solar_energy))

        return {
            'Consumed': consumed,
            'Available renewable': available_solar_energy,
            'Solar radiation': solar_radiation
        }
