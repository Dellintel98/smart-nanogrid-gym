import numpy as np
import scipy.io


def get_price_day(current_price_model):
    # high_tariff = 0.028 + 0.148933
    # low_tariff = 0.013333 + 0.087613
    price_day = []
    # if current_price_model == 0:
    #     price_day = np.array([low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff,
    #                           high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
    #                           high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
    #                           low_tariff, low_tariff, low_tariff, low_tariff])
    if current_price_model == 1:
        price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    elif current_price_model == 2:
        price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06,
                              0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
    elif current_price_model == 3:
        price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                              0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
    elif current_price_model == 4:
        price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                              0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
    elif current_price_model == 5:
        price_day[1, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
        price_day[2, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05,
                           0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05]
        price_day[3, :] = [0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                           0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070]
        price_day[4, :] = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                           0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]

    price_day = np.concatenate([price_day, price_day], axis=0)
    return price_day


def get_energy_price(price_day, experiment_length_in_days):
    price = np.zeros((experiment_length_in_days, 2 * 24))
    for day in range(0, experiment_length_in_days):
        price[day, :] = price_day
    return price


def calculate_solar_irradiance_mean_per_timestep(timestep_in_minutes,
                                                 experiment_time_period_plus_day_ahead,
                                                 solar_irradiance_forecast):
    solar_irradiance = np.zeros([experiment_time_period_plus_day_ahead, 1])
    count = 0
    for time_interval in range(0, timestep_in_minutes * experiment_time_period_plus_day_ahead, timestep_in_minutes):
        next_time_interval = time_interval + timestep_in_minutes - 1
        solar_irradiance[count, 0] = (np.mean(solar_irradiance_forecast[time_interval: next_time_interval, 2]))
        count = count + 1
    return solar_irradiance


def calculate_pv_scaling_coefficient(pv_system_total_dimensions, pv_system_efficiency):
    return pv_system_total_dimensions * pv_system_efficiency / 1000


def calculate_available_renewable_energy(solar_irradiance, scaling_pv, pv_system_availability):
    scaling_sol = 1.5
    return solar_irradiance * scaling_pv * scaling_sol * pv_system_availability


def calculate_available_solar_radiation_and_energy(
        experiment_length_in_days, single_experiment_day_length_in_minutes, pv_system_total_dimensions,
        pv_system_efficiency, solar_irradiance, pv_system_availability
):
    renewable_energy = np.zeros([experiment_length_in_days, single_experiment_day_length_in_minutes * 2])
    solar_radiation = np.zeros([experiment_length_in_days, single_experiment_day_length_in_minutes * 2])

    count = 0
    for day in range(0, int(experiment_length_in_days)):
        for time_interval in range(0, single_experiment_day_length_in_minutes * 2):
            scaling_pv = calculate_pv_scaling_coefficient(pv_system_total_dimensions, pv_system_efficiency)

            renewable_energy[day, time_interval] = calculate_available_renewable_energy(solar_irradiance[count, 0],
                                                                                        scaling_pv,
                                                                                        pv_system_availability)
            solar_radiation[day, time_interval] = solar_irradiance[count, 0]
            count = count + 1

    reshaped_solar_irradiance = np.reshape(
        solar_irradiance,
        (int(experiment_length_in_days), single_experiment_day_length_in_minutes * 2)
    )

    return renewable_energy, solar_radiation


def get_energy(experiment_length_in_days, current_price_model, pv_system_availability, file_directory_path,
               pv_system_total_dimensions, pv_system_efficiency):
    # pv_system_availability = int(self.PV_SYSTEM_AVAILABLE_IN_MODEL)
    # pv_system_availability = 1 if self.PV_SYSTEM_AVAILABLE_IN_MODEL else 0

    atmospheric_conditions = scipy.io.loadmat(file_directory_path + 'atmospheric_conditions.mat')
    atmospheric_conditions_forecast = atmospheric_conditions['mydata']

    experiment_time_period_plus_day_ahead = 24 * (experiment_length_in_days + 1)
    timestep_in_minutes = 60
    single_experiment_day_length_in_minutes = int(60 / timestep_in_minutes) * 24
    experiment_length_in_minutes = experiment_length_in_days * single_experiment_day_length_in_minutes

    solar_irradiance = calculate_solar_irradiance_mean_per_timestep(timestep_in_minutes,
                                                                    experiment_time_period_plus_day_ahead,
                                                                    atmospheric_conditions_forecast)
    available_renewable_energy, solar_radiation = calculate_available_solar_radiation_and_energy(
        experiment_length_in_days,
        single_experiment_day_length_in_minutes,
        pv_system_total_dimensions,
        pv_system_efficiency,
        solar_irradiance,
        pv_system_availability
    )

    price_day = get_price_day(current_price_model)
    energy_price = get_energy_price(price_day, experiment_length_in_days)
    consumed = np.zeros(np.shape(available_renewable_energy))

    return {
        'Consumed': consumed,
        'Available renewable': available_renewable_energy,
        'Price': energy_price,
        'Solar radiation': solar_radiation
    }
