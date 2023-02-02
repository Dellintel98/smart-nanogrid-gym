import numpy as np
import scipy.io


def get_energy(self):
    days_of_experiment = self.NUMBER_OF_DAYS_TO_PREDICT
    current_price_model = self.CURRENT_PRICE_MODEL
    # pv_system_availability = int(self.PV_SYSTEM_AVAILABLE_IN_MODEL)
    pv_system_availability = 1 if self.PV_SYSTEM_AVAILABLE_IN_MODEL else 0

    atmospheric_conditions = scipy.io.loadmat(self.file_directory_path + 'atmospheric_conditions.mat')
    atmospheric_conditions_forecast = atmospheric_conditions['mydata']

    experiment_time_period_plus_day_ahead = 24 * (days_of_experiment + 1)
    temperature = np.zeros([experiment_time_period_plus_day_ahead, 1])
    humidity = np.zeros([experiment_time_period_plus_day_ahead, 1])
    solar_irradiance = np.zeros([experiment_time_period_plus_day_ahead, 1])
    timestep_in_minutes = 60

    count = 0
    for time_interval in range(0, timestep_in_minutes * experiment_time_period_plus_day_ahead, timestep_in_minutes):
        next_time_interval = time_interval + timestep_in_minutes - 1

        temperature[count, 0] = (np.mean(atmospheric_conditions_forecast[time_interval: next_time_interval, 0]))
        humidity[count, 0] = (np.mean(atmospheric_conditions_forecast[time_interval: next_time_interval, 1]))
        solar_irradiance[count, 0] = (np.mean(atmospheric_conditions_forecast[time_interval: next_time_interval, 2]))

        count = count + 1

    single_experiment_day_length_in_minutes = int(60 / timestep_in_minutes) * 24
    experiment_length_in_minutes = days_of_experiment * single_experiment_day_length_in_minutes

    available_renewable_energy = np.zeros([days_of_experiment, single_experiment_day_length_in_minutes * 2])
    solar_radiation = np.zeros([days_of_experiment, single_experiment_day_length_in_minutes * 2])

    count = 0
    for day in range(0, int(days_of_experiment)):
        for time_interval in range(0, single_experiment_day_length_in_minutes * 2):
            scaling_pv = self.PV_SYSTEM_PARAMETERS['TOTAL DIMENSIONS'] * self.PV_SYSTEM_PARAMETERS['EFFICIENCY'] / 1000
            scaling_sol = 1.5

            temp = solar_irradiance[count, 0] * scaling_sol * scaling_pv * pv_system_availability
            available_renewable_energy[day, time_interval] = temp
            solar_radiation[day, time_interval] = solar_irradiance[count, 0]
            count = count + 1

    # --------------------------------------
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
    price = np.zeros((days_of_experiment, 48))
    for day in range(0, days_of_experiment):
        price[day, :] = price_day

    # for ii in range(1,days_of_experiment):
    #   Mixing_functions[ii] = sum(Solar[(ii - 1) * 24 + 1:(ii - 1) * 24 + 24]) / 16

    consumed = np.zeros(np.shape(available_renewable_energy))

    return {
        'Consumed': consumed,
        'Available renewable': available_renewable_energy,
        'Price': price,
        'Solar radiation': solar_radiation
    }
