import numpy as np
import scipy.io

def Energy_Calculation(self):
    days_of_experiment = self.NUMBER_OF_DAYS_TO_PREDICT
    current_folder = self.file_directory_path
    price_flag = self.CURRENT_PRICE_MODEL
    solar_flag = self.PV_SYSTEM_AVAILABLE_IN_MODEL

    contents=scipy.io.loadmat(current_folder+'weather.mat')
    x_forecast = contents['mydata']


    temperature=np.zeros([24*(days_of_experiment+1),1])
    humidity=np.zeros([24*(days_of_experiment+1),1])
    solar_radiation=np.zeros([24*(days_of_experiment+1),1])
    minutes_of_timestep=60
    count=0
    for ii in range(0,minutes_of_timestep*24*(days_of_experiment+1),minutes_of_timestep):
        temperature[count,0]=(np.mean(x_forecast[ii: ii + 59,0]))
        humidity[count,0]=(np.mean(x_forecast[ii: ii + 59,1]))
        solar_radiation[count,0]=(np.mean(x_forecast[ii: ii + 59,2]))
        count=count+1

    experiment_length = days_of_experiment * (60/minutes_of_timestep)*24
    Renewable=np.zeros([days_of_experiment,int(60/minutes_of_timestep)*48])
    Radiation = np.zeros([days_of_experiment, int(60 / minutes_of_timestep) * 48])
    count=0
    for ii in range(0,int(days_of_experiment)):
        for jj in range(0,int((60/minutes_of_timestep)*48)):
            scaling_PV = self.PV_SYSTEM_PARAMETERS['PV_Surface'] * self.PV_SYSTEM_PARAMETERS['PV_effic'] / 1000
            scaling_sol = 1.5
            xx=solar_radiation[count,0] * scaling_sol * scaling_PV * solar_flag
            Radiation[ii, jj] = solar_radiation[count,0]
            Renewable[ii,jj]=xx
            count=count+1



    Price_day=[]
    #--------------------------------------
    if price_flag==1:
        Price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                     0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    elif price_flag==2:
        Price_day=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08 ,0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06 ,0.06 ,0.06, 0.05, 0.05, 0.05])
    elif price_flag==3:
        Price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080, 0.080, 0.1, 0.1, 0.076, 0.076,
                     0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
    elif price_flag==4:

       Price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1,
                    0.1, 0.1])
    elif price_flag==5:
        Price_day[1, :]=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05,
                        0.05, 0.05]
        Price_day[2, :]= [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,0.1, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06,
                         0.06, 0.06, 0.05, 0.05, 0.05]
        Price_day[3, :] = [0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080, 0.080, 0.1, 0.1, 0.076,
                          0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070]
        Price_day[4, :] = [0.1, 0.1, 0.05, 0.05, 0.05 ,0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06 ,0.06, 0.1,
                          0.1, 0.1, 0.1]

    Price_day = np.concatenate([Price_day,Price_day],axis=0)
    Price = np.zeros((days_of_experiment, 48))
    for ii in range(0, days_of_experiment):
        Price[ii, :] = Price_day


    #for ii in range(1,days_of_experiment):
     #   Mixing_functions[ii] = sum(Solar[(ii - 1) * 24 + 1:(ii - 1) * 24 + 24]) / 16

    Consumed=np.zeros(np.shape(Renewable))
    return Consumed,Renewable,Price,Radiation
