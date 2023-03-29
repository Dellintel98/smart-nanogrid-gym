def load_initial_values(self):
    self.clear_initialisation_variables()

    initial_values = loadmat(data_files_directory_path + '\\initial_values.mat')
    with open(data_files_directory_path + "\\initial_values.json", "r") as fp:
        initials = json.load(fp)
        a = 2

    arrival_times = initial_values['Arrivals']
    departure_times = initial_values['Departures']

    self.vehicle_state_of_charge = initial_values['SOC']
    self.charger_occupancy = initial_values['Charger_occupancy']

    for charger in range(self.NUMBER_OF_CHARGERS):
        if arrival_times.shape == (1, self.NUMBER_OF_CHARGERS):
            arrivals = arrival_times[0][charger][0]
            departures = departure_times[0][charger][0]
        elif arrival_times.shape == (self.NUMBER_OF_CHARGERS, 3):
            arrivals = arrival_times[charger]
            departures = departure_times[charger]
        else:
            raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

        if isinstance(departures[0], ndarray):
            departures = array(departures.tolist()).flatten()
            arrivals = array(arrivals.tolist()).flatten()

        self.arrivals.append(arrivals.tolist())
        self.departures.append(departures.tolist())
        self.chargers[charger].vehicle_arrivals = self.arrivals[charger]
        self.chargers[charger].vehicle_state_of_charge = self.vehicle_state_of_charge[charger, :]
        self.chargers[charger].occupancy = initial_values['Charger_occupancy'][charger, :]