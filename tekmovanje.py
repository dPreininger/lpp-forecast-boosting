from calendar import weekday
from datetime import datetime, time
import numpy as np
import pandas as pd
import xgboost as xgb
import lpputils

route_dict = {}
direction_dict = {}

def read_data(filename, sep='\t'):
    """
    Reads the data from the given file.
    """
    return pd.read_csv(filename, sep=sep)

def add_day_of_week(data):
    """
    Adds the day of the week to the data represented as int.
    """
    rows = pd.to_datetime(data['Departure time'])
    days = np.zeros((len(rows), 7))
    for index, row in enumerate(rows):
        days[index, row.weekday()] = 1
    days = pd.DataFrame(days, columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    data = pd.concat([data, days], axis=1)
    return data

def add_holiday_info(data):
    """
    Adds information if a day is holiday to the data.
    """
    holidays = read_data('prazniki.csv', ';')
    dates = list(holidays['DATUM'])
    data['Holiday'] = data['Departure time'].apply(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').strftime('%-d.%m.%Y') in dates else 0)
    return data

def add_duration(data):
    """
    Adds the duration of the trip to the data.
    """
    data['Duration'] = data.apply(lambda x: lpputils.tsdiff(x['Arrival time'], x['Departure time']), axis=1)
    return data

def add_structured_time(data):
    """
    Adds structured departure time to the data.
    """
    data['DP hour'] = pd.to_datetime(data['Departure time']).dt.hour
    data['DP min'] = pd.to_datetime(data['Departure time']).dt.minute
    data['DP day'] = pd.to_datetime(data['Departure time']).dt.day
    data['DP month'] = pd.to_datetime(data['Departure time']).dt.month
    return data

def build_route_key(row):
    return (str(row['Route']) + row['Route Direction'] + row['Route description']).replace(' ', '')

def build_route_key_man(route, direction, desc):
    return (str(route) + direction + desc).replace(' ', '')

def get_all_routes(data):
    """
    Returns all unique routes.
    """
    return data['Route'].unique()

def get_all_directions(data, route):
    """
    Returns all unique directions for a given route.
    """
    return data[(data['Route'] == route)]['Route Direction'].unique()

def get_all_discriptions(data, route, direction):
    """
    Returns all unique descriptions for a given route and direction.
    """
    return data[(data['Route'] == route) & (data['Route Direction'] == direction)]['Route description'].unique()

def split_by_route(data):
    """
    Creates seperate dataset for each route.
    """
    datasets = {}
    for route in get_all_routes(data):
        for direction in get_all_directions(data, route):
            for description in get_all_discriptions(data, route, direction):
                datasets[build_route_key_man(route, direction, description)] = data.copy()[(data['Route'] == route) & (data['Route Direction'] == direction) & (data['Route description'] == description)]

    return datasets


def pre_process_data(data, train=True):
    """
    Pre-processes the data.
    """

    data = add_day_of_week(data)
    data = add_holiday_info(data)
    data = add_structured_time(data)

    # not really needed since they are the same everywhere
    data = data.drop('First station' , axis=1)
    data = data.drop('Last station', axis=1)

    data = data.drop('Registration', axis=1)
    data = data.drop('Driver ID', axis=1)

    if train:
        add_duration(data)
    data = data.drop('Arrival time', axis=1)

    departures = data['Departure time']
    data = data.drop('Departure time', axis=1)

    datasets = None
    if train:
        datasets = split_by_route(data)
        for set in datasets.values():
            set.drop(['Route', 'Route Direction', 'Route description'], axis=1, inplace=True)

    return data, departures, datasets

def train_model(data, label='Duration'):
    """
    Trains the model.
    """
    models = {}
    for route, dataset in data.items():
        X = dataset.drop(label, axis=1).to_numpy()
        y = dataset[label].to_numpy()

        model = xgb.XGBRegressor(eval_metric='mae', verbosity=0, n_threads=4, max_depth=7, learning_rate=0.25)
        model.fit(X, y)
        models[route] = model
    return models

def predict(models, data: pd.DataFrame):
    """
    Predicts the arrival time for the given data. Data should be pre-processed.
    """
    results = []
    for _, row in data.iterrows():
        try:
            model = models[build_route_key(row)]
        except KeyError:
            model = models[build_route_key_man(row['Route'], row['Route Direction'], row['Route description'].split(';')[0])]
        pred_data = [row.drop(['Route', 'Route Direction', 'Route description']).to_numpy()]
        results.append(model.predict(pred_data))

    data['Duration'] = results
    return data

def post_process(data, departures):
    """
    Post-processes the data.
    """
    data['Departure time'] = departures
    data['Arrival time'] = data.apply(lambda x: lpputils.tsadd(x['Departure time'], int(x['Duration'][0])), axis=1)
    return data

def create_output(data, departures, filename='out.txt'):
    """
    Creates the output file.
    """
    data = post_process(data, departures)
    data['Arrival time'].to_csv(filename, sep='\n', index=False, header=False)

if __name__ == '__main__':
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')
    train_data, departures_train, train_datasets = pre_process_data(train_data)
    test_data, departures_test, _ = pre_process_data(test_data, train=False)
    models = train_model(train_datasets)
    pred = predict(models, test_data)
    create_output(pred, departures_test)
    print('done')
    