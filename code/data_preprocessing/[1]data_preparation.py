import os
import pandas as pd
import numpy as np


base_dir = os.path.split(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])[0]
os.chdir(os.path.join(base_dir, 'data/elec_price'))
data_raw = pd.read_excel('elec_prices.xlsx')

data_raw['date'] = data_raw['Month'].astype(str) + '-' + data_raw['Year'].astype(str)
data_raw['date'] = pd.to_datetime(data_raw['date'], format='%m-%Y')
data_raw = data_raw.drop(columns=['Year', 'Month'])
data_raw = data_raw.set_index('date')

data_raw.index = pd.to_datetime(data_raw.index)
data_raw.index.name = None



list_states = data_raw['State'].unique()
state_dict_res = {}
for state in list_states:
    state_dict_res[state] = np.array(data_raw.loc[data_raw['State'] == state, 'residential price']) # we keep only residental price

# Create a dataframe with the residential prices as this the data we are interested in
data = pd.DataFrame(state_dict_res , index=data_raw.index.unique()).sort_index()
data = data.loc["1992-01-01":"1998-12-01",:]
#Check for missing values
print(data.isnull().sum().sum())

data=data.transpose()

print (data.head())
