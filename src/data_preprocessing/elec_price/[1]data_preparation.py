import os
import pandas as pd
import numpy as np


base_dir = os.path.join(os.path.dirname(__file__),'..', '..', '..')
data_path = os.path.join(base_dir, 'data/elec_price')
data_raw = pd.read_excel(os.path.join(data_path, 'elec_prices.xlsx'))

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
print(data.isnull().sum().sum()) # 0 missing values

data=data.transpose()
data.insert(0, 'state', data.index)
data = data.reset_index(drop=True)

# Writing the full dataset in a txt file
data.to_csv(os.path.join(data_path, 'elec_price_full.txt'), index= False, header=False)

# Train dataset - 01.1992 - 12.1997
data_train = data.iloc[:,0:73]
data_train.to_csv(os.path.join(data_path,'elec_price_train.txt'), index= False, header=False)

# Test dataset - 01.1998 - 12.1998
data_test = data.iloc[:,73:85]
data_test.to_csv(os.path.join(data_path,'elec_price_test.txt'), index= False, header=False)


# Adjust training set to right format for forecasting task 
data_train_adj = data_train.iloc[:,1:]
data_train_adj.to_csv(os.path.join(data_path,'priceMT_dataset.txt'), index= False, header=False)


# Adjust test set to right format for forecasting task
data_test.to_csv(os.path.join(data_path,'priceMT_results.txt'), index = True, header = False)