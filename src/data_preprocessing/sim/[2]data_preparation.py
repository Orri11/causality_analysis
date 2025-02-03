import os
import pandas as pd
import numpy as np


base_dir = os.path.join(os.path.dirname(__file__),'..', '..', '..')
data_path = os.path.join(base_dir, 'data/sim/')


seq_lens = [90,420]
no_series = [50,300]
trend = ['stationary','trend']
interv_type = ['hom'] 
pred_len = 24

for seq in seq_lens:
    for nseries in no_series:
        for tr in trend:
            for interv in interv_type:
                name = 'sim_{}_{}_{}_{}'.format(seq, nseries, tr ,interv)
                data = pd.read_csv(os.path.join(data_path, name + '.csv'))
                data=data.transpose()
                data.insert(0, 'ser_num', data.index)
                # Writing the full dataset in a txt file
                data.to_csv(os.path.join(data_path, name + '_full.txt'), index= False, header=False)
                # Train dataset 
                data_train = data.iloc[:,:data.shape[1] - pred_len]
                data_train.to_csv(os.path.join(data_path, name + '_train.txt'), index= False, header=False)
                # Test dataset - last prediction length columns
                data_test = data.iloc[:,data.shape[1] - pred_len:]
                data_test.to_csv(os.path.join(data_path, name + '_test.txt'), index= False, header=False)
