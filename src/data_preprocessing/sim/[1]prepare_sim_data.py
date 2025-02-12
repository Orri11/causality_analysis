import pandas as pd
import numpy as np
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.data_preprocessing.sim.sim_data_generator import SynthesisTS

root_dir = os.path.join(os.path.dirname(__file__), '..','..','..')
data_dir = os.path.join(root_dir, 'data/sim/') 

seq_lens = [90,420]
no_series = [50,300]
trend = [True,False]
interv_type = ['hom'] 

synth_maker = SynthesisTS(cycle_periods = [1,7,30] , seq_len = 90, distort_cycles=False)
stationary = synth_maker.synthesize_series()
trend = synth_maker.add_multiplicative_trend(stationary, trend_rate=1.003)
intervention = synth_maker.add_intervention(stationary, type='het')

for seq in seq_lens:
    for series in no_series:
        for tr in trend:
            for interv in interv_type:
                synth_maker = SynthesisTS(cycle_periods = [1,7,30] , seq_len = seq, distort_cycles=False, series_amount = series)
                stationary = synth_maker.synthesize_series()
                if tr:
                    if seq == 90:
                        trend = synth_maker.add_multiplicative_trend(stationary, trend_rate=1.003)
                    else:
                        trend = synth_maker.add_multiplicative_trend(stationary, trend_rate=1.001)
                else:
                    trend = stationary
                if interv == 'hom':
                    intervention, treat_index = synth_maker.add_intervention(trend, length = 24, type='hom')
                else:
                    intervention, treat_index = synth_maker.add_intervention(trend, length = 24, type='het')
                if tr:
                    intervention.to_csv(data_dir + 'sim_{}_{}_{}_{}.csv'.format(seq, series, 'trend', interv),index=False)
                    trend.to_csv(data_dir + 'sim_{}_{}_{}_{}_true_counterfactual.csv'.format(seq, series, 'trend', interv),index=False)
                    np.savetxt(data_dir+ 'sim_{}_{}_{}_{}_treated_indices.txt'.format(seq, series, 'trend', interv), treat_index, delimiter=",", fmt='%d')
                else:
                    intervention.to_csv(data_dir + 'sim_{}_{}_{}_{}.csv'.format(seq, series, 'stationary', interv),index=False)
                    trend.to_csv(data_dir + 'sim_{}_{}_{}_{}_true_counterfactual.csv'.format(seq, series, 'stationary', interv),index=False)
                    np.savetxt(data_dir + 'sim_{}_{}_{}_{}_treated_indices.txt'.format(seq, series, 'stationary', interv), treat_index, delimiter=",", fmt='%d')