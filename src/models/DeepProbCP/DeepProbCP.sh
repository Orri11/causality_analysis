#!/usr/bin/env bash

############## LSTM Cell
#### NASS AUSTRALIAN EMS CALLS DATASET
###  DeepCPNet Framework
## window size 15 + 1 (STL decomposition as exogenous variable) = 16
## forecasting horizon 12


###############################################################################################################################################################
### electricity_price 
python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type elec_price --dataset_name priceMT --contain_zero_values 0 --input_size 37 --forecast_horizon 24 --no_of_series 50 --optimizer cocob --seasonality_period 12 --address_near_zero_instability 0 --without_stl_decomposition 1 &

### sim 
#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_90_50_stationary_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 50 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &
#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_90_50_trend_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 50 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &

#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_90_300_stationary_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 300 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &
#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_90_300_trend_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 300 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &

#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_420_50_stationary_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 50 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &
#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_420_50_trend_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 50 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &

#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_420_300_stationary_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 300 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &
#python ./src/models/DeepProbCP/generic_model_handler.py --dataset_type sim --dataset_name sim_420_300_trend_hom --contain_zero_values 0 --input_size 13 --forecast_horizon 24 --no_of_series 300 --optimizer cocob --seasonality_period 30 --address_near_zero_instability 0 --without_stl_decomposition 1 &

# python ./benchmark_model_handler.py --model DeepProbCP --dataset_name calls911_benchmarks --feature_type 'MS' --initial_hyperparameter_values_file configs/initial_hyperparameter_values/ems_DeepProbCP_cocob --no_of_series 62 --input_size 15 --forecast_horizon 7 --optimizer cocob --seasonality_period 12 --original_data_file datasets/text_data/calls911/callsMT2_dataset.txt --address_near_zero_instability 0 --without_stl_decomposition 1 &
# python ./benchmark_model_handler.py --model CausalImpact --dataset_name calls911_benchmarks --feature_type 'MS' --initial_hyperparameter_values_file configs/initial_hyperparameter_values/ems_DeepProbCP_cocob --no_of_series 62 --input_size 15 --forecast_horizon 7 --optimizer cocob --seasonality_period 12 --original_data_file datasets/text_data/calls911/callsMT2_dataset.txt --address_near_zero_instability 0 --without_stl_decomposition 1 &

#stacking_moving_window_smac_adagrad
#python ./generic_model_handler.py --dataset_name callsMT215_without_stl --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/ems_adagrad --binary_train_file_train_mode datasets/binary_data/calls911/moving_window/without_stl_decomposition/callsMT2_7_15.tfrecords --binary_valid_file_train_mode datasets/binary_data/calls911/moving_window/without_stl_decomposition/callsMT2_7_15v.tfrecords --binary_train_file_test_mode datasets/binary_data/calls911/moving_window/without_stl_decomposition/callsMT2_7_15v.tfrecords --binary_test_file_test_mode datasets/binary_data/calls911/moving_window/without_stl_decomposition/callsMT2_test_7_15.tfrecords --txt_test_file datasets/text_data/calls911/moving_window/without_stl_decomposition/callsMT2_test_7_15.txt --actual_results_file datasets/text_data/calls911/callsMT2_results.txt --input_size 16 --forecast_horizon 7 --no_of_series 62 --optimizer adagrad --seasonality_period 12 --original_data_file datasets/text_data/calls911/callsMT2_dataset.txt --address_near_zero_instability 0 --without_stl_decomposition 1 &

