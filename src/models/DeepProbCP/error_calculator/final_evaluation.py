import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from quantile_utils.CRPS_QL import mean_weighted_quantile_loss
import pickle

# def custom_smape(forecasts, actual, epsilon=0.1):
#     comparator = 0.5 + epsilon
#     sum_term = np.maximum(comparator, (np.abs(forecasts) + np.abs(actual) + epsilon))
#     return 2 * np.abs(forecasts - actual) / sum_term

def custom_sort_key(s):
    parts = s.split('_')
    return int(parts[1])

def mase_greybox(holdout, forecast, scale):
    """
    Calculates Mean Absolute Scaled Error as in Hyndman & Koehler, 2006.
    
    Reference: https://github.com/config-i1/greybox/blob/6c84c729786f33a474ef833a13b7715831bd29e6/R/error-measures.R#L267

    Parameters:
        holdout (list or numpy array): Holdout values.
        forecast (list or numpy array): Forecasted values.
        scale (float): The measure to scale errors with. Usually - MAE of in-sample.
        na_rm (bool, optional): Whether to remove NA values from calculations.
                                Default is True.

    Returns:
        float: Mean Absolute Scaled Error.
    """
    if len(holdout) != len(forecast):
        print("The length of the provided data differs.")
        print(f"Length of holdout: {len(holdout)}")
        print(f"Length of forecast: {len(forecast)}")
        raise ValueError("Cannot proceed.")
    else:
        return np.mean(np.abs(np.array(holdout) - np.array(forecast)) / scale)

def evaluate(evaluate_args, ensembled_forecasts):

    # Example values for testing
    rnn_forecast_file_path = evaluate_args[0]
    errors_directory = evaluate_args[1]
    processed_forecasts_directory = evaluate_args[2]
    txt_test_file_name = evaluate_args[4]
    actual_results_file_name = evaluate_args[5]
    # original_data_file_name = evaluate_args[6]
    input_size = evaluate_args[6]
    output_size = evaluate_args[7]
    contain_zero_values = evaluate_args[8]
    address_near_zero_instability = evaluate_args[9]
    integer_conversion = evaluate_args[10]
    seasonality_period = evaluate_args[11]
    without_stl_decomposition = evaluate_args[12]
    dataset_type = evaluate_args[13]
    if dataset_type == 'sim':
        treated_units_indices = np.loadtxt(evaluate_args[14], dtype=int)
    # root_directory = '/path/to/root/directory/'

    # Errors file names
    errors_file_name = evaluate_args[3]
    errors_file_name_mean_median_control = 'mean_median_control_' + errors_file_name
    errors_file_name_mean_median_treated = 'mean_median_treated_' + errors_file_name
    SMAPE_file_name_all_errors_control = 'all_smape_errors_control_' + errors_file_name
    SMAPE_file_name_all_errors_treated = 'all_smape_errors_treated_' + errors_file_name
    MASE_file_name_all_errors_control = 'all_mase_errors_control_' + errors_file_name
    MASE_file_name_all_errors_treated = 'all_mase_errors_treated_' + errors_file_name
    CRPS_file_name_cs_control = errors_file_name + '_control_cs'
    CRPS_file_name_cs_treated= errors_file_name + '_treated_cs'
    errors_file_full_name_mean_median_control = errors_directory + errors_file_name_mean_median_control + '.txt'
    errors_file_full_name_mean_median_treated = errors_directory + errors_file_name_mean_median_treated + '.txt'
    SMAPE_file_full_name_all_errors_control = errors_directory + SMAPE_file_name_all_errors_control
    SMAPE_file_full_name_all_errors_treated = errors_directory + SMAPE_file_name_all_errors_treated
    MASE_file_full_name_all_errors_control = errors_directory + MASE_file_name_all_errors_control
    MASE_file_full_name_all_errors_treated = errors_directory + MASE_file_name_all_errors_treated
    CRPS_file_cs_control = errors_directory + CRPS_file_name_cs_control 
    CRPS_file_cs_treated = errors_directory + CRPS_file_name_cs_treated 

    if dataset_type=='elec_price':
        actual_results = pd.read_csv(actual_results_file_name,header=None).T
        actual_results.columns = actual_results.iloc[0]
        actual_results = actual_results.drop(0).reset_index(drop=True)
    elif dataset_type=='sim':
        actual_results = pd.read_csv(actual_results_file_name,header=None).iloc[:,1:].T.reset_index(drop=True)
    # print(actual_results)
    # print(" ")
    # if "series_id" in actual_results.columns:
    #     actual_results = actual_results.pivot(index='series_id', columns='time')['value']
    #     original_dataset = pd.read_csv(original_data_file_name)
    #     original_dataset = original_dataset.pivot(index='series_id', columns='time')['value']
    #     # Use the custom sorting function as the key
    #     actual_results = actual_results.loc[sorted(actual_results.index, key=custom_sort_key),:]
    #     original_dataset = original_dataset.loc[sorted(original_dataset.index, key=custom_sort_key),:]
    # else:
    #     actual_results = actual_results.iloc[:,1:]
    #     original_dataset = pd.read_csv(original_data_file_name)
    length_of_series = len(actual_results.index)

    data_row_A = actual_results.iloc[length_of_series-output_size:, :].T
    data_row_B = actual_results.iloc[:length_of_series-output_size, :].T

    # print(actual_results)
    # print(" ")
    # Text test data file name
    txt_test_df = pd.read_csv(txt_test_file_name, sep=' ', header=None)

    # RNN forecasts file name as one of the argument which is ensembled_forecasts

    # # Reading the original data to calculate the MASE errors
    # with open(original_data_file_name, 'r') as file:
    #     original_dataset = [line.strip().split(',') for line in file]
    

    # Persisting the final forecasts
    processed_forecasts_file = processed_forecasts_directory + errors_file_name

    # take the uniqueindexes
    value = list(txt_test_df[0])
    uniqueindexes = [i-2 for i, val in enumerate(value, start=1) if val != value[i-2] and i!= 1]
    uniqueindexes.append(len(value)-1)

    data_row_A = data_row_A.dropna()

    # initialize
    converted_forecasts_matrix = np.zeros((len(ensembled_forecasts[0.5]), output_size))
    # if dataset_type == 'calls911':
    #     control = ["BRIDGEPORT", "BRYN ATHYN", "DOUGLASS", "HATBORO", "HATFIELD BORO",
    #                 "LOWER FREDERICK", "NEW HANOVER", "NORRISTOWN", "NORTH WALES", "SALFORD",
    #                 "SPRINGFIELD", "TRAPPE"]
    #     converted_forecasts_matrix = np.zeros((len(control), output_size))
    # else:
    #     converted_forecasts_matrix = np.zeros((len(ensembled_forecasts[0.5]), output_size))

    mase_vector_treated = []
    mase_vector_control= []
    crps_vector_treated = []
    crps_vector_control= []
    # lambda_val = -0.7 useless

    for k, v in ensembled_forecasts.items():
        # Perform spline regression for each time series
        num_time_series = v.shape[0]
        print(num_time_series)

        if dataset_type == 'elec_price':
            control = ['AK', 'AL', 'AR', 'AZ', 'CO', 'DE', 'ID', 'FL', 'GA', 'HI', 'IA', 'IN', 'KS', 'KY', 'LA', 
                       'ME', 'MN', 'MI', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NM', 'NV', 'OH', 'OK', 'OR',
                       'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
            data_row_cols = actual_results.columns
            v['names'] = data_row_cols
            v.set_index('names', inplace=True)

        elif dataset_type == 'sim':
            control = np.setdiff1d(np.arange(0,v.shape[0]), treated_units_indices)
            data_row_cols = actual_results.columns


        for i in range(num_time_series):
            # post-processing
            # print(v.index[i])
            one_ts_forecasts = v.iloc[i].values
            finalindex = uniqueindexes[i]
            one_line_test_data = txt_test_df.iloc[finalindex].values
            mean_value = one_line_test_data[input_size + 2]
            level_value = one_line_test_data[input_size + 3]
            if without_stl_decomposition:
                converted_forecasts_df = np.exp(one_ts_forecasts + level_value)
            else:
                seasonal_values = one_line_test_data[(input_size + 4):(3 + input_size + output_size)]
                converted_forecasts_df = np.exp(one_ts_forecasts + level_value + seasonal_values)
            
            if contain_zero_values:
                converted_forecasts_df = converted_forecasts_df - 1

            converted_forecasts_df = mean_value * converted_forecasts_df
            converted_forecasts_df[converted_forecasts_df < 0] = 0

            if integer_conversion:
                converted_forecasts_df = np.round(converted_forecasts_df)

            converted_forecasts_matrix[i, :] = converted_forecasts_df # one_ts_forecasts
            
            if k == 0.5:
                # print(v.index[i])
                # np.diff(np.array(original_dataset[i]), lag=seasonality_period, differences=1))
                # original_values = list(map(float, original_dataset[i]))
                # print(original_dataset)
                # original_dataset_df = pd.DataFrame(original_dataset)
                # original_values = list(original_dataset_df.index+1)
                # print(original_dataset)
                #c = control.index(v.index[i])
                lagged_diff = [data_row_B.iloc[i,j] - \
                               data_row_B.iloc[i,j - \
                                seasonality_period] for j in \
                                range(seasonality_period, len(data_row_B.columns))]
                # print(np.array(actual_results_df.iloc[i]))
                # print(" ")
                # print(converted_forecasts_df)
                if v.index[i] in control:
                    mase_vector_control.append(mase_greybox(np.array(data_row_A.iloc[i]),\
                    converted_forecasts_df, np.mean(np.abs(lagged_diff))))
                else:
                    mase_vector_treated.append(mase_greybox(np.array(data_row_A.iloc[i]),\
                    converted_forecasts_df, np.mean(np.abs(lagged_diff))))

        converted_forecasts_m_df = pd.DataFrame(converted_forecasts_matrix)
        if dataset_type =='elec_price':
            converted_forecasts_m_df['names'] = data_row_cols
            converted_forecasts_m_df.set_index('names', inplace=True)
        if k == 0.5:  
            converted_forecasts_smape_control = converted_forecasts_m_df.loc[control,:]
            converted_forecasts_smape_treated = converted_forecasts_m_df.loc[~converted_forecasts_m_df.index.isin(control),:]
        crps_vector_control.append(converted_forecasts_m_df.loc[control,:])
        crps_vector_treated.append(converted_forecasts_m_df.loc[~converted_forecasts_m_df.index.isin(control),:])
        # Persisting the converted forecasts
        np.savetxt(processed_forecasts_file+'_'+str(k)+'.txt', converted_forecasts_matrix, delimiter=",")

    cs_control = CubicSpline(list(ensembled_forecasts.keys()), crps_vector_control, bc_type='natural')
    crps_y_pred_control = np.transpose(cs_control(list(ensembled_forecasts.keys())), (1, 0, 2))

    cs_treated = CubicSpline(list(ensembled_forecasts.keys()), crps_vector_treated, bc_type='natural')
    crps_y_pred_treated = np.transpose(cs_treated(list(ensembled_forecasts.keys())), (1, 0, 2))    
    
    # Calculating the CRPS
    crps_qs_control = mean_weighted_quantile_loss(crps_y_pred_control, np.array(data_row_A.loc[control,:]), ensembled_forecasts.keys())
    crps_qs_treated = mean_weighted_quantile_loss(crps_y_pred_treated, np.array(data_row_A.loc[~data_row_A.index.isin(control),:]), ensembled_forecasts.keys())
    
    mean_CRPS_control = np.mean(crps_qs_control)
    mean_CRPS_treated = np.mean(crps_qs_treated)
    

    mean_CRPS_str_control = f"mean_CRPS:{mean_CRPS_control}"
    all_CRPS_qs_control = f"CRPS for different quantiles:{crps_qs_control}"
    # std_CRPS_str = f"std_CRPS:{std_CRPS}"

    print(mean_CRPS_str_control)
    print(all_CRPS_qs_control)

    mean_CRPS_str_treated = f"mean_CRPS:{mean_CRPS_treated}"
    all_CRPS_qs_treated= f"CRPS for different quantiles:{crps_qs_treated}"
    

    # Calculating the SMAPE
    if address_near_zero_instability:
        epsilon = 0.1
        comparator = 0.5 + epsilon
        sum_term_control = np.maximum(comparator, (np.abs(converted_forecasts_smape_control) + np.abs(np.array(data_row_A.loc[control,:])) + epsilon))
        time_series_wise_SMAPE_control = 2 * np.abs(converted_forecasts_smape_control - np.array(data_row_A.loc[control,:])) / sum_term_control

        sum_term_treated = np.maximum(comparator, (np.abs(converted_forecasts_smape_treated) + np.abs(np.array(data_row_A.loc[~data_row_A.index.isin(control),:])) + epsilon))
        time_series_wise_SMAPE_treated = 2 * np.abs(converted_forecasts_smape_treated - np.array(data_row_A.loc[~data_row_A.index.isin(control),:])) / sum_term_treated
    else:
        time_series_wise_SMAPE_control = 2 * np.abs(converted_forecasts_smape_control - np.array(data_row_A.loc[control,:])) / \
            (np.abs(converted_forecasts_smape_control) + np.abs(np.array(data_row_A.loc[control,:])))

        time_series_wise_SMAPE_treated = 2 * np.abs(converted_forecasts_smape_treated - np.array(data_row_A.loc[~data_row_A.index.isin(control),:])) / \
            (np.abs(converted_forecasts_smape_treated) + np.abs(np.array(data_row_A.loc[~data_row_A.index.isin(control),:])))
    SMAPEPerSeries_control = np.mean(time_series_wise_SMAPE_control, axis=1) #
    SMAPEPerSeries_treated= np.mean(time_series_wise_SMAPE_treated, axis=1)

    mean_SMAPE_control = np.mean(SMAPEPerSeries_control)
    mean_SMAPE_treated= np.mean(SMAPEPerSeries_treated)
    # median_SMAPE = np.median(SMAPEPerSeries)
    # std_SMAPE = np.std(SMAPEPerSeries)

    mean_SMAPE_str_control = f"mean_SMAPE:{mean_SMAPE_control}"
    mean_SMAPE_str_treated = f"mean_SMAPE:{mean_SMAPE_treated}"
    # median_SMAPE_str = f"median_SMAPE:{median_SMAPE}"
    # std_SMAPE_str = f"std_SMAPE:{std_SMAPE}"

    print(mean_SMAPE_str_control)
    # print(median_SMAPE_str)
    # print(std_SMAPE_str)

    # MASE
    mean_MASE_control = np.mean(mase_vector_control)
    mean_MASE_treated = np.mean(mase_vector_treated)
    # median_MASE = np.median(mase_vector)
    # std_MASE = np.std(mase_vector)

    mean_MASE_str_control = f"mean_MASE:{mean_MASE_control}"
    mean_MASE_str_treated = f"mean_MASE:{mean_MASE_treated}"
    # median_MASE_str = f"median_MASE:{median_MASE}"
    # std_MASE_str = f"std_MASE:{std_MASE}"

    print(mean_MASE_str_control)
    # print(median_MASE_str)
    # print(std_MASE_str)


    # Writing the SMAPE results to file
    with open(errors_file_full_name_mean_median_control, 'w') as f:
        # f.write('\n'.join([mean_SMAPE_str, median_SMAPE_str, std_SMAPE_str]))
        f.write('\n'.join([mean_SMAPE_str_control]))

    with open(errors_file_full_name_mean_median_treated, 'w') as f:
        # f.write('\n'.join([mean_SMAPE_str, median_SMAPE_str, std_SMAPE_str]))
        f.write('\n'.join([mean_SMAPE_str_treated]))

    np.savetxt(SMAPE_file_full_name_all_errors_control+'.txt', SMAPEPerSeries_control, delimiter=",", fmt='%f')
    np.savetxt(SMAPE_file_full_name_all_errors_treated+'.txt', SMAPEPerSeries_treated, delimiter=",", fmt='%f')
    # Writing the MASE results to file
    with open(errors_file_full_name_mean_median_control, 'a') as f:
        # f.write('\n'.join([mean_MASE_str, median_MASE_str, std_MASE_str]))
        f.write('\n'.join([mean_MASE_str_control]))
    with open(errors_file_full_name_mean_median_treated, 'a') as f:
        # f.write('\n'.join([mean_MASE_str, median_MASE_str, std_MASE_str]))
        f.write('\n'.join([mean_MASE_str_treated]))

    np.savetxt(MASE_file_full_name_all_errors_control+'.txt', mase_vector_control, delimiter=",", fmt='%f')
    np.savetxt(MASE_file_full_name_all_errors_treated+'.txt', mase_vector_treated, delimiter=",", fmt='%f')
    # Writing the CRPS results to file
    with open(errors_file_full_name_mean_median_control, 'a') as f:
        f.write('\n'.join([mean_CRPS_str_control]))
    with open(errors_file_full_name_mean_median_treated, 'a') as f:
        f.write('\n'.join([mean_CRPS_str_treated]))

    #with open(CRPS_file_cs+'.pickle', 'wb') as f:
        #pickle.dump(crps_vector, f)

    np.savetxt(CRPS_file_cs_control + '.txt' , crps_qs_control, delimiter=",", fmt='%f')
    np.savetxt(CRPS_file_cs_treated + '.txt' , crps_qs_treated, delimiter=",", fmt='%f')   