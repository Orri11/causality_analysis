# Record the running time
print('begin')
import time
T1 = time.time()

# Inbuilt or External Modules
import argparse # customized arguments in .bash
import csv # input and output .csv data
import glob 
import numpy as np 
import os 
import pandas as pd
import pickle

# Customized Modules
from configs.global_configs import hyperparameter_tuning_configs
from configs.global_configs import model_testing_configs
from error_calculator.final_evaluation import evaluate
from src.models.DeepProbCP.ensembling_forecasts import ensembling_forecasts
# from utility_scripts.invoke_final_evaluation import invoke_script # for invoking R
from src.models.DeepProbCP.hyperparameter_config_reader import read_initial_hyperparameter_values, read_optimal_hyperparameter_values
from src.models.DeepProbCP.persist_optimized_config_results import persist_results

# import SMAC utilities
# import the config space and the different types of parameters
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

# stacking model
from rnn_architectures.stacking_model_p import StackingModel


LSTM_USE_PEEPHOLES = True # LSTM with “peephole connections"
BIAS = False # in tf.keras.layers.dense


# final execution with the optimized config
def train_model(configs):
    print(configs)

    hyperparameter_values = {
        "num_hidden_layers": configs["num_hidden_layers"],
        "cell_dimension": configs["cell_dimension"],
        "minibatch_size": configs["minibatch_size"],
        "max_epoch_size": configs["max_epoch_size"],
        "max_num_epochs": configs["max_num_epochs"],
        "l2_regularization": configs["l2_regularization"],
        "gaussian_noise_stdev": configs["gaussian_noise_stdev"],
        "random_normal_initializer_stdev": configs["random_normal_initializer_stdev"],
    }

    if configs["optimizer"] != "cocob":
        hyperparameter_values["initial_learning_rate"] = configs["initial_learning_rate"]

    error = model.tune_hyperparameters(**hyperparameter_values)

    print(model_identifier)
    print(error)
    return error.item()

def smac():
    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    initial_learning_rate = UniformFloatHyperparameter("initial_learning_rate", hyperparameter_values_dic['initial_learning_rate'][0],
                                                  hyperparameter_values_dic['initial_learning_rate'][1],
                                                  default_value=hyperparameter_values_dic['initial_learning_rate'][0])
    cell_dimension = UniformIntegerHyperparameter("cell_dimension",
                                                  hyperparameter_values_dic['cell_dimension'][0],
                                                  hyperparameter_values_dic['cell_dimension'][1],
                                                  default_value=hyperparameter_values_dic['cell_dimension'][
                                                      0])
    no_hidden_layers = UniformIntegerHyperparameter("num_hidden_layers",
                                                    hyperparameter_values_dic['num_hidden_layers'][0],
                                                    hyperparameter_values_dic['num_hidden_layers'][1],
                                                    default_value=hyperparameter_values_dic['num_hidden_layers'][0])
    minibatch_size = UniformIntegerHyperparameter("minibatch_size", hyperparameter_values_dic['minibatch_size'][0],
                                                  hyperparameter_values_dic['minibatch_size'][1],
                                                  default_value=hyperparameter_values_dic['minibatch_size'][0])
    max_epoch_size = UniformIntegerHyperparameter("max_epoch_size", hyperparameter_values_dic['max_epoch_size'][0],
                                                  hyperparameter_values_dic['max_epoch_size'][1],
                                                  default_value=hyperparameter_values_dic['max_epoch_size'][0])
    max_num_of_epochs = UniformIntegerHyperparameter("max_num_epochs", hyperparameter_values_dic['max_num_epochs'][0],
                                                     hyperparameter_values_dic['max_num_epochs'][1],
                                                     default_value=hyperparameter_values_dic['max_num_epochs'][0])
    l2_regularization = UniformFloatHyperparameter("l2_regularization",
                                                   hyperparameter_values_dic['l2_regularization'][0],
                                                   hyperparameter_values_dic['l2_regularization'][1],
                                                   default_value=hyperparameter_values_dic['l2_regularization'][0])
    gaussian_noise_stdev = UniformFloatHyperparameter("gaussian_noise_stdev",
                                                      hyperparameter_values_dic['gaussian_noise_stdev'][0],
                                                      hyperparameter_values_dic['gaussian_noise_stdev'][1],
                                                      default_value=hyperparameter_values_dic['gaussian_noise_stdev'][
                                                          0])
    random_normal_initializer_stdev = UniformFloatHyperparameter("random_normal_initializer_stdev",
                                                                 hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][0],
                                                                 hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][1],
                                                                 default_value=hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][
                                                                     0])

    # add the hyperparameter for learning rate only if the  optimization is not cocob
    if optimizer == "cocob":
        configuration_space.add_hyperparameters(
            [cell_dimension, no_hidden_layers, minibatch_size, max_epoch_size, max_num_of_epochs,
             l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])
    else:

        configuration_space.add_hyperparameters(
            [initial_learning_rate, cell_dimension, minibatch_size, max_epoch_size,
             max_num_of_epochs, no_hidden_layers,
             l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])

    # creating the scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": hyperparameter_tuning_configs.SMAC_RUNCOUNT_LIMIT,
        "cs": configuration_space,
        "deterministic": "true",
        "abort_on_first_run_crash": "false"
    })

    # optimize using an SMAC object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=train_model)

    incumbent = smac.optimize()
    return incumbent.get_dictionary()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    argument_parser.add_argument('--dataset_type', required=True,
                                 help='elec_price/sim/...')
    argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required=False,
                                 help='Whether the dataset contains zero values(0/1). Default is 0')
    argument_parser.add_argument('--address_near_zero_instability', required=False,
                                 help='Whether to use a custom SMAPE function to address near zero instability(0/1). Default is 0')
    argument_parser.add_argument('--integer_conversion', required=False,
                                 help='Whether to convert the final forecasts to integers(0/1). Default is 0')
    argument_parser.add_argument('--no_of_series', required=True,
                                 help='The number of series in the dataset')
    argument_parser.add_argument('--cell_type', required=False,
                                 help='The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM')
    argument_parser.add_argument('--input_size', required=True,
                                 help='The input size of the moving window. Default is 0')
    argument_parser.add_argument('--seasonality_period', required=True, help='The seasonality period of the time series')
    argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    argument_parser.add_argument('--optimizer', required=False, help='The type of the optimizer(cocob/adam/adagrad...). Default is cocob')
    argument_parser.add_argument('--quantile_range', required=False, 
                                 help='The range of the quantile for quantile forecasting. Default is np.linspace(0, 1, 21)')
    argument_parser.add_argument('--evaluation_metric', required=False, 
                                 help='The evaluation metric like sMAPE. Default is CRPS')
    argument_parser.add_argument('--without_stl_decomposition', required=False,
                                 help='Whether not to use stl decomposition(0/1). Default is 0')

    # parse the user arguments
    args = argument_parser.parse_args()

    # arguments with no default values
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    no_of_series = int(args.no_of_series)
    input_size = int(args.input_size)
    output_size = int(args.forecast_horizon)
    seasonality_period = int(args.seasonality_period)
    seed = 1234

    # arguments with default values
    if args.contain_zero_values:
        contain_zero_values = bool(int(args.contain_zero_values))
    else:
        contain_zero_values = False

    if args.optimizer:
        optimizer = args.optimizer
    else:
        optimizer = "cocob"
    
    if args.quantile_range:
        quantile_range = args.quantile_range
    else:
        quantile_range = [0.1,0.5,0.9]
    
    if args.evaluation_metric:
        evaluation_metric = args.evaluation_metric
    else:
        # evaluation_metric = "sMAPE"
        evaluation_metric = "CRPS"

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

    if args.cell_type:
        cell_type = args.cell_type
    else:
        cell_type = "LSTM"

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False

    if without_stl_decomposition:
        stl_decomposition_identifier = "without_stl_decomposition"
    else:
        stl_decomposition_identifier = "with_stl_decomposition"

    if dataset_type == 'elec_price':
        model_identifier = dataset_name + "_" + cell_type + "cell" + "_" +  optimizer + "_" + \
                       str(input_size - 1) + "_" + str(output_size) + "_" + stl_decomposition_identifier
        print("Model Training Started for {}".format(model_identifier))
    

        initial_hyperparameter_values_file = "src/models/DeepProbCP/configs/initial_hyperparameter_values/" + \
        "ems_adagrad"
        binary_train_file_path_train_mode = "data/" + args.dataset_type + "/binary_data/" + \
          dataset_name + "_" + str(input_size -1) + "_" + str(args.forecast_horizon) + "_" + "train" + ".tfrecords"
        binary_validation_file_path_train_mode = "data/" + args.dataset_type + "/binary_data/"  +  \
         dataset_name +  "_" + str(input_size - 1) + "_" + str(args.forecast_horizon) + "_" + "val" + ".tfrecords"
        binary_train_file_test_mode = "data/" + args.dataset_type + "/binary_data/"  +  \
         dataset_name + "_" + str(input_size -1) + "_" + str(args.forecast_horizon) + "_" + "val" + ".tfrecords"
        binary_test_file_path_test_mode = "data/" + args.dataset_type + "/binary_data/"  +  \
         dataset_name + "_" + str(input_size -1)  + "_" + str(args.forecast_horizon) + "_" + "test" + ".tfrecords"
        txt_test_file_path = "data/" + args.dataset_type +  "/moving_window/" + dataset_name + "_" + \
         str(input_size -1) + "_" + str(args.forecast_horizon) + "_" +  "test" + ".txt" 
        actual_results_file_path = "data/" + args.dataset_type +  \
        "/" + dataset_name + "_full.txt"
    if dataset_type == 'sim':
        treated_units_index_file = "data/" + args.dataset_type + "/" + dataset_name + "_treated_indices.txt"
        counterfactuals_file_path = "data/" + args.dataset_type +  \
        "/" + dataset_name + "_true_counterfactual.csv"
    # original_data_file_path = "datasets/text_data/" + args.dataset_type +  \
    #     "/" + dataset_name + "_train.csv"
    # define the key word arguments for the different model types
    model_kwargs = {
        'use_bias': BIAS,
        'use_peepholes': LSTM_USE_PEEPHOLES,
        'input_size': input_size,
        'output_size': output_size,
        'optimizer': optimizer,
        'quantile_range': quantile_range,
        'evaluation_metric': evaluation_metric,
        'no_of_series': no_of_series,
        'binary_train_file_path': binary_train_file_path_train_mode,
        'binary_test_file_path': binary_test_file_path_test_mode,
        'binary_validation_file_path': binary_validation_file_path_train_mode,
        'contain_zero_values': contain_zero_values,
        'address_near_zero_instability': address_near_zero_instability,
        'integer_conversion': integer_conversion,
        'seed': seed,
        'cell_type': cell_type,
        'without_stl_decomposition': without_stl_decomposition
    }
    
    # select the model type
    model = StackingModel(**model_kwargs)
    
    # delete hyperparameter configs files if existing
    if dataset_type == 'elec_price':
        for file in glob.glob(hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY_ELEC + model_identifier + "*"):
            os.remove(file)
    elif dataset_type == 'sim':
        for file in glob.glob(hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY_SIM + model_identifier + "*"):
            os.remove(file)

    # read the initial hyperparamter configurations from the file
    hyperparameter_values_dic = read_initial_hyperparameter_values(initial_hyperparameter_values_file)
    # tune the hyperparameters
    optimized_configuration = smac()
    print(optimized_configuration)

    # persist the optimized configuration to a file
    if dataset_type == 'elec_price':
        persist_results(optimized_configuration, hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY_ELEC + '/' + model_identifier + '.txt')
    elif dataset_type == 'sim':
        persist_results(optimized_configuration, hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY_SIM + '/' + model_identifier + '.txt')

    # not training again but just read in
    # # optimized_configuration = read_optimal_hyperparameter_values("./results/DeepProbCP/optimized_configurations/" + model_identifier + ".txt")
    # # print(optimized_configuration)

    # delete the forecast files if existing
    if dataset_type == 'elec_price':
        for file in glob.glob(
            model_testing_configs.FORECASTS_DIRECTORY_ELEC + model_identifier + "*"):
            os.remove(file)
    elif dataset_type == 'sim':
        for file in glob.glob(
            model_testing_configs.FORECASTS_DIRECTORY_SIM + model_identifier + "*"):
            os.remove(file)

    print("tuning finished")
    T2 = time.time()
    print(T2)
    print(dataset_name, no_of_series)
    # train the model with the optimized configuration and generate forcacsts
    for seed in range(1, 11):
         forecasts = model.test_model(optimized_configuration, seed)

         model_identifier_extended = model_identifier + "_" + str(seed)
         for k, v in forecasts.items():
            
            if dataset_type == 'elec_price':
                rnn_forecasts_file_path = model_testing_configs.FORECASTS_DIRECTORY_ELEC + model_identifier_extended + 'q_' + str(k) + '.txt'
            elif dataset_type == 'sim':
                rnn_forecasts_file_path = model_testing_configs.FORECASTS_DIRECTORY_SIM + model_identifier_extended + 'q_' + str(k) + '.txt'
            
            with open(rnn_forecasts_file_path, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(forecasts[k])
    print("prediction finished")
    T3 = time.time()
    
    
    # delete the ensembled forecast files if existing
    if dataset_type == 'elec_price':
        for file in glob.glob(
            model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_ELEC + model_identifier + "*"):
            os.remove(file)
    elif dataset_type == 'sim':
        for file in glob.glob(
            model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_SIM + model_identifier + "*"):
            os.remove(file)

    # ensemble the forecasts
    if dataset_type == 'elec_price':
        ensembled_forecasts = ensembling_forecasts(model_identifier, model_testing_configs.FORECASTS_DIRECTORY_ELEC,
                          model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_ELEC,quantile_range)
    elif dataset_type == 'sim':
        ensembled_forecasts = ensembling_forecasts(model_identifier, model_testing_configs.FORECASTS_DIRECTORY_SIM,
                          model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_SIM,quantile_range)

    '''
    
    # not training again but just read in
    ensembled_forecasts = {}
    if dataset_type == 'elec_price':
        for q in quantile_range:
            ensembled_forecasts[q] = pd.read_csv(model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_ELEC +\
                                                model_identifier + "_" + str(q) +".txt",sep = ",", header=None)
    elif dataset_type == 'sim':
        for q in quantile_range:
            ensembled_forecasts[q] = pd.read_csv(model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_SIM +\
                                               model_identifier + "_" + str(q) +".txt",sep = ",", header=None)
    '''

    print("ensembled finished")
    T4 = time.time()
    
    if dataset_type == 'elec_price':
       evaluate_args = [model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_ELEC,
                   model_testing_configs.ENSEMBLE_ERRORS_DIRECTORY_ELEC,
                   model_testing_configs.PROCESSED_ENSEMBLE_FORECASTS_DIRECTORY_ELEC,
                   model_identifier,
                   txt_test_file_path,
                   actual_results_file_path,
                #    original_data_file_path,
                   input_size,
                   output_size,
                   int(contain_zero_values),
                   int(address_near_zero_instability),
                   int(integer_conversion),
                   seasonality_period,
                   int(without_stl_decomposition),
                   args.dataset_type] 
    elif dataset_type == 'sim':
        evaluate_args = [model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY_SIM,
                   model_testing_configs.ENSEMBLE_ERRORS_DIRECTORY_SIM,
                   model_testing_configs.PROCESSED_ENSEMBLE_FORECASTS_DIRECTORY_SIM,
                   model_identifier,
                   txt_test_file_path,
                   counterfactuals_file_path,
                #    original_data_file_path,
                   input_size,
                   output_size,
                   int(contain_zero_values),
                   int(address_near_zero_instability),
                   int(integer_conversion),
                   seasonality_period,
                   int(without_stl_decomposition),
                   args.dataset_type,
                   treated_units_index_file]

    evaluate(evaluate_args, ensembled_forecasts)
    
    T5 = time.time()


    print('Running time: %s m' % ((T5 - T1) / 60))
