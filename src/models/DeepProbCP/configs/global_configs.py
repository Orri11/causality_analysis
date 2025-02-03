import os

# configs for the model training
class data_configs:
    TXT_DATA_ROOT_FOLDER = "text_data/"
    BINARY_DATA_ROOT_FOLDER = "binary_data/"

# class preprocess_configs:
#     PREPROCESS_SCRIPTS_FOLDER = "preprocess_scripts/"

class model_training_configs:
    VALIDATION_ERROR_CALCULATOR_MODULE_PATH = 'error_calculator.validation'

# configs for the model testing
class model_testing_configs:
    TESTING_ERROR_CALCULATOR_DIRECTORY = './error_calculator/final_evaluation.R'
    ERRORS_DIRECTORY = './results/errors/'

    # RNN related
    FORECASTS_DIRECTORY_ELEC = './results/elec_price/DeepProbCP/forecasts/'
    ENSEMBLE_FORECASTS_DIRECTORY_ELEC  = './results/elec_price/DeepProbCP/ensemble_forecasts/'
    ENSEMBLE_CS_FORECASTS_DIRECTORY_ELEC  = './results/elec_price/DeepProbCP/ensemble_interpolated_forecasts/'
    ENSEMBLE_ERRORS_DIRECTORY_ELEC  = './results/elec_price/DeepProbCP/metrics/'
    PROCESSED_ENSEMBLE_FORECASTS_DIRECTORY_ELEC  = './results/elec_price/DeepProbCP/processed_ensemble_forecasts/'
    AGGREGATED_ENSEMBLE_ERRORS_DIRECTORY_ELEC  = './results/elec_price/DeepProbCP/ensemble_errors/aggregate_errors/'

    FORECASTS_DIRECTORY_SIM = './results/sim/DeepProbCP/forecasts/'
    ENSEMBLE_FORECASTS_DIRECTORY_SIM   = './results/sim/DeepProbCP/ensemble_forecasts/'
    ENSEMBLE_CS_FORECASTS_DIRECTORY_SIM  = './results/sim/DeepProbCP/ensemble_interpolated_forecasts/'
    ENSEMBLE_ERRORS_DIRECTORY_SIM  = './results/sim/DeepProbCP/metrics/'
    PROCESSED_ENSEMBLE_FORECASTS_DIRECTORY_SIM  = './results/sim/DeepProbCP/processed_ensemble_forecasts/'
    AGGREGATED_ENSEMBLE_ERRORS_DIRECTORY_SIM  = './results/sim/DeepProbCP/ensemble_errors/aggregate_errors/'

    # aggregated general errors
    AGGREGATED_ERRORS_DIRECTORY_ELEC = './results/elec_price/errors/aggregate_errors/'
    AGGREGATED_ERRORS_DIRECTORY_SIM = './results/sim/errors/aggregate_errors/'
# configs for hyperparameter tuning(SMAC3)
class hyperparameter_tuning_configs:
    SMAC_RUNCOUNT_LIMIT = 50
    SMAC_RUNCOUNT_LIMIT_PER_SEQ = 50

    # RNN related
    OPTIMIZED_CONFIG_DIRECTORY_ELEC = './results/elec_price/DeepProbCP/optimized_configurations'
    OPTIMIZED_CONFIG_DIRECTORY_SIM = './results/sim/DeepProbCP/optimized_configurations'
class training_data_configs:
    SHUFFLE_BUFFER_SIZE = 1000
