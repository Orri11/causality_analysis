# Causality analysis of electricity market liberalization on electricity price using novel Machine Learning methods
Type: Master's Thesis

Author: Orr Shahar

1st Examiner: Prof. Doc. Stefan Lessmann

2nd Examiner: Doc. Sigbert Klinke

Electricity price data

![Screenshot 2025-03-26 at 16 04 07](https://github.com/user-attachments/assets/f524cf66-3c85-44e8-9986-0469c07b43ce)

![Screenshot 2025-03-26 at 16 09 36](https://github.com/user-attachments/assets/6675ed7f-0a43-4bd1-8cb5-18095b42982c)

Synthetic dataset results:

![Screenshot 2025-03-26 at 16 06 49](https://github.com/user-attachments/assets/d13d7c58-7d84-4b24-8f07-3055ba2d75ad)

Electricity price results:

![Screenshot 2025-03-26 at 16 07 22](https://github.com/user-attachments/assets/8fa236ee-fd6e-45cd-9a6a-5fea7f528609)

ATT estimation:

![Screenshot 2025-03-26 at 16 08 06](https://github.com/user-attachments/assets/40805db8-cb6f-4304-9a80-f4d062412f2e)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Data](#Data)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
- [Results](#Results)
- [Project structure](#Project-structure)

## Summary

Relationships between the energy and the finance markets are increasingly important. Understanding these relationships is vital for policymakers and other stakeholders. One of the most complex yet important relationships is the identification and estimation of causality impact. The development of machine learning in recent years opened the door for a new branch of machine learning models for causality impact. In this research, we wish to evaluate novel machine learning methods for causality analysis and assess their adequacy for the single intervention case. We empirically compare the performance of different frameworks on synthetic data, and then utilize these models to estimate the causal effect of electricity market liberalization on the electricity price in the US. By performing this analysis, we aim to shed more light on the applicability of causal frameworks to energy policy intervention cases. We also aim to provide new insights into the ongoing debate about the benefits of electricity market liberalization

## Working with the repo

### Dependencies

Python = 3.9

### Setup

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements from requirements.txt

4. Path Variables
Set the PYTHONPATH env variable of the system. Append absolute paths of both the project root directory and the directory of the src/models/DeepProbCP/cocob_optimizer into the PYTHONPATH for the model DeepProbCP
Set the project root directory as the working directory.

For R scripts, make sure to set the working directory to the project root folder.


## Reproducing results

### Data

#### Synthetic Dataset

The synthetic dataset is constructed using the code /src/data_preprocessing/sim/sim_data_generator.py. This is the base code for the creation of a stationary dataset, as well as an additional trend and intervention. 
For the synthetic dataset, we create multiple scenarios combining long & short lengths with many & few time series. Each combination is created as stationary as well as with a trend. The scenarios are created in the code available in /src/data_preprocessing/sim/[1]prepare_sim_data.py.

For DeepProbCP, the data goes through further pre-processing steps. These steps include organizing the data for the moving window and conversion to .tfrecords. All modules for these pre-processing steps are in /src/data_preprocessing/sim/

#### Real World Data

The raw electricity price dataset is located at /data/elec_price/elec_prices.xlsx. The additional datasets for EDA dereg_year.xlsx and gen_prod_type.xls, as well as the external covariates raw datasets are located in the same folder.
The code for the pre-processing and EDA of the real-world dataset is located at src/data_preprocessing/elec_price/.
For DeepProbCP, the data goes through the same pre-processing steps as in the synthetic dataset. For the rest of the models, the data is set for the right format through [5]benchmarks.py.
Additionally, the external covariates are pre-processed in external_covariates.ipynb.

### Training code

1. ASCM -  Run the codes in src/models/ascm/ascm_sim.R for synthetic dataset and src/models/ascm/ascm_elec_price.R for electricity price
2. CausalArima -  Run the codes in src/models/ascm/CausalArima_sim.R.R for synthetic dataset and src/models/ascm/CausalArima_elec_price.R for electricity price
3. TSMixer Run the codes in src/models/TSMixer/tsmixer.ipynb
4. DeepProbCP Run the codes in src/models/DeepProbCP/DeepProbCP.sh

### Evaluation code

1. Metrics - The sMAPE and MASE are calculated and saved in each model's code. These metrics can be found at /results/../../metrics/
2. The final evaluation and ATE estimation of the synthetic dataset can be found at /results/sim_results_evaluation.ipynb
3. The final evaluation and ATE estimation of the real-world dataset, along with the placebo test results can be found at /results/elec_price_results_evaluation.ipynb.ipynb

## Results

The figures can be found at /figs. 
All figures and tables related to the models' performance and ATE are created in the scripts /results/sim_results_evaluation.ipynb and /results/elec_price_results_evaluation.ipynb

## Project structure

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file 
├── figs                                            -- stores image files
├── results
    ├── elec_price_results_evaluation.ipynb         -- elec_price final evaluation     
    ├── sim_results_evaluation.ipynb                -- sim final evaluation
    ├── elec_price                                  -- Models' forecasts and metrics
    ├── sim                                         -- Models' forecasts and metrics
└── src
    ├── data_preprocessing                          
        ├── elec_price                              -- preprocesses data
        ├── sim                                     -- preprocesses data
    ├── models
         ├── DeepProbCP
         ├── TSMixer
         ├── ASCM
         ├── CausalArima               
        
```
