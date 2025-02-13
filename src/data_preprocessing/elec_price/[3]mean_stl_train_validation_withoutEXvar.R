## Dataset: electricity price in US states
# Forecasting task: counterfactual prediction in order to estimate the treatment effect
# of electricity market liberalization on the price.
# Counterfactual period : 1998-01 - 1999-12. Training dataset - 1992-01 - 1997-12

## Preprocessing script code that: (1) split the training dataset creating the training and 
## validation files dataset; (2) applys the preprocessing steps to both these datasets

# Please uncomment the below command, in case you haven't installed the 
# following pakcage in your enviornment.
#install.packages("forecast")
library(forecast)

# Please set your working directory as the local repository directory before running the script

# reproducability.
set.seed(1234)

# Loading the training dataset.with training period from 1992-01 to 1997-12
main_dir = normalizePath("./")
input_file = paste(main_dir,"/data/elec_price/elec_price_train.txt", sep = '') 
df_train <- read.csv(file=input_file, header = FALSE)
#df_income <- read.csv(paste0(main_dir,"/data/elec_price/row_income_data_train.txt"),header = FALSE)
#df_gas <- read.csv(paste0(main_dir,"/data/elec_price/gas_data_train.txt"),header = FALSE)
# Defining output directory, input window size, forecasting horizon, and seasonality respectively.
output_dir = paste(main_dir,"/data/elec_price/moving_window/",sep = '')
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing

input_size = 12 
max_forecast_horizon <- 24
seasonality_period <- 12

for (validation in c(TRUE, FALSE)) {
  for (idr in 1 : nrow(df_train)) {
    print(idr)
    output_path = paste(output_dir, "priceMT", sep = '/')
    output_path = paste(output_path, input_size, sep = '_')
    output_path = paste(output_path, max_forecast_horizon, sep = '_')

    if (validation) {
      output_path = paste(output_path, 'val', sep = '_')
    }
    else {
      output_path = paste(output_path, 'train', sep = '_')
    }
    output_path = paste(output_path, 'txt', sep = '.')
    
    #income_ts <- as.numeric(df_income[idr,c(2:97)])
    #income_ts <- round(income_ts*1.1)
    #income_ts_mean <- mean(income_ts)
    #income_final_ts <- income_ts / (income_ts_mean)
    #aggregated_timeseries_income <- income_final_ts
    
    #gas_ts <- as.numeric(df_gas[idr,c(2:97)])
    #gas_ts_mean <- mean(gas_ts)
    #gas_final_ts <- gas_ts / (gas_ts_mean)
    #aggregated_timeseries_gas <- gas_final_ts
    
    time_series_data <- as.numeric(df_train[idr,c(2:97)])
    time_series_mean <- mean(time_series_data)
    time_series_data <- time_series_data / (time_series_mean)
    
    time_series_log <- log(time_series_data)
    time_series_length = length(time_series_log)
      
    if (! validation) {
      time_series_length = time_series_length - max_forecast_horizon
      time_series_log = time_series_log[1 : time_series_length]
    }
      # apply stl
    decomp_result = tryCatch({
      sstl = stl(ts(time_series_log, frequency = seasonality_period), s.window = "period")
      seasonal_vect = as.numeric(sstl$time.series[,1])
      levels_vect = as.numeric(sstl$time.series[,2])
      values_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3])
      cbind(seasonal_vect, levels_vect, values_vect)
      }, error = function(e) {
      seasonal_vect = rep(0, length(time_series_length))#stl() may fail, and then we would go on with the seasonality vector=0
      levels_vect = time_series_log
      values_vect = time_series_log
      cbind(seasonal_vect, levels_vect, values_vect)
      })
      
    # Generating input and output windows using the original time series.
    input_windows = embed(time_series_log[1:(time_series_length - max_forecast_horizon)], input_size)[, input_size:1]
    #exogenous_windows_income = embed(aggregated_timeseries_income[1:(time_series_length - max_forecast_horizon)], input_size)[, input_size:1]
    #exogenous_windows_gas = embed(aggregated_timeseries_gas[1:(time_series_length - max_forecast_horizon)], input_size)[, input_size:1]
    output_windows = embed(time_series_log[-(1:input_size)], max_forecast_horizon)[, max_forecast_horizon:1]
    # Generating seasonal components to use as exogenous variables.
    seasonality_windows = embed(decomp_result[1:(time_series_length - max_forecast_horizon), 1], input_size)[, input_size:1]
    seasonality_windows =  seasonality_windows[, c(12)] # c(value of input_size)
    # Generating the final window values.
    meanvalues <- rowMeans(input_windows)
    input_windows <- input_windows - meanvalues
    output_windows <- output_windows - meanvalues
    
    if (validation) {
      # Saving into a dataframe with the respective values.
      sav_df = matrix(
        NA,
        #ncol = ((3 * input_size) + 6 + max_forecast_horizon),
        ncol = (input_size + 6 + max_forecast_horizon),
        nrow = nrow(input_windows)
      )
      sav_df = as.data.frame(sav_df)
      sav_df[, 1] = paste(idr - 1, '|i', sep = '')
      sav_df[, 2] = seasonality_windows
      sav_df[, 3:(input_size + 2)] = input_windows
      sav_df[, (input_size + 2 + 1)] = '|o'
      sav_df[, (input_size + 2 + 1 + 1):(input_size + 2 + 1 + max_forecast_horizon)] = output_windows
      sav_df[, (input_size + max_forecast_horizon + 4)] = '|#'
      sav_df[, (input_size + max_forecast_horizon + 5)] = time_series_mean
      sav_df[, (input_size + max_forecast_horizon + 6)] = meanvalues
      #sav_df = as.data.frame(sav_df)
      #sav_df[, 1] = paste(idr - 1, '|i', sep = '')
      #sav_df[, 2:(input_size + 1)] = exogenous_windows_income
      #sav_df[, (input_size + 2):((input_size*2) + 1)] = exogenous_windows_gas
      #sav_df[, ((input_size*2) + 2)] = seasonality_windows
      #sav_df[, ((input_size*2) + 3):((input_size*3) + 2)] = input_windows
      #sav_df[, ((input_size*3) + 3)] = '|o'
      #sav_df[, ((input_size*3) + 4):((input_size*3) + 3 + max_forecast_horizon)] = output_windows
      #sav_df[, ((input_size*3) + max_forecast_horizon + 4)] = '|#'
      #sav_df[, ((input_size*3) + max_forecast_horizon + 5)] = time_series_mean
      #sav_df[, ((input_size*3) + max_forecast_horizon + 6)] = meanvalues
    } else {
      sav_df = matrix(
        NA,
        #ncol = ((3 * input_size) + 3 + max_forecast_horizon),
        ncol = (2 + input_size + 1 + max_forecast_horizon),
        nrow = nrow(input_windows)
      )
      sav_df = as.data.frame(sav_df)
      sav_df[, 1] = paste(idr - 1, '|i', sep = '')
      sav_df[, 2] = seasonality_windows
      sav_df[, 3:(input_size + 2)] = input_windows
      sav_df[, (input_size + 2 + 1)] = '|o'
      sav_df[, (input_size + 2 + 1 + 1):(input_size + 2 + 1 + max_forecast_horizon)] = output_windows
      #sav_df = as.data.frame(sav_df)
      #sav_df[, 1] = paste(idr - 1, '|i', sep = '')
      #sav_df[, 2:(input_size + 1)] = exogenous_windows_income
      #sav_df[, (input_size + 2):((input_size*2) + 1)] = exogenous_windows_income
      #sav_df[, ((input_size*2) + 2)] = seasonality_windows
      #sav_df[, ((input_size*2) + 3):((input_size*3) + 2)] = input_windows
      #sav_df[, ((input_size*3) + 3)] = '|o'
      #sav_df[, ((input_size*3) + 4):((input_size*3) + 3 + max_forecast_horizon)] = output_windows
    }
    
    #Writing the dataframe into a file. 
    write.table(
      sav_df,
      file = output_path,
      row.names = F,
      col.names = F,
      sep = " ",
      quote = F,
      append = TRUE
    )
  }
}
