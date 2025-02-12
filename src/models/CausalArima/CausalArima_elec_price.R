###Causal Arima Model for causal impact analysis###

devtools::install_github("FMenchetti/CausalArima")

library(CausalArima)

###Auxiliary functions for evaluation###

#MASE
mase_greybox <- function(holdout, forecast, scale) {
  # Check if the lengths of the holdout and forecast are the same
  if (length(holdout) != length(forecast)) {
    stop("The length of the provided data differs.")
  }
  
  # Calculate the Mean Absolute Scaled Error (MASE)
  return(mean(abs(holdout - forecast) / scale))
}

MASE <- function(preds, trues, df_a, seasonality_period) {
  mase_vector <- c()  # Initialize the vector for MASE values
  
  # Loop over the columns in the dataframe (representing different time series)
  for (i in 1:ncol(df_a)) {
    # Calculate the lagged differences
    lagged_diff <- sapply((seasonality_period + 1):nrow(df_a), function(j) {
      df_a[j, i] - df_a[j - seasonality_period, i]
    })
    
    # Call the mase_greybox function to calculate MASE for this series
    mase_value <- mase_greybox(trues, preds, mean(abs(lagged_diff)))
    mase_vector <- c(mase_vector, mase_value)  # Append the result
    mean_mase = mean(mase_vector)
  }
  
  # Return the mean MASE across all time series
  return(mean_mase)
}

#SMAPE
SMAPE <- function(pred, true) {
  smape_value <- mean(2 * abs(pred - true) / (abs(pred) + abs(true)), na.rm = TRUE)
  return(smape_value)
}

data_type <- 'elec_price'
dataset_name <- 'priceMT'
pred_len = 24
seasonality_period = 12
results_path = paste0(getwd(),'/results/',data_type,'/CausalArima/')


control <- c('AK', 'AL', 'AR', 'AZ', 'CO', 'DE', 'ID', 'FL', 'GA', 'HI', 'IA', 'IN', 'KS', 'KY', 'LA', 
              'ME', 'MN', 'MI', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NM', 'NV', 'OH', 'OK', 'OR',
              'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY')
data_folder = paste0(getwd(),'/data/',data_type,'/')
price_data_path = paste0(data_folder,dataset_name,'_full_table.csv')
income_data_path = paste0(data_folder,'income_data','_full_table.csv')
gas_data_path = paste0(data_folder,'gas_data','_full_table.csv')
      
# Read and rearrange data to right format
df_raw = read.csv((price_data_path))
df_income = read.csv(income_data_path)
df_gas = read.csv(gas_data_path)
dates <- seq.Date(from = as.Date("1990-01-01"), by = "months", length.out = nrow(df_raw))
int_date = seq.Date(as.Date("1990-01-01"),by="months",length.out = (nrow(df_raw)-pred_len+1))[(nrow(df_raw)-pred_len+1)]
preds = matrix(nrow=pred_len,ncol = 0)
      
for (i in 1:ncol(df_raw)){
  y = df_raw[,i]
  x = cbind(df_gas[,i],df_income[,i])
  ca = CausalArima(y = y, dates = dates, int.date = int_date)
  preds = cbind(preds, ca$forecast)
}
preds = data.frame(preds)
colnames(preds) <- colnames(df_raw)
      
preds_control <- as.matrix(preds[,colnames(preds) %in% control])
trues <- df_raw[(nrow(df_raw)-pred_len + 1):nrow(df_raw), ]
trues_control <- as.matrix(trues[,colnames(trues) %in% control])
      
df_A <- df_raw[1:(nrow(df_raw)-pred_len), ]
df_A_control <- as.matrix(df_A[,colnames(trues) %in% control])
      
mase_control = MASE(preds_control,trues_control,df_A_control,seasonality_period)
smape_control = SMAPE(preds_control,trues_control)
      
metrics_control = list(mase = mase_control,smape = smape_control)
      
forecasts_path = paste0(results_path,'/forecasts/')
metrics_path = paste0(results_path,'/metrics/')
      
if (!file.exists(forecasts_path)) {
  dir.create(forecasts_path, recursive = TRUE)
}
if (!file.exists(metrics_path)) {
  dir.create(metrics_path, recursive = TRUE)
}
      
write.csv(preds, paste0(forecasts_path,dataset_name,"_predictions.csv"), row.names = FALSE)
cat("mase ", metrics_control$mase, "\nsmape ", metrics_control$smape, 
    file = paste0(metrics_path,dataset_name,"_metrics_control.txt"), sep = "")




