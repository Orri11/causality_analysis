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

data_type <- 'sim'
lengths = c(90,420)
num_series = c(50,300)
structures = c('stationary','trend')
int_type = 'hom'
pred_len = 24
seasonality_period = 30
results_path = paste0(getwd(),'/results/',data_type,'/CausalArima/')

for (len in lengths){
  for (series in num_series){
    for (struc in structures){
      dataset_name <- paste('sim',len,series,struc,'hom',sep = '_')
      treated_indices <- readLines(paste0(getwd(),'/data/',data_type,'/',dataset_name,
                                          '_treated_indices','.txt'))
      treated_indices <- as.integer(treated_indices) +1
      data_folder = paste0(getwd(),'/data/',data_type,'/',dataset_name)
      data_path = paste0(data_folder,'.csv')
      
      # Read and rearrange data to right format
      df_raw = read.csv((data_path))
      colnames(df_raw) <- as.integer(sub("X","",colnames(df_raw)))
      colnames(df_raw) <- seq(1,series,length.out=series)
      dates <- seq.Date(from = as.Date("2000-01-01"), by = "days", length.out = len)
      int_date = seq.Date(as.Date("2000-01-01"),by="day",length.out = (len-pred_len+1))[(len-pred_len+1)]
      preds = matrix(nrow=pred_len,ncol = 0)
      
      for (i in 1:ncol(df_raw)){
        y = df_raw[,i]
        ca = CausalArima(y = y, dates = dates, int.date = int_date)
        preds = cbind(preds, ca$forecast)
      }
      preds = data.frame(preds)
      colnames(preds) <- as.integer(sub("X","",colnames(preds)))
      
      preds_treated <- as.matrix(preds[,colnames(preds) %in% treated_indices])
      preds_control <- as.matrix(preds[,!colnames(preds) %in% treated_indices])
      counterfactuals_path = paste0(data_folder,'_true_counterfactual.csv')
      counterfactuals = read.csv((counterfactuals_path))
      colnames(counterfactuals) <- sub("X","",colnames(counterfactuals))
      colnames(counterfactuals) <- as.numeric(colnames(counterfactuals)) + 1
      
      trues <- counterfactuals[(len-pred_len + 1):len, ]
      trues_treated <- as.matrix(trues[,colnames(trues) %in% treated_indices])
      trues_control <- as.matrix(trues[,!colnames(trues) %in% treated_indices])
      
      df_A <- counterfactuals[1:(len-pred_len), ]
      df_A_treated <- as.matrix(df_A[,colnames(trues) %in% treated_indices])
      df_A_control <- as.matrix(df_A[,!colnames(trues) %in% treated_indices])
      
      mase_control = MASE(preds_control,trues_control,df_A_control,seasonality_period)
      smape_control = SMAPE(preds_control,trues_control)
      
      mase_treated = MASE(preds_treated,trues_treated,df_A_treated,seasonality_period)
      smape_treated = SMAPE(preds_treated,trues_treated)
      
      metrics_control = list(mase = mase_control,smape = smape_control)
      metrics_treated = list(mase = mase_treated,smape = smape_treated)
      
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
      cat("mase ", metrics_treated$mase, "\nsmape ", metrics_treated$smape, 
          file = paste0(metrics_path,dataset_name,"_metrics_treated.txt"), sep = "")
    }
  }
}
      

