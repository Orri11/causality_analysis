import argparse
import glob
import logging
import os
import time
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers


# TSMixer API===========================================================================
def drop_last_for_tensorflow(df, batch_size, seq_len, pred_len):
    """
    Helper function to emulate PyTorch dataloaders' option for drop_last = True.
    """
    total_length = len(df) - (seq_len + pred_len - 1)
    excess = total_length % batch_size
    if excess > 0:
        adjusted_length = len(df) - excess
        df = df.iloc[:adjusted_length]
    return df


# Metrics
def RSE(pred, true):
    """
    Calculates relative quared error.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    """
    Calculates correlation coefficient.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Calculates mean absolute error.
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Calculates mean squared error.
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Calculates root mean suared error.
    """
    return np.sqrt(MSE(pred, true))


def SMAPE(pred, true):
    """
    Calculates mean absolute percentage error.
    """
    return np.mean(np.abs(pred - true) / (abs(pred) + np.abs(true)) * 2)

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


def MASE(pred, true, df_a, seasonality_period):
    """
    Calculates mean squared percentage error.
    """
    mase_vector = []
    for i in range (len(df_a.columns)):
        lagged_diff = [df_a.iloc[j,i] - \
                       df_a.iloc[j,i - seasonality_period] for j in range(seasonality_period, len(df_a))]
        mase_vector.append(mase_greybox(true, pred, np.mean(np.abs(lagged_diff))))
    mean_mase = np.mean(mase_vector)
    return mean_mase

def metric(pred, true, df_a, seasonality_period):
    """
    Wraps up metric functions, calculates and returns all.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(pred, true)
    mase = MASE(pred, true, df_a, seasonality_period)

    return mae, mse, rmse, smape, mase


# Dot dictionary for holding args
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#  Data loader and dependencies
class TSFDataLoader:
    """Generate data loader from raw data."""

    def __init__(
        self,
        batch_size,
        seq_len,
        pred_len,
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'),
        data_name = 'priceMT',
        data_type = 'elec_price',
        features="M",
        target="TARGET",
        drop_last = False
    ):
        self.root_path = root_path
        self.data_name = data_name
        self.data_type = data_type
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.drop_last = False
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        if self.data_type == "elec_price":
            df = pd.read_csv(self.root_path + "data/" + self.data_type + "/" + self.data_name + "_full_table.csv")
        else:
            df = pd.read_csv(self.root_path + "data/" + self.data_type + "/" + self.data_name + ".csv")

        # S: univariate-univariate,
        # M: multivariate-multivariate,
        # MS: multivariate-univariate

        if self.features == "S":
            df = df[[self.target]]
        elif self.features == "MS":
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        # split train/valid/test
        n = len(df)
        test_end = n
        val_end = test_end-self.pred_len 
        train_end = val_end-self.pred_len

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]

        # NOTE drop last added for compatibility with Torch models======
        if self.drop_last == True:
            train_df = drop_last_for_tensorflow(
                train_df, self.batch_size, self.seq_len, self.pred_len
            )
            val_df = drop_last_for_tensorflow(
                val_df, self.batch_size, self.seq_len, self.pred_len
            )
            test_df = drop_last_for_tensorflow(
                test_df, self.batch_size, self.seq_len, self.pred_len
            )
        # ==============================================================

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def _split_window(self, data):
        inputs = data[:, : self.seq_len, :]
        labels = data[:, self.seq_len :, self.target_slice]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.seq_len, None])
        labels.set_shape([None, self.pred_len, None])
        return inputs, labels

    def _make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=(self.seq_len + self.pred_len),
            sequence_stride=1,  # window stride
            shuffle=shuffle,
            batch_size=self.batch_size,
        )
        ds = ds.map(self._split_window)
        return ds

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)


# Reversible Instance Normalization
class RevNorm(layers.Layer):
    """Reversible Instance Normalization."""

    def __init__(self, axis, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            self.affine_weight = self.add_weight(
                "affine_weight", shape=input_shape[-1], initializer="ones"
            )
            self.affine_bias = self.add_weight(
                "affine_bias", shape=input_shape[-1], initializer="zeros"
            )

    def call(self, x, mode, target_slice=None):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = tf.stop_gradient(tf.reduce_mean(x, axis=self.axis, keepdims=True))
        self.stdev = tf.stop_gradient(
            tf.sqrt(
                tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
            )
        )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / self.affine_weight[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


# TSMIxer Block
def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""

    # Normalization
    norm = layers.LayerNormalization if norm_type == "L" else layers.BatchNormalization

    # Temporal Linear MLP
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear MLP
    x = norm(axis=[-2, -1])(res)
    x = layers.Dense(ff_dim, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)

    return x + res


# Build TSMixer with Reversible Instance Normalization
def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice
):
    """Build TSMixer with Reversible Instance Normalization model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    rev_norm = RevNorm(axis=-2)
    x = rev_norm(x, "norm")
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)
    if target_slice:
        x = x[:, :, target_slice]
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
    outputs = rev_norm(outputs, "denorm", target_slice)

    return tf.keras.Model(inputs, outputs)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)


class TSMixer:
    def __init__(self):
        self.args = dotdict()
        self.args.seed = 100
        # Possible choices - ["S", "M", "MS]
        self.args.features = "M"
        self.args.target = "TARGET"
        self.args.checkpoints = "./checkpoints"
        self.args.delete_checkpoint = False
        ## Variables for TS
        self.args.seq_len = 12  #
        # Model Architecture
        # self.kernel_size = 4  # NOTE redundant, used for CNN model
        self.args.n_block = 2  # number of blocks for deep architecture
        self.args.ff_dim = 2048  # feed-forward dimension
        self.args.dropout = 0.05
        self.args.norm_type = "B"  # BatchNorm, L LayerNorm also possible
        self.args.activation = "relu"  # gelu also possible
        self.args.temporal_dim = 16  # temporal feature dimension
        self.args.hidden_dim = 64  # hidden feature dimension
        self.args.num_workers = 0
        self.args.itr = 3 # number of iterations

    def compile(self, learning_rate=1e-4, loss="mae", early_stopping_patience=3):
        self.args.loss = loss
        self.args.learning_rate = learning_rate
        self.args.patience = early_stopping_patience

    def fit(
        self,
        data_name = 'priceMT',
        data_type = 'elec_price',
        data_root_path= os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'),
        seasonality_period = 12,
        treatment_rate = None,
        batch_size=32,
        epochs=100,
        pred_len=24,
        seq_len=12,
        features="M",
        target="TARGET",
        iter=1,
    ):
        self.args.data_name = data_name  
        self.args.data_type = data_type
        self.args.root_path = data_root_path
        self.args.seasonality_period = seasonality_period
        self.args.treatment_rate = treatment_rate,
        self.args.pred_len = pred_len
        self.args.batch_size = batch_size  # 32 is the authors' default
        self.args.train_epochs = epochs  # 100 is the authors' default
        self.args.seq_len = seq_len
        self.args.iter = iter
        self.args.features = features
        self.args.target = target
        self.args.iter = iter  #how many experiment rounds

        print("Beginning to fit the model with the following arguments:")
        print(f"{self.args}")
        print("=" * 150)

        self.setting = f"TSMixer_{self.args.data_type}_{self.args.data_name}_{self.args.features}_sl{self.args.seq_len}_pl{self.args.pred_len}_iter{self.args.iter}"

        tf.random.set_seed(self.args.seed)

        # Initialize the data loader
        data_loader = TSFDataLoader(
            root_path=self.args.root_path,
            data_name=self.args.data_name,
            data_type=self.args.data_type,
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            features=self.args.features,
            target=self.args.target,
        )

        # Load train, val, test data
        self.train_data = data_loader.get_train()
        self.val_data = data_loader.get_val()
        self.test_data = data_loader.get_test()
        self.data_loader = data_loader

        # Build model
        model = build_model(
            input_shape=(self.args.seq_len, data_loader.n_feature),
            pred_len=self.args.pred_len,
            norm_type=self.args.norm_type,
            activation=self.args.activation,
            dropout=self.args.dropout,
            n_block=self.args.n_block,
            ff_dim=self.args.ff_dim,
            target_slice=data_loader.target_slice,
        )

        # Set up optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        # True compilation
        model.compile(optimizer=optimizer, loss=self.args.loss, metrics=self.args.loss)
        checkpoint_path = os.path.join(self.args.checkpoints, f"{self.setting}_best")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.args.patience
        )
        start_training_time = time.time()

        # Fit the model
        history = model.fit(
            self.train_data,
            epochs=self.args.train_epochs,
            validation_data=self.val_data,
            callbacks=[checkpoint_callback, early_stop_callback],
        )
        end_training_time = time.time()
        elasped_training_time = end_training_time - start_training_time
        print(f"Training finished in {elasped_training_time} secconds")

        # Evaluate best model on the val set
        # Load weights from the checkpoint
        best_epoch = np.argmin(history.history["val_loss"])
        model.load_weights(checkpoint_path)
        self.model = model  # Save as self to move on to .predict()

        # return self.model  # NOTE activate to return model

    def predict(self):
        if self.args.data_type =='sim':
            treated_units_indices_path = self.args.root_path + 'data/' + self.args.data_type + '/' + \
            self.args.data_name + '_treated_indices.txt'
            treated_units_indices = np.loadtxt(treated_units_indices_path, dtype=int)
        # Generate predictions
        prediction = self.model.predict(self.test_data, batch_size=self.args.batch_size)
        preds = self.data_loader.inverse_transform(prediction[0])
        self.preds = preds
        
        if self.args.data_type == "elec_price":
            names = pd.read_csv(self.args.root_path + "data/" + self.args.data_type + "/" + self.args.data_name + \
                                "_full_table.csv").iloc[:,1:].columns
            control = ['AK', 'AL', 'AR', 'AZ', 'CO', 'DE', 'ID', 'FL', 'GA', 'HI', 'IA', 'IN', 'KS', 'KY', 'LA', 
                       'ME', 'MN', 'MI', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NM', 'NV', 'OH', 'OK', 'OR',
                       'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
            preds_df = pd.DataFrame(preds)
            preds_df.columns = names
            preds_for_errors_control = np.array(preds_df.loc[:,control])
        else:
            names = np.int64(pd.read_csv(self.args.root_path + "data/" + self.args.data_type + "/" + self.args.data_name + \
                                '.csv').columns.tolist())
            preds_df = pd.DataFrame(preds)
            preds_df.columns = names
            control = np.setdiff1d(np.arange(0,preds_df.shape[1]), treated_units_indices)
            preds_for_errors_control = np.array(preds_df.loc[:,control])
            preds_for_errors_treated = np.array(preds_df.loc[:,~preds_df.columns.isin(control)])

        
        # Extract y_trues from DataLoader
        #trues_list = []
        #for _, targets in self.test_data:
            #trues_list.append(targets.numpy())
        #self.trues = np.concatenate(trues_list, axis=0)

        
        if self.args.data_type == "sim":
            df_raw = pd.read_csv(self.args.root_path + "data/" + self.args.data_type + "/" + self.args.data_name + "_" + \
                                  "true_counterfactual" + ".csv")
            df_raw.columns = np.int64(df_raw.columns.to_list())
            trues = df_raw.loc[len(df_raw)- self.args.pred_len:,:]
            trues_control = np.array(df_raw.loc[len(df_raw)- self.args.pred_len:,control])        
            df_a_control =  df_raw.loc[:len(df_raw)- self.args.pred_len,control]  

            trues_treated = np.array(df_raw.loc[len(df_raw)- self.args.pred_len:,~df_raw.columns.isin(control)])        
            df_a_treated =  df_raw.loc[:len(df_raw)- self.args.pred_len,~df_raw.columns.isin(control)]                     
        else:
            df_raw = pd.read_csv(self.args.root_path + "data/" + self.args.data_type + "/" + self.args.data_name + \
                                 "_full_table.csv").iloc[:,1:]
            trues = df_raw.loc[len(df_raw)- self.args.pred_len:,:]
            trues_control = np.array(df_control.iloc[len(df_control) - self.args.pred_len:,control])
            df_a_control = df_control.loc[:len(df_control)- self.args.pred_len,control]

            trues_treated = np.array(df_control.iloc[len(df_control) - self.args.pred_len:,~df_raw.columns.isin(control)])
            df_a_ = df_control.loc[:len(df_control)- self.args.pred_len,~df_raw.columns.isin(control)]

        self.trues_control = trues_control
        self.trues_treated = trues_treated

        if self.args.delete_checkpoint:
            for f in glob.glob(self.args.checkpoint_path + "*"):
                os.remove(f)

        # Save results
        metric_folder_path = self.args.root_path + "/results/" + self.args.data_type + "/" + "tsmixer/"  +  "metrics" + "/"
        data_folder_path = self.args.root_path + "/results/" + self.args.data_type + "/" + "tsmixer/"  +  "forecasts" + "/"
        if not os.path.exists(metric_folder_path):
            os.makedirs(metric_folder_path)
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)


        mae_control, mse_control, rmse_control, smape_control, mase_control = metric( 
        preds_for_errors_control, trues_control ,df_a_control, self.args.seasonality_period)
        all_metrics_control = [mae_control, mse_control, rmse_control, smape_control, mase_control]
        metric_list = ['mae', 'mse', 'rmse', 'smape', 'mase']
        
        metric_df_control = pd.DataFrame([all_metrics_control], columns=metric_list)

        if self.args.dataset_type =='sim':
            mae_treated, mse_treated, rmse_treated, smape_treated, mase_treated = metric( 
            preds_for_errors_treated, trues_treated, df_a_treated, self.args.seasonality_period)
            all_metrics_treated = [mae_treated, mse_treated, rmse_treated, smape_treated, mase_treate]

            metric_df_treated = pd.DataFrame([all_metrics_treated], columns=metric_list)
       

        metric_df_control.to_csv(metric_folder_path + self.setting + "_" + "metrics_control.csv", index = False)
        if self.args.dataset_type =='sim':
            metric_df_treated.to_csv(metric_folder_path + self.setting + "_" + "metrics_treated.csv", index = False)
        preds_df.to_csv(data_folder_path + self.setting + "_" + "preds.csv", index = False)
        trues.to_csv(data_folder_path + self.setting + "_" + "trues.csv", index = False)

        return preds

        
