import os
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.metrics import mae, mape, mse
from darts.models import RandomForest
from darts.models import LinearRegressionModel
from darts.models import LightGBMModel
from darts.models import BlockRNNModel
from darts.models import TCNModel, TFTModel

LEN = 1200  # max example num for each time series
HIS = 120  # length to look back on each chuck
AHEAD = 60  # length to predict on each chuck

if __name__ == '__main__':

    #  Building training set
    
    tar_list = []
    cov_df_list = []  # TimeSeries 변환 전, DataFrame 상태로 임시 보관
    switch = 0
    for path, dir_list, file_list in os.walk(r"./train"):
        for file_name in file_list:
            if file_name == '-.csv':
                switch = 0
            if switch == 0:
                print('Start processing file ' + file_name)
                file = os.path.join(path, file_name)
                df = pd.read_csv(file, index_col='time', low_memory=False)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.resample('S').asfreq()
                df = df.apply(pd.to_numeric, errors='ignore')
                df[df['c16'] > 5] = np.nan
                df[df['c16'] <= 0] = np.nan
                df = pd.get_dummies(df.drop(['error'], axis=1), columns=['c10'])
                df.fillna(method='ffill', inplace=True)
                data_tar = df.iloc[:, :5]
                data_cov = df.iloc[:, 5:]
                series_tar = TimeSeries.from_dataframe(data_tar, freq='S').astype(np.float32)
                tar_list.append(series_tar)
                cov_df_list.append(data_cov)  # 아직 컬럼 수가 파일마다 다를 수 있음
                print('Finish processing file ' + file_name)

    # ---- 모든 파일에서 나온 c10 더미 컬럼을 하나로 통일 ----
    all_cov_columns = sorted(set().union(*[d.columns for d in cov_df_list]))

    cov_list = []
    for data_cov in cov_df_list:
        data_cov = data_cov.reindex(columns=all_cov_columns, fill_value=0)
        series_cov = TimeSeries.from_dataframe(data_cov, freq='S').astype(np.float32)
        cov_list.append(series_cov)

    # Building model

    model_Regression = RegressionModel(
        lags=HIS,
        lags_past_covariates=HIS,
        output_chunk_length=AHEAD,
        model=BayesianRidge()
    )

    model_RandomForest = RandomForest(
        lags=HIS,
        lags_past_covariates=HIS,
        output_chunk_length=AHEAD,
        n_estimators=100,
        max_depth=None

    )

    model_LinearRegressionModel = LinearRegressionModel(
        lags=HIS,
        lags_past_covariates=HIS,
        output_chunk_length=AHEAD,
        random_state=43

    )

    model_LightGBMModel = LightGBMModel(
        lags=HIS,
        lags_past_covariates=HIS,
        output_chunk_length=AHEAD,
        random_state=43,
    )

    model_RNN = BlockRNNModel(
        input_chunk_length=HIS,
        output_chunk_length=AHEAD,
        model='RNN',  # “LSTM” or “GRU”
        random_state=42,
        hidden_dim=64,
        n_rnn_layers=2,
        dropout=0.2,
        pl_trainer_kwargs={"accelerator": "gpu"}
    )

    model_GRU = BlockRNNModel(
        input_chunk_length=HIS,
        output_chunk_length=AHEAD,
        model='GRU',  # “LSTM” or “GRU”
        random_state=42,
        hidden_dim=64,
        n_rnn_layers=2,
        dropout=0.2,
        pl_trainer_kwargs={"accelerator": "gpu"}

    )

    model_LSTM = BlockRNNModel(
        input_chunk_length=HIS,
        output_chunk_length=AHEAD,
        model='LSTM',  # “LSTM” or “GRU”
        random_state=42,
        hidden_dim=64,
        n_rnn_layers=2,
        dropout=0.2,
        pl_trainer_kwargs={"accelerator": "gpu"}

    )

    model_TCNModel = TCNModel(
        input_chunk_length=HIS,
        output_chunk_length=AHEAD,
        kernel_size=3,
        num_filters=3,
        dilation_base=2,
        weight_norm=False,
        dropout=0.2,
        pl_trainer_kwargs={"accelerator": "gpu"}
    )

    model_TFTModel = TFTModel(
        input_chunk_length=HIS,
        output_chunk_length=AHEAD,
        pl_trainer_kwargs={"accelerator": "gpu"}

    )

    model_NBEATS = NBEATSModel(
        input_chunk_length=HIS,
        output_chunk_length=AHEAD,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "gpu"}

    )

    #  Training model

    model_Regression.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN)
    model_RandomForest.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN)
    model_LinearRegressionModel.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN)
    model_LightGBMModel.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN)
    model_RNN.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, epochs=100, verbose=True)
    model_GRU.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, epochs=100, verbose=True)
    model_LSTM.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, epochs=100, verbose=True)
    model_TCNModel.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, epochs=100, verbose=True)
    model_TFTModel.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, epochs=100, verbose=True)
    model_NBEATS.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, epochs=100, verbose=True)

    #  Testing

    switch = 0  # =1 when something case stop the training set building in middle
    for path, dir_list, file_list in os.walk(r"./test"):
        for file_name in file_list:
            if file_name == '-.csv':  # Mark
                switch = 0
                
            if switch == 0:
                # pre-process test data
                print('Start processing test file '+file_name)
                file = os.path.join(path, file_name)
                df = pd.read_csv(file, index_col='time', low_memory=False)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.resample('S').asfreq()
                df = df.apply(pd.to_numeric, errors='ignore')
                df[df['c16'] > 5] = np.nan  # outlier data
                df[df['c16'] <= 0] = np.nan
                df = pd.get_dummies(df.drop(['error'], axis=1), columns=['c10'])
                df.fillna(method='ffill', inplace=True)  # gap imputation
                data_tar = df.iloc[:, :5]  # target time series
                data_cov = df.iloc[:, 5:]
                data_cov = data_cov.reindex(columns=all_cov_columns, fill_value=0)  # train에서 구한 리스트 재사용
                series_tar = TimeSeries.from_dataframe(data_tar, freq='S').astype(np.float32)
                series_cov = TimeSeries.from_dataframe(data_cov, freq='S').astype(np.float32)

                # test

                pred_Regression = model_Regression.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./BayesianRidge' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['BayesianRidge'],
                     'mae': [mae(series_tar, pred_Regression)],
                     'mape': [mape(series_tar, pred_Regression)],
                     'mse': [mse(series_tar, pred_Regression)]}
                ).to_csv('./metric of ' + file_name, mode='a')  # metric

                predl_RandomForest = model_RandomForest.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./RandomForest' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['RandomForest'],
                     'mae': [mae(series_tar, predl_RandomForest)],
                     'mape': [mape(series_tar, predl_RandomForest)],
                     'mse': [mse(series_tar, predl_RandomForest)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_LinearRegressionModel = model_LinearRegressionModel.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./LinearRegression' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['LinearRegression'],
                     'mae': [mae(series_tar, predl_LinearRegressionModel)],
                     'mape': [mape(series_tar, predl_LinearRegressionModel)],
                     'mse': [mse(series_tar, predl_LinearRegressionModel)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_LightGBMModel = model_LightGBMModel.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./LightGB' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['LightGB'],
                     'mae': [mae(series_tar, predl_LightGBMModel)],
                     'mape': [mape(series_tar, predl_LightGBMModel)],
                     'mse': [mse(series_tar, predl_LightGBMModel)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_RNN = model_RNN.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./RNN' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['RNN'],
                     'mae': [mae(series_tar, predl_RNN)],
                     'mape': [mape(series_tar, predl_RNN)],
                     'mse': [mse(series_tar, predl_RNN)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_GRU = model_GRU.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./GRU' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['GRU'],
                     'mae': [mae(series_tar, predl_GRU)],
                     'mape': [mape(series_tar, predl_GRU)],
                     'mse': [mse(series_tar, predl_GRU)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_LSTM = model_LSTM.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./LSTM' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['LSTM'],
                     'mae': [mae(series_tar, predl_LSTM)],
                     'mape': [mape(series_tar, predl_LSTM)],
                     'mse': [mse(series_tar, predl_LSTM)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_TCNModel = model_TCNModel.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./TCN' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['TCN'],
                     'mae': [mae(series_tar, predl_TCNModel)],
                     'mape': [mape(series_tar, predl_TCNModel)],
                     'mse': [mse(series_tar, predl_TCNModel)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_TFTModel = model_TFTModel.predict(series=series_tar, n=120, past_covariates=series_cov)
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./TFT' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['TFT'],
                     'mae': [mae(series_tar, predl_TFTModel)],
                     'mape': [mape(series_tar, predl_TFTModel)],
                     'mse': [mse(series_tar, predl_TFTModel)]}
                ).to_csv('./metric of ' + file_name, mode='a', header=None)  # metric

                predl_NBEATS = model_NBEATS.predict(series=series_tar, n=120, past_covariates=series_cov)  # predict
                plt.figure(figsize=(35, 10))  # plot
                data_tar.plot(label="actual")
                predl_NBEATS.plot(label="forecast")
                plt.legend()
                plt.savefig('./NBEATS' + file_name[-4:] + '.png'
                            # , format='pdf'
                            , bbox_inches='tight'
                            , pad_inches=0.1, dpi=500)
                pd.DataFrame(
                    {'file_name': ['NBEATS'],
                     'mae': [mae(series_tar, predl_NBEATS)],
                     'mape': [mape(series_tar, predl_NBEATS)],
                     'mse': [mse(series_tar, predl_NBEATS)]}
                ).to_csv('./metric of '+file_name, mode='a', header=None)  # metric

                print('Finish testing file ' + file_name)