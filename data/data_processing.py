import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from pyts.image import GADF, GASF, MTF
from pykalman import KalmanFilter

from utils import load_pickle, log, dump_pickle, remove_all_files_from_dir
from config.hyperparams import DEFAULT_FILES_NAMES, DEFAULT_END_DATE, DEFAULT_START_DATE


class DataHandler:
    def __init__(self, encoding_method='GADF', window_len=64, image_size=16, retrain_freq=5,
                 start_date: int = DEFAULT_START_DATE, targets_methods=['close'],
                 end_date: int = DEFAULT_END_DATE, frac_of_stocks=1., stock_data_dir_path: str = 'data',
                 dir_for_samples='data/cnn_samples/regular', nb_of_stocks_by_file=50, kwargs_target_methods=None
                 ):

        self._window_len = window_len
        self._image_size = image_size
        self._retrain_freq = retrain_freq
        self._encoding_method = encoding_method
        self._targets_methods = targets_methods
        # self._features = ['']
        self._kwargs_target_methods = kwargs_target_methods or {}

        self._start_date = start_date
        self._end_date = end_date
        self._frac_of_stocks_to_get = frac_of_stocks
        self._nb_of_stocks_by_file = nb_of_stocks_by_file

        self._directory_for_samples = dir_for_samples
        self._stock_data_dir_path = stock_data_dir_path

        self._N_FILES_CRSP = 27
        self._LOGGER_ENV = 'image_encoding'

        self.df_data = None
        self._stocks_list = None

    def get_df_data(self):
        """
        identifies how many files it must read (randomly)
        Reads all the files if self._frac_of_stocks_to_get is 1.

        :instanciates:
            * self.df_data: dataframe of all data with dates as index
            * sekf._stocks_list: all the uniqu epermno present in the data

        :return: Nothing
        """
        nb_files_to_get = max(round(self._frac_of_stocks_to_get * self._N_FILES_CRSP), 1)
        choices = np.random.choice(np.arange(1, self._N_FILES_CRSP + 1), nb_files_to_get, replace=False)
        file_names = ['stockData_{}'.format(i) for i in choices]
        df_data = self._load_stock_data(file_names, data_dir_path=self._stock_data_dir_path,
                                        logger_env=self._LOGGER_ENV)
        df_data = self._get_data_from_stocks(df_data, ['10026'])  # todo

        self._stocks_list = np.unique(df_data.index)
        log('Data finalized in attribute df_data, number of stocks {}'.format(len(self._stocks_list)),
            environment=self._LOGGER_ENV)
        self.df_data = self._get_data_between_and_sort(df_data, self._start_date, self._end_date, self._LOGGER_ENV)

    def build_and_dump_images_and_targets(self, use_smoothed_data=False):
        """
             * Selects only the data we want (dates and stocks)
             * Builds images with the timeseries
             * Builds targets with the specified methods
             * Pickles dictionnaries on disk with keys/values :
                - batch_name: str
                - n_samples: int
                - samples: (numpy array)
                - VWAP_targets (for VWAP if it is one of the specified methods)
                - ..._targets if ther is other targets
                - df_original_data

        Builds image data
        :param use_smoothed_data: id we use KalmanFilter # todo think about FFT and wavelet

        """
        nb_stocks = len(self._stocks_list)
        n_files_to_dump = nb_stocks // self._nb_of_stocks_by_file + ((nb_stocks % self._nb_of_stocks_by_file) != 0)
        df_data_multi_index = self.df_data.reset_index(drop=False).set_index(['PERMNO', 'date'])
        log('***** Dumping data in {} different files'.format(n_files_to_dump), environment=self._LOGGER_ENV)

        # Removing existing files in the folder
        remove_all_files_from_dir(self._directory_for_samples, logger_env=self._LOGGER_ENV)

        for batch in range(n_files_to_dump):
            batch_name = 'image_data_{}'.format(batch + 1)
            batch_stocks = self._stocks_list[
                           batch * self._nb_of_stocks_by_file:(batch + 1) * self._nb_of_stocks_by_file]
            df_batch_data = self._extract_data_for_stocks(df_data_multi_index, batch_stocks)
            # build images and targets
            df_res = self._build_images_one_batch(df_batch_data, batch_name, use_smoothed_data=use_smoothed_data)
            # SORT BY DATES
            df_res = df_res.set_index('date').sort_index()
            self.test = df_res
            # Dumping the pickle dataframe
            dump_pickle(df_res, os.path.join(self._directory_for_samples, batch_name),     logger_env=self._LOGGER_ENV)

        # self.__delete_df_data_from_memory()

    def show_image(self, df_window_data):

        data = df_window_data.reset_index().set_index('date').drop('PERMNO', axis=1).T
        if self._encoding_method == 'GADF':
            gadf = GADF(self._image_size)
            image_data = (gadf.fit_transform(data).T)

        elif self._encoding_method == 'GASF':
            gasf = GASF(self._image_size)
            image_data = (gasf.fit_transform(data).T)
        elif self._encoding_method == 'MTF':
            gasf = MTF(self._image_size)
            image_data = (gasf.fit_transform(data).T)
        else:
            raise BaseException('Method must be either GADF, GASF or MTF not {}'.format(self._encoding_method))

        plt.imshow(image_data, cmap='rainbow', origin='lower')
        plt.title(self._encoding_method, fontsize=16)

    # SHOULD NOT BE USED ANYMORE
    def _get_targets_one_batch(self, df_batch_data: pd.DataFrame, batch_name: str):
        pass
        # dict_targets = {}
        # log('***** Building Targets for batch {}, methods will be {}'.format(batch_name, self._targets_methods),
        #     environment=self._LOGGER_ENV)
        # assert len(self._targets_methods) >= 1, 'The number of methods specified must be >=1'
        #
        # for method in self._targets_methods:
        #     log('Target building method {}'.format(method), environment=self._LOGGER_ENV)
        #     if method == 'VWAP':
        #         # TODO put the up and down returns as params
        #         labels_array, df_for_backtest = self._build_VWAP_returns(df_batch_data, self._window_len,
        #                                                                  self._retrain_freq,
        #                                                                  **self._kwargs_target_methods)
        #         dict_targets.update({'VWAP_targets': labels_array, 'df_for_backtest': df_for_backtest})
        #     elif method == 'close':
        #         labels_array, df_for_backtest = self._build_close_returns(df_batch_data, self._window_len,
        #                                                                   self._retrain_freq,
        #                                                                   **self._kwargs_target_methods)
        #         dict_targets.update({'close_targets': labels_array, 'df_for_backtest': df_for_backtest})
        #
        #     else:
        #         raise BaseException('So far the targets can only be computed by close prices or VWAP')
        # log('Targets for batch {} are built'.format(batch_name), environment=self._LOGGER_ENV)
        # return dict_targets

    @staticmethod
    def _build_close_returns(df, window_len=64, retrain_freq=5, up_return=0.0125, down_return=-0.0125,
                             buy_on_last_date=True):
        data = df[['PRC']]
        n_sample = len(data)
        targets = []

        _long = [1, 0, 0]
        _hold = [0, 1, 0]
        _short = [0, 0, 1]
        rebalance_indexes = []
        for i in range(window_len, n_sample, retrain_freq):
            j = i - 1 if buy_on_last_date else i

            lastprice_ = data.PRC.iloc[j]
            nextprice_ = data.PRC.iloc[np.min([n_sample - 1, i - 1 + retrain_freq])]

            price_return = (nextprice_ - lastprice_) / lastprice_

            if price_return > up_return:
                targets.append(_long)
            elif price_return < down_return:
                targets.append(_short)
            else:
                targets.append(_hold)

            # we keep the indexes of the dates when there will be a rebalance in the portfolio
            rebalance_indexes.append(j)
        df_for_backtest = df.iloc[rebalance_indexes]

        return np.asarray(targets), df_for_backtest

    @staticmethod
    def _build_VWAP_returns(df, window_len=64, retrain_freq=5, up_return=0.0125, down_return=-0.0125):
        data = df[['PRC', 'VOL']]
        n_sample = len(data)
        targets = []

        _long = [1, 0, 0]
        _hold = [0, 1, 0]
        _short = [0, 0, 1]
        rebalance_indexes = []

        for i in range(window_len, n_sample, retrain_freq):

            if data.VOL.iloc[i - retrain_freq:i].values.sum() > 0:
                lastVWAP = np.average(data.PRC.iloc[i - retrain_freq:i].values,
                                      weights=data.VOL.iloc[i - retrain_freq:i].values)
            else:
                lastVWAP = data.PRC.iloc[i]

            if data.VOL.iloc[i:np.min([n_sample - 1, i + retrain_freq])].values.sum() > 0:
                nextVWAP = np.average(data.PRC.iloc[i:np.min([n_sample - 1, i + retrain_freq])].values,
                                      weights=data.VOL.iloc[i:np.min([n_sample - 1, i + retrain_freq])].values)
            else:
                nextVWAP = data.PRC.iloc[np.min([n_sample - 1, i + retrain_freq])]

            VWAPReturn = (nextVWAP - lastVWAP) / lastVWAP

            if VWAPReturn > up_return:
                targets.append(_long)
            elif VWAPReturn < down_return:
                targets.append(_short)
            else:
                targets.append(_hold)

            # we keep the indexes of the dates when there will be a rebalance in the portfolio
            rebalance_indexes.append(i - 1)
        df_for_backtest = df.iloc[rebalance_indexes]

        return np.asarray(targets), df_for_backtest

    @staticmethod
    def _build_images_one_stock(df_one_permno, window_len, retrain_freq, encoding_method, image_size,
                                use_smoothed_data=False):

        n_days = df_one_permno.T.shape[-1]
        samples_list, dates_list, prc_list = [], [], []
        for i in range(window_len, n_days, retrain_freq):
            window_data = df_one_permno.T.iloc[:, i - window_len:i]
            date = df_one_permno.index[i]
            dates_list.append(date)
            prc_list.append(df_one_permno.loc[date, 'PRC'])

            # todo understand how this works
            if use_smoothed_data:
                Smoother = KalmanFilter(n_dim_obs=window_data.shape[0], n_dim_state=window_data.shape[0],
                                        em_vars=['transition_matrices', 'observation_matrices',
                                                 'transition_offsets', 'observation_offsets',
                                                 'transition_covariance', 'observation_convariance',
                                                 'initial_state_mean', 'initial_state_covariance'])
                measurements = window_data.T.values
                Smoother.em(measurements, n_iter=5)
                window_data, _ = Smoother.smooth(measurements)
                window_data = window_data.T

            if encoding_method == 'GADF':
                gadf = GADF(image_size)
                samples_list.append(gadf.fit_transform(window_data).T)

            elif encoding_method == 'GASF':
                gasf = GASF(image_size)
                samples_list.append(gasf.fit_transform(window_data).T)
            elif encoding_method == 'MTF':
                mtf = MTF(image_size)
                samples_list.append(mtf.fit_transform(window_data).T)
            else:
                raise BaseException('Method must be either GADF, GASF or MTF not {}'.format(encoding_method))
        samples_list = np.asarray(samples_list)
        dates_list = np.asarray(dates_list)
        prc_list = np.asarray(prc_list)
        return samples_list, dates_list, prc_list

    def _build_images_one_batch(self, df_batch_data, batch_name, use_smoothed_data=False):
        log('Building Targets and Images for batch {}'.format(batch_name), environment=self._LOGGER_ENV)
        log('Targets will be constructed with methods: {}'.format(self._targets_methods), environment=self._LOGGER_ENV)

        df_batch_data = df_batch_data.reset_index(drop=False).set_index(['PERMNO', 'date'])
        all_permnos = df_batch_data.index.levels[0]
        # The empty dataframe initialized
        columns_df_res = ['sample', 'date', 'PRC'] + self._targets_methods
        df_res = pd.DataFrame(columns=columns_df_res)

        for permno in all_permnos:
            df_one_permno = df_batch_data.loc[permno]
            samples_list, dates_list, prc_list = self._build_images_one_stock(df_one_permno, self._window_len,
                                                                              self._retrain_freq, self._encoding_method,
                                                                              self._image_size, use_smoothed_data=use_smoothed_data)

            for method in self._targets_methods:
                targets_list = []
                if method == 'VWAP':
                    labels_array, df_for_backtest = self._build_VWAP_returns(df_one_permno, self._window_len,
                                                                             self._retrain_freq,
                                                                             **self._kwargs_target_methods)

                elif method == 'close':
                    labels_array, df_for_backtest = self._build_close_returns(df_one_permno, self._window_len,
                                                                              self._retrain_freq,
                                                                              **self._kwargs_target_methods)

                else:
                    raise BaseException('So far the targets can only be computed by close prices or VWAP')
                targets_list.append(labels_array)

            # building dataframe
            df_res_one_permno = pd.DataFrame(columns=columns_df_res)

            for k, date in enumerate(dates_list):
                data = [samples_list[k], date, prc_list[k], *[labels_array[k] for labels_array in targets_list]]
                row_df = pd.DataFrame(columns=columns_df_res, data=[data])
                df_res_one_permno = pd.concat([df_res_one_permno, row_df])
            df_res_one_permno['PERMNO'] = permno
            df_res = pd.concat([df_res, df_res_one_permno])

        log('Targets and Images for batch {} are built'.format(batch_name), environment=self._LOGGER_ENV)

        return df_res


    @staticmethod
    def _get_data_from_stocks(df: pd.DataFrame, stocks_list: list):
        # TODO
        # raise NotImplementedError('Not yet implemented')
        return df

    # TODO this function should build a df with the new features we want
    def _get_features(self):
        # shoudl call at least _get_high_low_feature and the functions that gets returns
        pass

    # TODO this function should take a df with BIDLO and ASKHI and return a df with
    # todo : (PRC - BIDLO)/PRC and (ASKHI - PRC) / PRC
    @staticmethod
    def _get_high_low_feature():
        pass

    def __delete_df_data_from_memory(self):
        self.df_data = None

    @staticmethod
    def _get_data_between_and_sort(df: pd.DataFrame, start_date: int = DEFAULT_START_DATE,
                                   end_date: int = DEFAULT_END_DATE,
                                   logger_env='image_encoding'):
        """

        :param df: dataframe with PERMNO as index, date must be a column in the dataframe
        :param start_date: int Example:
        :param end_date:
        :return:
        """
        log('Getting data between {} and {}'.format(start_date, end_date), environment=logger_env)
        assert 'date' in df.columns, 'date must be one of the columns but columns are {}'.format(df.columns)
        df_res = df[df.date >= start_date]
        df_res = df_res[df_res.date <= end_date]

        sorted_dates = np.sort(df_res.date.values)
        new_start_date, new_end_date = sorted_dates[0], sorted_dates[-1]
        log('New boundary dates: {} and {}'.format(new_start_date, new_end_date), environment=logger_env)

        return df_res

    @staticmethod
    def _load_stock_data(file_names: list = DEFAULT_FILES_NAMES, data_dir_path: str = 'data',
                         logger_env: str = 'Pickling'):
        assert len(file_names) >= 1, 'the list of file names is <1'
        for i, file_name in enumerate(file_names):
            if i == 0:
                df_res = load_pickle(os.path.join(data_dir_path, file_name), logger_env=logger_env)
                assert isinstance(df_res, pd.DataFrame), 'the data from {} is not a DataFrame but {}'.format(file_name,
                                                                                                             df_res.__class__)
            else:
                df_temp = load_pickle(os.path.join(data_dir_path, file_name), logger_env=logger_env)
                assert isinstance(df_temp, pd.DataFrame), 'the data from {} is not a DataFrame but {}'.format(file_name,
                                                                                                              df_temp.__class__)
                df_res = pd.concat([df_res, df_temp])

        # df_res = df_res.reset_index(drop=False).set_index(['PERMNO', 'date'])

        return df_res

    @staticmethod
    def _extract_data_for_stocks(df_data_multi_ind: pd.DataFrame, list_stocks: list):
        df_res = df_data_multi_ind.loc[list_stocks]
        df_res = df_res.reset_index(level=0, drop=False)
        return df_res


def get_training_data_from_path(samples_path='data/cnn_samples/regular',
                                targets_type='VWAP_targets',
                                train_val_size=2 / 3.,
                                train_size=0.75,
                                logger_env='Training'
                                ):
    """
    :param samples_path: path for the folder with the data we need: only the files we need need to be in this folder
    :param targets_type: str: the targets to consider for this training
    :param train_val_size: training/(training+validation)
    :param train_size: size of (training+validation)/(training+test+validation)
    :return: X_train, X_val, X_test, Y_train, Y_val, Y_test as numpy arrays
    """
    # list of files in the folder: samples_paths
    list_file_names = os.listdir(samples_path)
    assert len(list_file_names) >= 1, 'The number of files in the folder {} is probably 0, it must be >=1'.format(
        samples_path)
    log('******* Getting data from folder: {}, Nb of files : {}, First file : {}'.format(samples_path,
                                                                                         len(list_file_names),
                                                                                         list_file_names[0]),
        logger_env)
    for i, file_name in enumerate(list_file_names):
        path = os.path.join(samples_path, file_name)
        df_all_data = load_pickle(path, logger_env=logger_env)

        log('first_date: {}, last_date: {}'.format(df_all_data.index[0], df_all_data.index[-1]),
            environment=logger_env)
        # in each of the pickle files the data is sorted in chronologic order
        X = np.concatenate([[sample for sample in df_all_data['sample'].values]], axis=0)  # need this to get an array
        Y = np.concatenate([[sample for sample in df_all_data[targets_type].values]], axis=0)
        n_samples = Y.shape[0]

        stop_1 = round(train_val_size * n_samples * train_size)
        stop_2 = round(n_samples * train_size)
        if i == 0:
            X_train = X[:stop_1]
            Y_train = Y[:stop_1]

            X_val = X[stop_1:stop_2]
            Y_val = Y[stop_1:stop_2]

            X_test = X[stop_2:]
            Y_test = Y[stop_2:]
        else:

            X_train = np.concatenate([X_train, X[:stop_1]])
            Y_train = np.concatenate([Y_train, Y[:stop_1]])

            X_val = np.concatenate([X_val, X[stop_1:stop_2]])
            Y_val = np.concatenate([Y_val, Y[stop_1:stop_2]])

            X_test = np.concatenate([X_test, X[stop_2:]])
            Y_test = np.concatenate([Y_test, Y[stop_2:]])

    # Distributions of Labels
    train_d = np.sum(Y_train, axis=0) / len(Y_train)
    val_d = np.sum(Y_val, axis=0) / len(Y_val)
    tst_d = np.sum(Y_test, axis=0) / len(Y_test)

    text_template = 'long: {:5.2f}%, hold: {:5.2f}%, short: {:5.2f}%'
    log('Training Distribution of Labels :' + text_template.format(train_d[0] * 100, train_d[1] * 100,
                                                                   train_d[2] * 100),
        environment=logger_env)
    log('Validation Distribution of Labels :' + text_template.format(val_d[0] * 100, val_d[1] * 100, val_d[2] * 100),
        environment=logger_env)
    log('Test Distribution of Labels :' + text_template.format(tst_d[0] * 100, tst_d[1] * 100, tst_d[2] * 100),
        environment=logger_env)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


# def get_training_data_from_path(samples_path='data/cnn_samples/regular',
#                                 targets_type='VWAP_targets',
#                                 train_val_size=2 / 3.,
#                                 train_size=0.75,
#                                 logger_env='Training'
#                                 ):
#     """
#     :param samples_path: path for the folder with the data we need: only the files we need need to be in this folder
#     :param targets_type: str: the targets to consider for this training
#     :param train_val_size: training/(training+validation)
#     :param train_size: size of (training+validation)/(training+test+validation)
#     :return: X_train, X_val, X_test, Y_train, Y_val, Y_test as numpy arrays
#     """
#     # list of files in the folder: samples_paths
#     list_file_names = os.listdir(samples_path)
#     assert len(list_file_names) >= 1, 'The number of files in the folder {} is probably 0, it must be >=1'.format(
#         samples_path)
#     log('******* Getting data from folder: {}, Nb of files : {}, First file : {}'.format(samples_path,
#                                                                                          len(list_file_names),
#                                                                                          list_file_names[0]),
#         logger_env)
#     for i, file_name in enumerate(list_file_names):
#         path = os.path.join(samples_path, file_name)
#         dict_from_pickle = load_pickle(path, logger_env=logger_env)
#
#         log('first_date: {}, last_date: {}'.format(dict_from_pickle['first_date'], dict_from_pickle['last_date']),
#             environment=logger_env)
#
#         X = dict_from_pickle['samples']
#         Y = dict_from_pickle[targets_type]
#         n_samples = dict_from_pickle['n_samples']
#
#         stop_1 = round(train_val_size * n_samples * train_size)
#         stop_2 = round(n_samples * train_size)
#         if i == 0:
#             X_train = X[:stop_1]
#             Y_train = Y[:stop_1]
#
#             X_val = X[stop_1:stop_2]
#             Y_val = Y[stop_1:stop_2]
#
#             X_test = X[stop_2:]
#             Y_test = Y[stop_2:]
#         else:
#
#             X_train = np.concatenate([X_train, X[:stop_1]])
#             Y_train = np.concatenate([Y_train, Y[:stop_1]])
#
#             X_val = np.concatenate([X_val, X[stop_1:stop_2]])
#             Y_val = np.concatenate([Y_val, Y[stop_1:stop_2]])
#
#             X_test = np.concatenate([X_test, X[stop_2:]])
#             Y_test = np.concatenate([Y_test, Y[stop_2:]])
#
#     # Distributions of Labels
#     train_d = np.sum(Y_train, axis=0) / len(Y_train)
#     val_d = np.sum(Y_val, axis=0) / len(Y_val)
#     tst_d = np.sum(Y_test, axis=0) / len(Y_test)
#
#     text_template = 'long: {:5.2f}%, hold: {:5.2f}%, short: {:5.2f}%'
#     log('Training Distribution of Labels :' + text_template.format(train_d[0] * 100, train_d[1] * 100,
#                                                                    train_d[2] * 100),
#         environment=logger_env)
#     log('Validation Distribution of Labels :' + text_template.format(val_d[0] * 100, val_d[1] * 100, val_d[2] * 100),
#         environment=logger_env)
#     log('Test Distribution of Labels :' + text_template.format(tst_d[0] * 100, tst_d[1] * 100, tst_d[2] * 100),
#         environment=logger_env)
#
#     return X_train, X_val, X_test, Y_train, Y_val, Y_test


def generate_dummy_data(batch_size):
    size_third = int(batch_size / 3.)
    rest = batch_size - 3 * size_third

    data_x_label_1 = np.random.uniform(-1, 0.9, (size_third, 16, 16, 4))
    data_x_label_2 = np.random.uniform(-0.95, 0.95, (size_third, 16, 16, 4))
    data_x_label_3 = np.random.uniform(-0.9, 1, (size_third + rest, 16, 16, 4))

    data_x_label_1 = np.asarray(data_x_label_1, np.float32)
    data_x_label_2 = np.asarray(data_x_label_2, np.float32)
    data_x_label_3 = np.asarray(data_x_label_3, np.float32)

    data_y_label_1 = np.asarray([[1, 0, 0] for i in range(size_third)], np.float32)
    data_y_label_2 = np.asarray([[0, 1, 0] for i in range(size_third)], np.float32)
    data_y_label_3 = np.asarray([[0, 0, 1] for i in range(size_third + rest)], np.float32)

    data_x = np.concatenate((data_x_label_1, data_x_label_2, data_x_label_3))
    data_y = np.concatenate((data_y_label_1, data_y_label_2, data_y_label_3))

    return data_x, data_y
