
"""
Created on Tue Oct 04 16:48:16 2022

Author: Chengqing Hu
Copyright 2022

This code is for data loading and pre-processing prior to ML modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from IPython.display import display



def sheet_retrieval(file_name, sheet_name=None, sep=None, low_memory=True,
                    engine='python'):  # , keep_default_na=False):
    # sheet_name is optional for '.xlsx' file
    # sep is optional for '.csv' file
    # engine is for pd.read_csv only for treating sep
    df_list = []

    if file_name.endswith('.csv'):
        try:
            df = pd.read_csv(file_name, sep=sep, low_memory=low_memory, engine=engine)
        except ValueError:
            df = None
            print(f"The csv file {str(file_name)} does not exist.")
    elif file_name.endswith('.xlsx'):
        if sheet_name is not None:
            try:
                df = pd.read_excel(file_name, sheet_name=sheet_name)  # , keep_default_na=keep_default_na)
            except ValueError:
                df = None
                print(f"Sheet {sheet_name} does not exist.")
        else:
            sheetname_list = pd.ExcelFile(file_name).sheet_names
            for sheet in sheetname_list:
                df_list.append(pd.read_excel(file_name, sheet_name=sheet))
    else:
        df = None
        raise ValueError("Only 'csv' or 'xlsx' file format is accepted.")

    if len(df_list) == 0:
        return df
    else:
        return sheetname_list, df_list


class DataLoader():
    def __init__(self, folder_path, file_name_train, file_name_test, target_col_name, list_target_vals, target_dtype=None,
                 cols_for_splitting=None, splitting_marks=None, list_new_cols_split=None, cols_to_drop=None,
                 dict_features_dtype=None, verbose=False, random_state=None):
        self.folder_path = folder_path                  # str
        self.filename_train = file_name_train           # str
        self.filename_test = file_name_test             # str
        self.target_col_name = target_col_name          # str
        self.target_vals = list_target_vals             # 1D list
        self.target_dtype = target_dtype                # valid dtype or None
        self.cols_for_splitting = cols_for_splitting    # 1D list of str or None
        self.splitting_marks = splitting_marks          # 1D list of str (of the same length as self.cols_for_splitting) or None
        self.list_new_cols_split = list_new_cols_split  # 2D list of str (each sublist should be of the same length as the number of elements after splitting that corresponding column) or None
        self.cols_to_drop = cols_to_drop                # 1D list of str  (should correspond to the columns after splitting operation is completed, and should not include any column shown in the cols_for_splitting list) or None
        self.dict_features_dtype = dict_features_dtype  # dict showing column to dtype mapping or None. Can be just part of the columns involved.
        self.verbose = verbose                          # bool
        self.random_state = random_state                # int

    def load(self):
        file_train_path = Path(self.folder_path) / self.filename_train
        file_test_path = Path(self.folder_path) / self.filename_test
        try:
            df_train_original = sheet_retrieval(str(file_train_path.resolve()))
            df_test_original = sheet_retrieval(str(file_test_path.resolve()))
        except:
            print(">>>>>> Error: Cannot load train/test data files.")
            return
        else:
            if self.cols_for_splitting is not None and self.splitting_marks is not None and self.list_new_cols_split is not None:
                try:
                    df_train_cols_splitting = df_train_original[self.cols_for_splitting].copy()         # pd.DataFrame
                    df_test_cols_splitting = df_test_original[self.cols_for_splitting].copy()           # pd.DataFrame
                    y_train_series = df_train_original[self.target_col_name].copy()                     # pd.Series (as the output y_train_series)
                    X_train_df = df_train_original.drop(self.cols_for_splitting, axis=1).copy()         # pd.DataFrame
                    X_train_df = X_train_df.drop(self.target_col_name, axis=1)                          # pd.DataFrame
                    X_test_df = df_test_original.drop(self.cols_for_splitting, axis=1).copy()           # pd.DataFrame
                except KeyError:
                    print(f">>>>>> KeyError: some column(s) in {str(self.cols_for_splitting)} or Column {self.target_col_name} do not exist in df_train_original or in df_test_original")
                    return
                else:
                    for col_to_split, split_mark, list_new_cols in zip(self.cols_for_splitting, self.splitting_marks, self.list_new_cols_split):
                        try:
                            X_train_df[list_new_cols] = df_train_cols_splitting[col_to_split].str.split(split_mark, expand=True)      # Add new columns to df_train after splitting the corresponding column
                            X_test_df[list_new_cols] = df_test_cols_splitting[col_to_split].str.split(split_mark, expand=True)        # Add new columns to df_test after splitting the corresponding column
                        except:
                            print(f">>>>>> Error: Wrong split mark '{split_mark}' or length mismatch for {str(list_new_cols)} vs. elements available by splitting Column {col_to_split}.")
                            return
            else:
                y_train_series = df_train_original[self.target_col_name].copy()             # pd.Series (as the output y_train_series)
                X_train_df = df_train_original.drop(self.target_col_name, axis=1).copy()    # pd.DataFrame
                X_test_df = df_test_original.copy()                                         # pd.DataFrame

            if self.cols_to_drop is not None:
                try:
                    X_train_df = X_train_df.drop(self.cols_to_drop, axis=1)
                    X_test_df = X_test_df.drop(self.cols_to_drop, axis=1)
                except KeyError:
                    print(f">>>>>> KeyError: some column(s) in {str(self.cols_to_drop)} do not exist in X_train_df or in X_test_df. Skipping dropping columns and continue.")

            X_train_df = X_train_df.fillna(0)
            X_test_df = X_test_df.fillna(0)
            y_train_series = y_train_series.fillna(1)

            if self.dict_features_dtype is not None:
                try:
                    X_train_df = X_train_df.astype(self.dict_features_dtype, errors='raise')
                    X_test_df = X_test_df.astype(self.dict_features_dtype, errors='raise')
                    y_train_series = y_train_series.astype(self.target_dtype)
                except:
                    print(f">>>>>> Error: Unable to convert column dtypes following user-provided "
                          f"{str(self.dict_features_dtype)} for features or {self.target_dtype} "
                          f"for target column. Skipping dtype conversion for columns and continue.")

            return X_train_df, X_test_df, y_train_series


class DataPreprocessor():
    def __init__(self, X_train_df, X_test_df, y_train_series, target_col_name, target_dtype,
                 cols_dtype_float=[], key_cols=[], verbose=False, random_state=35):
        self.X_train_df = X_train_df.copy()                                     # pd.DataFrame
        self.columns_train = self.X_train_df.columns.to_numpy()                 # 1D np.array
        self.X_test_df = X_test_df.copy()                                       # pd.DataFrame
        self.columns_test = self.X_test_df.columns.to_numpy()                   # 1D np.array
        self.y_train_series = y_train_series.copy()                             # pd.Series
        self.target_col_name = target_col_name                                  # str
        self.target_dtype = target_dtype                                        # object (dtype)
        self.cols_dtype_float = cols_dtype_float                                # 1D list, containing names of columns for conversion to float dtype. Default: []
        self.key_cols = key_cols                                                # 1D list, containing names of columns that are exempt from feature extraction. Default: []
        self.verbose = verbose                                                  # bool. Default: False
        self.random_state = random_state                                        # int, Default: 35
        #self.feature_estimator = None                                          # Initialized as None
        self.n_features_original = self.X_train_df.shape[1]                     # int
        self.n_classes = len(set(self.y_train_series.to_numpy().tolist()))      # int, number of classes (i.e., unique target labels)
        if set(self.columns_train.tolist()) == set(self.columns_test.tolist()):
            self.columns_X = self.columns_train                     # 1D np.array
            self.dtypes_X = self.X_train_df.dtypes.to_numpy()       # 1D np.array
            self.ready = True                                       # bool
        else:
            print(">>>>>> Error: Columns for train data are not the same as columns for test data. Further processing aborted")
            self.columns_X = None                                   # None
            self.dtypes_X = None                                    # None
            self.ready = False                                      # bool

    def conv_features_to_float(self):
        if self.ready and len(self.cols_dtype_float) > 0:
            if verbose:
                print("\n>>>>>> Dtype conversion (to float) in progress >>>>>>")
            try:
                self.X_train_df[self.cols_dtype_float] = self.X_train_df[self.cols_dtype_float].astype(float)
                self.X_test_df[self.cols_dtype_float] = self.X_test_df[self.cols_dtype_float].astype(float)
            except:
                print(">>>>>> Warning (TypeError/KeyError): Conversion of selected features to float type not successful.")
            finally:
                self.dtypes_X = self.X_train_df.dtypes.to_numpy()   # 1D np.array

    def feature_scaling(self, method):
        if self.ready:
            if method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif method == 'StandardScaler':
                scaler = StandardScaler()
            elif method == 'RobustScaler':
                scaler = RobustScaler()
            elif method == 'QuantileTransformer':
                scaler = QuantileTransformer()
            else:
                print(f">>>>>> Warning: User-provided scaling method ({method}) is not one of the available options: "
                      f"'MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'QuantileTransformer'. Scaling not "
                      f"executed.")
                return
            if verbose:
                print(f"\n>>>>>> Feature scaling using {method} method in progress >>>>>>")
            dtype_map_dict = dict(zip(self.columns_X.tolist(), self.dtypes_X.tolist()))
            self.X_train_df = pd.DataFrame(scaler.fit_transform(self.X_train_df), columns=self.columns_X).astype(dtype_map_dict, errors='raise')
            self.X_test_df = pd.DataFrame(scaler.fit_transform(self.X_test_df), columns=self.columns_X).astype(dtype_map_dict, errors='raise')

    def over_sampling(self):
        if self.ready:
            if verbose:
                print("\n>>>>>> Over-sampling in progress >>>>>>")
            smote = SMOTE()
            X_train_smote, y_train_smote = smote.fit_resample(self.X_train_df.to_numpy(), self.y_train_series.to_numpy())
            dtype_map_dict = dict(zip(self.columns_X.tolist(), self.dtypes_X.tolist()))
            self.X_train_df = pd.DataFrame(X_train_smote, columns=self.columns_X).astype(dtype_map_dict, errors='raise')
            try:
                self.y_train_series = pd.Series(y_train_smote, dtype=self.target_dtype, name=self.target_col_name)
            except:
                self.y_train_series = pd.Series(y_train_smote, name=self.target_col_name)
            if verbose:
                print(">>>>>> self.X_train_df after SMOTE:")
                self.X_train_df.info(verbose=verbose)
                print("\n>>>>>> self.y_train_series after SMOTE:")
                self.y_train_series.info(verbose=verbose)

    def feature_extraction(self, method='LDA', n_components=10):
        if self.ready and n_components < self.n_features_original:
            if len(self.key_cols) > 0:
                X_train_key_cols_df = self.X_train_df[self.key_cols].copy()
                X_test_key_cols_df = self.X_test_df[self.key_cols].copy()
                self.X_train_df = self.X_train_df.drop(self.key_cols, axis=1).copy()
                self.X_test_df = self.X_test_df.drop(self.key_cols, axis=1).copy()
            if method == 'LDA':     # Do not require feature scaling as a prerequisite
                try:
                    estimator = LDA(n_components=n_components)
                except:
                    n_components = self.n_classes - 1
                    estimator = LDA(n_components=n_components)       # This can happen especially when min(n_features_original, n_class - 1) < n_components.
                                                                     # In the case of binary classification, LDA will have to reduce the dimensionality to 1,
                                                                     # which does not necessarily serve the purpose of dimensionality reduction.
                estimator.fit(self.X_train_df.to_numpy(), self.y_train_series.to_numpy())
                self.X_train_df = pd.DataFrame(estimator.transform(self.X_train_df.to_numpy()), columns=np.arange(1, n_components + 1).astype(str))
                self.X_test_df = pd.DataFrame(estimator.transform(self.X_test_df.to_numpy()), columns=np.arange(1, n_components + 1).astype(str))
            elif method == 'PCA-UMAP-unsupervised':  # Require feature scaling as a prerequisite
                ### Checkpoint ###
                n_comp_pca = max(min(50, self.n_features_original), n_components)
                ### End of Checkpoint ###
                estimator_1 = PCA(n_components=n_comp_pca)
                X_train_array = estimator_1.fit_transform(self.X_train_df.to_numpy())
                X_test_array = estimator_1.fit_transform(self.X_test_df.to_numpy())
                estimator_2 = umap.UMAP(n_neighbors=20, n_components=n_components, metric='euclidean', min_dist=0.1, random_state=self.random_state)
                self.X_train_df = pd.DataFrame(estimator_2.fit_transform(X_train_array), columns=np.arange(1, n_components + 1).astype(str))
                self.X_test_df = pd.DataFrame(estimator_2.fit_transform(X_test_array), columns=np.arange(1, n_components + 1).astype(str))
            elif method == 'PCA-UMAP-supervised':    # Require feature scaling as a prerequisite
                ### Checkpoint ###
                n_comp_pca = max(min(50, self.n_features_original), n_components)
                ### End of Checkpoint ###
                estimator_1 = PCA(n_components=n_comp_pca)
                X_train_array = estimator_1.fit_transform(self.X_train_df.to_numpy())
                X_test_array = estimator_1.fit_transform(self.X_test_df.to_numpy())
                estimator_2 = umap.UMAP(n_neighbors=20, n_components=n_components, metric='euclidean', min_dist=0.1,
                                        random_state=self.random_state)
                estimator_2.fit(X_train_array, self.y_train_series.to_numpy())
                self.X_train_df = pd.DataFrame(estimator_2.transform(X_train_array),
                                               columns=np.arange(1, n_components + 1).astype(str))
                self.X_test_df = pd.DataFrame(estimator_2.transform(X_test_array),
                                              columns=np.arange(1, n_components + 1).astype(str))
            if len(self.key_cols) > 0:
                self.X_train_df = pd.concat([self.X_train_df, X_train_key_cols_df], axis=1)
                self.X_test_df = pd.concat([self.X_test_df, X_test_key_cols_df], axis=1)

            if verbose:
                print(f"\n>>>>>> self.X_train_df after feature extraction using {method} method and {n_components} n_components:")
                display(self.X_train_df.head(5).to_string())
                self.X_train_df.info(verbose=verbose)

                print(f"\n>>>>>> self.X_test_df after feature extraction using {method} method and {n_components} n_components:")
                display(self.X_test_df.head(5).to_string())
                self.X_test_df.info(verbose=verbose)



if __name__ == '__main__':
    ### Checkpoint ###
    folder_path = r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set"
    file_name_train = "Train.csv"
    file_name_test = "Test.csv"
    target_col_name = "passfail"
    list_target_vals = [1, 0]
    target_dtype = int
    cols_for_splitting = ['die_id']
    splitting_marks = ['-']
    list_new_cols_split = [['fablot', 'wafer', 'diex', 'diey']]
    cols_to_drop = ['fablot']
    dict_features_dtype = {'wafer': int, 'diex': int, 'diey': int}
    verbose = True
    random_state = 35

    key_cols = ['wafer', 'diex', 'diey']
    scaling_method = 'MinMaxScaler'                         # Available options: 'MinMaxScaler', 'StandardScaler', 'RobustScaler', 'QuantileTransformer'
    feature_extraction_method = 'PCA-UMAP-unsupervised'     # Available options: 'LDA', 'PCA-UMAP-unsupervised', 'PCA-UMAP-supervised'
    n_components = 50
    ### End of Checkpoint ###

    data_loader = DataLoader(folder_path=folder_path, file_name_train=file_name_train, file_name_test=file_name_test,
                             target_col_name=target_col_name, list_target_vals=list_target_vals,
                             target_dtype=target_dtype, cols_for_splitting=cols_for_splitting,
                             splitting_marks=splitting_marks, list_new_cols_split=list_new_cols_split,
                             cols_to_drop=cols_to_drop, dict_features_dtype=dict_features_dtype, verbose=verbose,
                             random_state=random_state)
    try:
        X_train_df, X_test_df, y_train_series = data_loader.load()
    except:
        print(f">>>>>> Error: Cannot load the train ({file_name_train}) and test ({file_name_test}) data properly.")
    else:
        print(f"\n>>>>>> First 5 rows of X_train_df >>>>>>")
        display(X_train_df.head(5).to_string())
        if verbose:
            print(f"X_train_df dtypes: {str(X_train_df.dtypes.to_numpy().tolist())}")
            X_train_df.info(verbose=verbose)

        print(f"\n>>>>>> First 5 rows of X_test_df >>>>>>")
        display(X_test_df.head(5).to_string())
        if verbose:
            print(f"X_test_df dtypes: {str(X_test_df.dtypes.to_numpy().tolist())}")
            X_test_df.info(verbose=verbose)

        print(f'\n>>>>>> First 5 elements of y_train_series >>>>>>')
        print(str(y_train_series[:5].to_numpy().tolist()))
        if verbose:
            y_train_series.info(verbose=verbose)

        """
        cols_dtype_float = ['sme1.dd00vth_sgddown_delta_u3s', 'sme1.dd00vth_sgddown_post_u3s', 'sme1.dd00vth_sgddown_pre_u3s',
                            'sme1.dd01vth_sgddown_delta_u3s', 'sme1.dd01vth_sgddown_pre_u3s', 'sme1.prg_12p_vth_b0w0_inn_l2s',
                            'sme1.prg_12p_vth_b0w0_out_u2s', 'sme1.prg_12p_vth_b0w1_inn_l2s', 'sme1.prg_12p_vth_b0w1_out_u2s',
                            'sme1.prg_12p_vth_b1w0_inn_med', 'sme1.prg_12p_vth_b1w0_out_med', 'sme1.prg_12p_vth_b1w1_inn_l2s',
                            'sme1.prg_12p_vth_b1w1_out_u2s', 'sme1.sblk1_pre_12pvt_out_wl95_u2s', 'sme1.sblk1_pst_12pvt_inn_wl00_l2s',
                            'sme1.sblk1_pst_12pvt_inn_wl00_u2s', 'sme1.sblk1_pst_12pvt_inn_wl26_l2s', 'sme1.sblk1_pst_12pvt_inn_wl26_u2s',
                            'sme1.sblk1_pst_12pvt_inn_wl95_l2s', 'sme1.sblk1_pst_12pvt_inn_wl95_u2s', 'sme1.sblk1_pst_12pvt_out_wl00_l2s',
                            'sme1.sblk1_pst_12pvt_out_wl00_u2s', 'sme1.sblk1_pst_12pvt_out_wl26_l2s', 'sme1.sblk1_pst_12pvt_out_wl26_u2s',
                            'sme1.sblk1_pst_12pvt_out_wl95_l2s', 'sme1.sblk1_pst_12pvt_out_wl95_u2s', 'sme1.sblk1_wlrc_val_w1_p0',
                            'sme1.sblk1_wlrc_val_w1_p1', 'sme1.sblk1_wlrc_val_w2_p0', 'sme1.sblk1_wlrc_val_w2_p1', 'sme1.sblk1_wlrc_val_w3_p0',
                            'sme1.sblk1_wlrc_val_w3_p1', 'sme1.sblk1_wlrc_val_w4_p1', 'sme1.sblk2_pre_12pvt_out_wl95_u2s',
                            'sme1.sblk2_pst_12pvt_inn_wl00_l2s', 'sme1.sblk2_pst_12pvt_inn_wl00_u2s', 'sme1.sblk2_pst_12pvt_inn_wl26_l2s',
                            'sme1.sblk2_pst_12pvt_inn_wl26_u2s', 'sme1.sblk2_pst_12pvt_inn_wl95_l2s', 'sme1.sblk2_pst_12pvt_inn_wl95_u2s',
                            'sme1.sblk2_pst_12pvt_out_wl00_l2s', 'sme1.sblk2_pst_12pvt_out_wl00_u2s', 'sme1.sblk2_pst_12pvt_out_wl26_l2s',
                            'sme1.sblk2_pst_12pvt_out_wl26_u2s',	'sme1.sblk2_pst_12pvt_out_wl95_l2s', 'sme1.sblk2_pst_12pvt_out_wl95_u2s',
                            'sme1.sblk2_wlrc_val_w1_p0', 'sme1.sblk2_wlrc_val_w1_p1', 'sme1.sblk2_wlrc_val_w2_p0', 'sme1.sblk2_wlrc_val_w2_p1',
                            'sme1.sblk2_wlrc_val_w3_p0', 'sme1.sblk2_wlrc_val_w3_p1', 'sme1.sblk2_wlrc_val_w4_p0', 'sme1.sblk2_wlrc_val_w4_p1',
                            'sme1.sgd0vth_sgddown_delta_l3s', 'sme1.sgd0vth_sgddown_post_l3s', 'sme1.sgd0vth_sgddown_pre_l3s',
                            'sme1.sgd1vth_sgddown_delta_l3s', 'sme1.sgdprog_erase_pass', 'sme1.sgdprog_prog_pass', 'sme1.trim_prgloop_tr0_wl00',
                            'sme1.trim_prgloop_tr0_wl47', 'sme1.trim_prgloop_tr0_wl48', 'sme1.trim_prgloop_tr0_wl68', 'sme1.trim_prgloop_tr0_wl95',
                            'sme1.vera_mlc_erloop_tr0', 'sme1.vera_mlc_erloop_tr1', 'sme1.vera_mlc_loop_delta', 'sme1.vera_mlc_rate',
                            'sme1.vera_slc_erloop_tr0', 'sme1.vera_slc_erloop_tr1', 'sme1.vera_slc_loop_delta', 'sme1.vera_slc_rate',
                            'sme1.vpgmslc_erase_rate', 'sme1.vpgmu_loop_delta', 'sme1.vpgmu_prog_rate', 'sme1.vth_from0_u3s',
                            'sme1.vth_from0_win', 'sme1.vth_from1_u3s', 'sme1.vth_from1_win', 'sme1.vth_sgd_afsgdmpw_l3s',
                            'sme1.vth_sgd_afsgdmpw_u3s', 'sme1.vth_sgd_afsgdmpw_win', 'sme1.vth_sgd_afsgdpr_u3s', 'sme1.vth_sgd_afsgdpr_win',
                            'sme1.vth_sgd_l3s', 'sme1.vth_sgd_med', 'sme1.vth_sgd_u3s', 'sme1.vth_sgd_win', 'sme1.vth_sgs_afsgspr_l3s',
                            'sme1.vth_sgs_afsgspr_win', 'sme1.vth_sgs_l3s', 'sme1.vth_sgs_med', 'sme1.vth_sgs_u3s', 'sme1.vth_sgs_win',
                            'sme1.vth_sgsb_l3s', 'sme1.vth_sgsb_med', 'sme1.vth_sgsb_u3s', 'sme1.vth_sgsb_win', 'sme1.vth_wl72_l3s',
                            'sme1.vth_wl72_med', 'sme1.vth_wl72_u3s', 'sme1.vth_wl72_win', 'sme1.vth_wldd0_aftprognvt_l3s',
                            'sme1.vth_wldd0_l3s', 'sme1.vth_wldd0_med', 'sme1.vth_wldd0_u3s', 'sme1.vth_wldd0_win',
                            'sme1.vth_wldd1_aftprognvt_l3s', 'sme1.vth_wldd1_aftprognvt_win', 'sme1.vth_wldd1_l3s', 'sme1.vth_wldd1_med',
                            'sme1.vth_wldd1_u3s', 'sme1.vth_wldd1_win', 'sme1.vth_wldl_aftprognvt_l3s', 'sme1.vth_wldl_aftprognvt_win',
                            'sme1.vth_wldl_l3s', 'sme1.vth_wldl_med', 'sme1.vth_wldl_u3s', 'sme1.vth_wldl_win', 'sme1.vth_wlds0_aftprognvt_l3s',
                            'sme1.vth_wlds0_aftprognvt_med', 'sme1.vth_wlds0_aftprognvt_u3s', 'sme1.vth_wlds0_aftprognvt_win', 'sme1.vth_wlds0_l3s',
                            'sme1.vth_wlds0_med', 'sme1.vth_wlds0_u3s', 'sme1.vth_wlds0_win', 'sme1.vth_wlds1_aftprognvt_l3s',
                            'sme1.vth_wlds1_aftprognvt_win', 'sme1.vth_wlds1_l3s', 'sme1.vth_wlds1_med', 'sme1.vth_wlds1_u3s',
                            'sme1.vth_wlds1_win', 'sme1.vth_wldu_aftprognvt_l3s', 'sme1.vth_wldu_aftprognvt_win', 'sme1.vth_wldu_l3s',
                            'sme1.vth_wldu_med', 'sme1.vth_wldu_u3s', 'sme1.vth_wldu_win']
        """

        cols_dtype_float = X_train_df.columns.to_numpy().tolist()
        data_preprocessor = DataPreprocessor(X_train_df, X_test_df, y_train_series, target_col_name, target_dtype,
                                             cols_dtype_float=cols_dtype_float, key_cols=key_cols, verbose=verbose,
                                             random_state=random_state)
        data_preprocessor.conv_features_to_float()
        data_preprocessor.feature_scaling(method=scaling_method)
        data_preprocessor.over_sampling()
        data_preprocessor.feature_extraction(method=feature_extraction_method, n_components=n_components)

        print(f"\n>>>>>> First 5 rows of X_train_df >>>>>>")
        display(data_preprocessor.X_train_df.head(5).to_string())
        if verbose:
            print(f"X_train_df dtypes: {str(data_preprocessor.X_train_df.dtypes.to_numpy().tolist())}")
            data_preprocessor.X_train_df.info(verbose=verbose)

        print(f"\n>>>>>> First 5 rows of X_test_df >>>>>>")
        display(data_preprocessor.X_test_df.head(5).to_string())
        if verbose:
            print(f"X_test_df dtypes: {str(data_preprocessor.X_test_df.dtypes.to_numpy().tolist())}")
            data_preprocessor.X_test_df.info(verbose=verbose)

        print(f'\n>>>>>> First 5 elements of y_train_series >>>>>>')
        print(str(data_preprocessor.y_train_series[:5].to_numpy().tolist()))
        if verbose:
            data_preprocessor.y_train_series.info(verbose=verbose)


        X_train_output = data_preprocessor.X_train_df.join(data_preprocessor.y_train_series)
        X_train_output = X_train_output.sample(frac=1)
        train_num = int(X_train_output.shape[0] * 0.8)
        train_set = X_train_output.iloc[:train_num, :]
        crsv_set = X_train_output.iloc[train_num + 1:-1, :]
        train_set.to_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/train_scale_train.csv", index=False)
        crsv_set.to_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/train_scale_crsv.csv", index=False)