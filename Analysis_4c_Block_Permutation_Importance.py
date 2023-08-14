import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from Fixed_params import cat_vars, drop_list, parental_degree_expectations, imputer_model, imputer_max_iter, block_file
from Functions.Block_Permutation_Importance import block_importance, parse_data_to_dict
from Functions.Check import all_vars_in_block
from Functions.Enet_preprocess import dummy_code, remove_special_char_colnames
from Functions.Generic_Fit_Transform import preprocess_transform
from Functions.Plotting import plot_perm_box, plot_perm
from Functions.Predict import build_pipeline_enet, build_pipeline_rf
from Functions.RF_preprocess import to_categorical
from RUN_Analysis_4 import start_string, global_seed, analysis_path, \
    use_preoptimised_model, version, preoptimised_model, drop_houspan, drop_p_degree_expect

print("Running block permutation importance analysis")
start_time = dt.datetime.now()
# ----
permutations = 1
data_used = 'test'
n_plot=25
scoring="r2"
include_T5 = False

np.random.seed(global_seed)
time_date_string = start_string

# time_date_string = "_29_Jul_2022__09.54"
# ^test string
# -------
data_path = "Data/Initial_Preprocess/"
directory_bits = os.fsencode("Data/Initial_Preprocess/")

if use_preoptimised_model == False:
    rf_param_path = analysis_path + "Results/Best_params/RF/"
    enet_param_path = analysis_path + "Results/Best_params/Enet/"

if use_preoptimised_model == True:
    load_analysis_path = "Analysis_{}{}/".format(preoptimised_model, version)
    rf_param_path = load_analysis_path + "Results/Best_params/RF/"
    enet_param_path = load_analysis_path + "Results/Best_params/Enet/"

fig_save_path_full = analysis_path + "Results/Importance/Plots/Full_Data"
csv_save_path_full = analysis_path + "Results/Importance/csv/Full_Data/"
fig_save_path_test = analysis_path + "Results/Importance/Plots/Test_Data/"
csv_save_path_test = analysis_path + "Results/Importance/csv/Test_Data/"
# -------

df1_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('4', time_date_string))
if include_T5 == True:
    df5_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('5', time_date_string))
    all_params_rf = {'1': df1_rf_best_params, '2': df2_rf_best_params, '3': df3_rf_best_params,
                     '4': df4_rf_best_params, '5': df5_rf_best_params}
else:
    all_params_rf = {'1': df1_rf_best_params, '2': df2_rf_best_params, '3': df3_rf_best_params,
                     '4': df4_rf_best_params}

df1_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('4', time_date_string))
if include_T5 == True:
    df5_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('5', time_date_string))
    all_params_enet = {'1': df1_enet_best_params, '2': df2_enet_best_params, '3': df3_enet_best_params,
                       '4': df4_enet_best_params, '5': df5_enet_best_params}
else:
    all_params_enet = {'1': df1_enet_best_params, '2': df2_enet_best_params, '3': df3_enet_best_params,
                       '4': df4_enet_best_params}

block_dict = parse_data_to_dict(block_file)

# loop through dfs:
for file in os.listdir(directory_bits):
    filename = os.fsdecode(file)
    if filename.endswith("_preprocessed{}.csv".format(version)):
        start_df = dt.datetime.now()
        print("--> Processing.... {}".format(filename))
        df_num = [s for s in re.findall(r'\d+', filename)][0]

        if include_T5 == False:
            if df_num == '5':
                print('skip')
                continue

        df = pd.read_csv(data_path+filename)

        vars_not_in = all_vars_in_block(df, blockdict=block_dict, print=False)

        if drop_p_degree_expect == True:
            drop_list = drop_list + parental_degree_expectations
        if drop_houspan == True:
            drop_list = drop_list + ["houspan"]
            cat_vars.remove("houspan")

        # code categorical vars:
        df = to_categorical(df, cat_vars)

        for col in df.columns:
            if col in drop_list:
                df.drop(col, axis=1, inplace=True)

        # define X and y:
        y = df["sges"]
        X = df.drop("sges", axis=1)

        # get best params:
        rf_params = all_params_rf[df_num]
        enet_params = all_params_enet[df_num]

        # ---------------------------------------
        # ENet
        # ---------------------------------------
        model = "Enet"

        # pipeline construction
        pipe, preprocessor = build_pipeline_enet(X=X, imputer_model=imputer_model,
                                                 oh_encoder_drop='first',
                                                 imputer_max_iter=imputer_max_iter)

        pipe.set_params(**enet_params)

        #  check all processed vars in block
        X_tr = preprocess_transform(X, X, pipe, preprocessor)
        enet_vars_not_in = all_vars_in_block(X_tr, blockdict=block_dict, print=False)

        if data_used == 'test':
            print('Analysing on test data...')
            # split data using same params (should be same as the train/test set used for prediction)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                        y_test.shape[0], round(y_test.mean(), 2)))
            print('Fitting Elastic Net...')

            result = block_importance(pipe=pipe, X=X_test, y=y_test, save_path=csv_save_path_test,
                                      random_state=global_seed, df_num=df_num, model=model, block_file=block_file)

        # -----------------------------------------
        # RF
        # -----------------------------------------
        model = "RF"

        # pipeline construction
        pipe, preprocessor = build_pipeline_rf(X=X, imputer_model=imputer_model,
                                               oh_encoder_drop='if_binary',
                                               imputer_max_iter=imputer_max_iter)
        pipe.set_params(**rf_params)

        if data_used == 'test':
            print('Analysing on test data...')
            # split data using same params (should be same as the train/test set used for prediction)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                        y_test.shape[0], round(y_test.mean(), 2)))
            print('Fitting RF...')

            result = block_importance(pipe=pipe, X=X_test, y=y_test, save_path=csv_save_path_test,
                                      random_state=global_seed, df_num=df_num, model=model, block_file=block_file)


end_time = dt.datetime.now()
run_time = end_time - start_time
print("permutation importance runtime: {}".format(run_time))

print('done!')