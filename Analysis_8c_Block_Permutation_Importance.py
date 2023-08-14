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

from Fixed_params import cat_vars, drop_list
from Functions.Block_Permutation_Importance import block_importance, add_aggregates_to_blockfile
from Functions.Enet_preprocess import dummy_code, remove_special_char_colnames
from Functions.Plotting import plot_perm_box, plot_perm
from Functions.RF_preprocess import to_categorical
from RUN_Analysis_8 import start_string, global_seed, analysis_path, use_preoptimised_model,\
    block_file, preoptimised_model, version

print("Running block permutation importance analysis")
start_time = dt.datetime.now()
# ----
permutations = 1
data_used = 'test'
n_plot=25
scoring="r2"

np.random.seed(global_seed)
time_date_string = start_string

# time_date_string = "_12_Jul_2022__19.40_test"
# # ^test string
# -------
data_path = "Data/Initial_Preprocess/Data_with_Aggregate_Features/"
directory_bits = os.fsencode(data_path)
fig_save_path_full = analysis_path + "Results/Importance/Plots/Full_Data"
csv_save_path_full = analysis_path + "Results/Importance/csv/Full_Data/"
fig_save_path_test = analysis_path + "Results/Importance/Plots/Test_Data/"
csv_save_path_test = analysis_path + "Results/Importance/csv/Test_Data/"

if use_preoptimised_model == False:
    rf_param_path = analysis_path + "Results/Best_params/RF/"
    enet_param_path = analysis_path + "Results/Best_params/Enet/"

if use_preoptimised_model == True:
    analysis_path = "Analysis_{}{}/".format(preoptimised_model, version)
    rf_param_path = analysis_path + "Results/Best_params/RF/"
    enet_param_path = analysis_path + "Results/Best_params/Enet/"
# -------
df1_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('4', time_date_string))
df5_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('5', time_date_string))

all_params_rf = {'1': df1_rf_best_params, '2': df2_rf_best_params, '3': df3_rf_best_params,
                 '4': df4_rf_best_params, '5': df5_rf_best_params}

df1_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('4', time_date_string))
df5_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('5', time_date_string))

all_params_enet = {'1': df1_enet_best_params, '2': df2_enet_best_params, '3': df3_enet_best_params,
                   '4': df4_enet_best_params, '5': df5_enet_best_params}

# add _C vars to blockfile:
agg_block_file = "aggregate_variable_blocks_final.csv"
save_path = "Data/MetaData/Variable_blocks/"
add_aggregates_to_blockfile(block_file, save_path="Data/MetaData/Variable_blocks/", save_name=agg_block_file)

# loop through dfs:
for file in os.listdir(directory_bits):
    filename = os.fsdecode(file)
    if filename.endswith("aggregate_preprocessed{}.csv".format(version)):
        start_df = dt.datetime.now()
        print("--> Processing.... {}".format(filename))
        df_num = [s for s in re.findall(r'\d+', filename)][0]

        df = pd.read_csv(data_path+filename)

        # code categorical vars:
        df = to_categorical(df, cat_vars)

        for col in df.columns:
            if col in drop_list:
                df.drop(col, axis=1, inplace=True)

        # get best params:

        rf_params = all_params_rf[df_num]
        enet_params = all_params_enet[df_num]

        # fit models ---------------------------------
        # pipeline steps:
        imputer = IterativeImputer(missing_values=np.nan, max_iter=10)
        transformer = StandardScaler()

        # ---------------------------------------
        # ENet
        # ---------------------------------------
        model = "Enet"

        # additional preprocess steps:
        df_enet = dummy_code(df=df, vars_to_code=cat_vars)
        df_enet = remove_special_char_colnames(df_enet, '.')

        # define X and y:
        y = df_enet["sges"]
        X = df_enet.drop("sges", axis=1)

        elastic_net_regression = ElasticNet()

        pipe = Pipeline([
            ('imputing', imputer),
            ('scaling', transformer),
            ('regression', elastic_net_regression)
        ])
        pipe.set_params(**enet_params)

        if data_used == 'test':
            print('Analysing on test data...')
            # split data using same params (should be same as the train/test set used for prediction)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                        y_test.shape[0], round(y_test.mean(), 2)))
            print('Fitting Elastic Net...')

            result = block_importance(pipe=pipe, X=X_test, y=y_test, save_path=csv_save_path_test,
                                      random_state=global_seed, df_num=df_num, model=model,
                                      block_file=save_path+agg_block_file)
        # -----------------------------------------
        # RF
        # -----------------------------------------
        model = "RF"
        random_forest = RandomForestRegressor()
        pipe = Pipeline([
            ('imputing', imputer),
            ('scaling', transformer),
            ('regression', random_forest)
        ])

        pipe.set_params(**rf_params)

        # define X and y:
        y = df["sges"]
        X = df.drop("sges", axis=1)

        if data_used == 'test':
            print('Analysing on test data...')
            # split data using same params (should be same as the train/test set used for prediction)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                        y_test.shape[0], round(y_test.mean(), 2)))
            print('Fitting RF...')

            result = block_importance(pipe=pipe, X=X_test, y=y_test, save_path=csv_save_path_test,
                                      random_state=global_seed, df_num=df_num, model=model,
                                      block_file=save_path+agg_block_file)

end_time = dt.datetime.now()
run_time = end_time - start_time
print("permutation importance runtime: {}".format(run_time))

print('done!')