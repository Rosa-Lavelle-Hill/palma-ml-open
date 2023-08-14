import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from Fixed_params import cat_vars, drop_list, cat_vars_agg
from Functions.Enet_preprocess import dummy_code, remove_special_char_colnames
from Functions.Plotting import plot_perm_box, plot_perm
from Functions.RF_preprocess import to_categorical
from RUN_Analysis_6 import start_string, global_seed, analysis_path, version

start_time = dt.datetime.now()
# ----
permutations = 1
data_used = 'test'
n_plot=25
scoring="r2"

np.random.seed(global_seed)
# start_string = "_18_Aug_2022__10.03"
# ^test string
time_date_string = start_string

# -------
data_path = analysis_path + "Processed_Data/Processed_Aggregate_Features/"
directory_bits = os.fsencode(data_path)
fig_save_path_full = analysis_path + "Results/Importance/Plots/Full_Data"
csv_save_path_full = analysis_path + "Results/Importance/csv/Full_Data/"
fig_save_path_test = analysis_path + "Results/Importance/Plots/Test_Data/"
csv_save_path_test = analysis_path + "Results/Importance/csv/Test_Data/"
rf_param_path = analysis_path + "Results/Best_params/RF/"
# -------
df1_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('4', time_date_string))
df5_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('5', time_date_string))

all_params_rf = {'1': df1_rf_best_params, '2': df2_rf_best_params, '3': df3_rf_best_params,
                 '4': df4_rf_best_params, '5': df5_rf_best_params}

enet_param_path = analysis_path + "Results/Best_params/Enet/"
df1_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('4', time_date_string))
df5_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('5', time_date_string))

all_params_enet = {'1': df1_enet_best_params, '2': df2_enet_best_params, '3': df3_enet_best_params,
                   '4': df4_enet_best_params, '5': df5_enet_best_params}

# loop through dfs:
for file in os.listdir(directory_bits):
    filename = os.fsdecode(file)
    if filename.endswith("_preprocessed{}.csv".format(version)):
        start_df = dt.datetime.now()
        print("--> Processing.... {}".format(filename))
        df_num = [s for s in re.findall(r'\d+', filename)][0]

        df = pd.read_csv(data_path+filename)

        # code categorical vars:
        df = to_categorical(df, cat_vars_agg)

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

        # if data_used == 'full':
        #     print('Analysing on full data...')
        #     print('Fitting Elastic Net...')
        #     pipe.fit(X, y)
        #     # ^ note: fitting to full dataset**
        #
        #     result = permutation_importance(
        #         pipe, X, y, n_repeats=permutations, random_state=93, n_jobs=2, scoring=scoring,
        #     )
        #
        #     if permutations == 1:
        #         perm_importances = result.importances_mean
        #         vars = X.columns.to_list()
        #         dict = {'Feature': vars, "Importance": perm_importances}
        #         perm_imp_df = pd.DataFrame(dict)
        #         perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
        #         perm_imp_df['Importance'] = round(perm_imp_df['Importance'], 4)
        #         perm_imp_df.to_csv(csv_save_path_full + "df{}_Enet_permutation_{}{}.csv".format(df_num,
        #                                                                                           scoring,
        #                                                                                           time_date_string))
        #         # only plot first n:
        #         perm_imp_df = perm_imp_df[0:n_plot]
        #         perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
        #         # plot
        #         plot_perm(df=perm_imp_df, n_plot=n_plot, save_path=fig_save_path_full,
        #                   title="Elastic Net Permutation Importance (full data)",
        #                   x='Importance', y='Feature',
        #                   save_name="df{}_Enet_permutation_{}{}.png".format(df_num, scoring, time_date_string))
        #     else:
        #         print('Elastic Net permutations started...')
        #         plot_perm_box(X=X, result=result, save_path=fig_save_path_full,
        #                       save_name="df{}_Enet_permutations_Box_n{}_{}{}".format(df_num,
        #                                                                                permutations,
        #                                                                                scoring,
        #                                                                                time_date_string
        #                                                                                ),
        #                       permutations=permutations, xlab='Importance', ylab='Feature',
        #                       title="Elastic Net Permutation Importance (full data) n={}".format(permutations))

        # =======
        # Test data:
        # perm on test data only (these features less likely to cause over-fitting):

        if data_used == 'test':
            print('Analysing on test data...')
            # split data using same params (should be same as the train/test set used for prediction)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                        y_test.shape[0], round(y_test.mean(), 2)))
            print('Fitting Elastic Net...')
            pipe.fit(X_train, y_train)

            result = permutation_importance(
                pipe, X_test, y_test, n_repeats=permutations, random_state=93, n_jobs=2, scoring=scoring
            )

            if permutations == 1:
                perm_importances = result.importances_mean
                vars = X.columns.to_list()
                dict = {'Feature': vars, "Importance": perm_importances}
                perm_imp_df = pd.DataFrame(dict)
                perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
                perm_imp_df['Importance'] = round(perm_imp_df['Importance'], 4)
                perm_imp_df.to_csv(csv_save_path_test + "df{}_Enet_permutation_{}{}.csv".format(df_num,
                                                                                                  scoring,
                                                                                                  time_date_string
                                                                                                  ))
                # only plot first 25:
                perm_imp_df = perm_imp_df[0: n_plot]
                perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)

                # plot
                plot_perm(df=perm_imp_df, n_plot=n_plot, save_path=fig_save_path_test,
                          title="Elastic Net Permutation Importance (test data)",
                          x='Importance', y='Feature',
                          save_name="df{}_Enet_permutation_{}{}.png".format(df_num, scoring, time_date_string))

            else:
                print('Elastic Net n={} permutations started...'.format(permutations))
                plot_perm_box(X=X_test, result=result, save_path=fig_save_path_test,
                              save_name="df{}_Enet_permutations_Box_n{}_{}{}".format(df_num,
                                                                                       permutations,
                                                                                       scoring,
                                                                                       time_date_string
                                                                                       ),
                              permutations=permutations, xlab='Importance', ylab='Feature', n_plot=n_plot,
                              title="Elastic Net Permutation Importance (test data) n={}".format(permutations))
                print('Elastic Net permutations finished.')

        # -----------------------------------------
        # RF
        # -----------------------------------------
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

        # =======
        # perm imp on 100% data (train and test)
        if data_used == 'full':
            print("Analysing on full data...")
            print("Fitting Random Forest...")
            pipe.fit(X, y)
            # ^ note: fitting to full dataset

            result = permutation_importance(
                pipe, X, y, n_repeats=permutations, random_state=93, n_jobs=2, scoring=scoring
            )
            if permutations == 1:
                perm_importances = result.importances_mean
                vars = X.columns.to_list()
                dict = {'Feature': vars, "Importance": perm_importances}
                perm_imp_df = pd.DataFrame(dict)
                perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
                perm_imp_df['Importance'] = round(perm_imp_df['Importance'], 4)
                perm_imp_df.to_csv(csv_save_path_full + "df{}_RF_permutation_{}{}.csv".format(df_num,
                                                                                                scoring,
                                                                                                time_date_string
                                                                                                ))
                # only plot first n:
                perm_imp_df = perm_imp_df[0:n_plot]
                perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
                # plot
                plot_perm(df=perm_imp_df, n_plot=n_plot, save_path=fig_save_path_full,
                          title="Random Forest Permutation Importance (full data)",
                          x='Importance', y='Feature',
                          save_name="df{}_RF_permutation_{}{}.png".format(df_num, scoring, time_date_string))
            else:
                print('Random Forest n={} permutations started...'.format(permutations))
                plot_perm_box(X=X, result=result, save_path=fig_save_path_test,
                              save_name="df{}_RF_permutations_Box_n{}_{}{}".format(df_num,
                                                                                     permutations,
                                                                                     scoring,
                                                                                     time_date_string
                                                                                     ),
                              permutations=permutations, xlab='Importance', ylab='Feature', n_plot=n_plot,
                              title="Random Forest Permutation Importance (full data) n={}".format(permutations))
                print('Random Forest permutations finished.')

        # =======
        # Test data:
        # perm on test data only (these features less likely to cause over-fitting):
        pipe.set_params(**rf_params)

        if data_used == 'test':
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                        y_test.shape[0], round(y_test.mean(), 2)))
            print('Analysing on test data...')
            print('Fitting Random Forest...')
            pipe.fit(X_train, y_train)

            result = permutation_importance(
                pipe, X_test, y_test, n_repeats=permutations, random_state=93, n_jobs=2, scoring=scoring
            )
            if permutations == 1:
                perm_importances = result.importances_mean
                vars = X.columns.to_list()
                dict = {'Feature': vars, "Importance": perm_importances}
                perm_imp_df = pd.DataFrame(dict)
                perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
                perm_imp_df['Importance'] = round(perm_imp_df['Importance'], 4)
                perm_imp_df.to_csv(csv_save_path_test + "df{}_RF_permutation_{}{}.csv".format(df_num,
                                                                                                scoring,
                                                                                                time_date_string
                                                                                                ))
                # only plot first 25:
                perm_imp_df = perm_imp_df[0:n_plot]
                perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
                # plot
                plot_perm(df=perm_imp_df, n_plot=n_plot, save_path=fig_save_path_test,
                          title="Random Forest Permutation Importance (test data)",
                          x='Importance', y='Feature',
                          save_name="df{}_RF_permutation_{}{}.png".format(df_num, scoring, time_date_string))
            else:
                print('Random Forest n={} permutations started...'.format(permutations))
                plot_perm_box(X=X_test, result=result, save_path=fig_save_path_test,
                              save_name="df{}_RF_permutations_Box_n{}_{}{}".format(df_num,
                                                                                     permutations,
                                                                                     scoring,
                                                                                     time_date_string
                                                                                     ),
                              permutations=permutations, xlab='Importance', ylab='Feature', n_plot=n_plot,
                              title="Random Forest Permutation Importance (test data) n={}".format(permutations))
                print('Random Forest permutations finished.')


end_time = dt.datetime.now()
run_time = end_time - start_time
print("permutation importance runtime: {}".format(run_time))

