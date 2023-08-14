
import os
import re
import pandas as pd
import numpy as np
import datetime as dt
import scipy
import joblib
import matplotlib.pyplot as plt
from py._builtin import execfile

from scipy.stats import linregress
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from Functions.Enet_preprocess import dummy_code
from Functions.RF_preprocess import to_categorical
from RUN_Analysis_2 import start_string, global_seed, analysis_path, version, Choose_drop_DV_T1
from Fixed_params import cat_vars, drop_list
from Analysis_2f_posthoc_tests import all_params_enet, all_params_rf
from Functions.Plotting import plot_results

results_path = "Analysis_2{}/Results/Predict_ahead_x_years/Performance_all/".format(version)
dependent_variable = "sges"
new_data_path = "Data/Predict_x_years_ahead/"
new_directory_bits = os.fsencode(new_data_path)

# set save paths and directories
old_data_path = analysis_path + "Processed_Data/"
old_directory_bits = os.fsencode(old_data_path)

drop_school_track = True

if __name__ == "__main__":
    create_dataframes = True
    decimal_places = 4
    dv_T1 = dependent_variable+'_T1'

    # Get best params
    rf_param_path = analysis_path + "Results/Best_params/RF/"
    enet_param_path = analysis_path + "Results/Best_params/Enet/"

    # collate old analysis dataframes:
    if create_dataframes == True:
        dfs_dict = {}
        # loop through dfs:
        for file in os.listdir(old_directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("_preprocessed{}.csv".format(version)):
                start_df = dt.datetime.now()
                print("--> Processing.... {}".format(filename))
                df_num = [s for s in re.findall(r'\d+', filename)][0]

                df = pd.read_csv(old_data_path + filename)

                # code categorical vars:
                df = to_categorical(df, cat_vars)

                if drop_school_track == True:
                    df.drop('sctyp', axis=1, inplace=True)

                print("df num {} shape: {}".format(df_num, str(df.shape)))

                for col in df.columns:
                    if col in drop_list:
                        df.drop(col, axis=1, inplace=True)

                model = "Enet"

                # define X and y:
                y = df["sges"]
                X = df.drop("sges", axis=1)

                # store X_and_y in dict:
                dfs_dict[df_num] = {'X': X, "y": y}

        y_future_num = 2
        counter = 1
        predict_ahead_dfs_dict = {}
        for df_num in [1, 2, 3, 4]:
            if df_num == 1:
                predict_ahead_dfs_dict[df_num] = dfs_dict[str(df_num)]
                # sges_T1 = dfs_dict[str(df_num)]['X']['sges_T1']
                print("df{}: X={}, y={}".format(df_num, str(df_num), str(df_num + 1)))
            else:
                y_future_num = y_future_num + 1
                predict_ahead_dfs_dict[df_num] = {'X': dfs_dict['1']['X'],
                                                  'y': dfs_dict[str(y_future_num)]['y']}
                print("df{}: X={}, y={}".format(df_num, str(1), str(y_future_num)))

            X_and_y = pd.concat([predict_ahead_dfs_dict[df_num]['X'],
                                 pd.DataFrame(predict_ahead_dfs_dict[df_num]['y'])], axis=1)
            # remove rows with no y (as don't want to impute). In this analysis N will drop off over time.
            X_and_y.dropna(axis=0, inplace=True, subset=[dependent_variable])
            # save
            X_and_y.to_csv(new_data_path + "predict_ahead_df{}.csv".format(str(df_num)))

    # run prediction and interpretation for each X and y

    model = "Enet"
    save_file_path = "Analysis_2_with_sges_T1/Results/Predict_ahead_x_years/Prediction/"
    lm_scores_dict = {}
    rf_scores_dict = {}
    dummy_scores_dict = {}
    DV_T1_scores_dict = {}
    for file in os.listdir(new_directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith(".csv".format(version)):
            start_df = dt.datetime.now()
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]
            save_file = save_file_path + "df{}_{}.txt".format(df_num, start_string)

            df = pd.read_csv(new_data_path + filename)

            # check removed rows where no y
            df.dropna(axis=0, inplace=True, subset=[dependent_variable])

            # code categorical vars:
            df = to_categorical(df, cat_vars)

            for col in df.columns:
                if col in drop_list:
                    df.drop(col, axis=1, inplace=True)

            # -------------
            # pipeline Enet:
            # -------------
            model = "Enet"
            print("Elastic net", file=open(save_file, "w"))

            # get best params:
            enet_params = all_params_enet[df_num]
            print("(best) params used: {}".format(enet_params), file=open(save_file, "a"))

            # additional preprocess steps:
            df_enet = dummy_code(df=df, vars_to_code=cat_vars)

            # define X and y:
            y = df_enet["sges"]
            X = df_enet.drop("sges", axis=1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)

            # ---------------------------------------------
            # pipeline steps:
            imputer = IterativeImputer(missing_values=np.nan, max_iter=10, random_state=93)
            transformer = StandardScaler()
            elastic_net_regression = ElasticNet()

            pipe = Pipeline([
                ('imputing', imputer),
                ('scaling', transformer),
                ('regression', elastic_net_regression)
            ])

            pipe.set_params(**enet_params)

            # Fit the best model to the training data
            pipe.fit(X_train, y_train)

            # Evaluate/score best out of sample
            y_pred_lm = pipe.predict(X_test)
            test_r2_score = round(metrics.r2_score(y_test, y_pred_lm), decimal_places)
            test_var_expl_score = round(metrics.explained_variance_score(y_test, y_pred_lm), decimal_places)
            test_mae_score = round(metrics.mean_absolute_error(y_test, y_pred_lm), decimal_places)
            test_rmse_score = round(metrics.mean_squared_error(y_test, y_pred_lm, squared=False), decimal_places)

            # save score to dict:
            lm_results_dict = {'r2': test_r2_score, "var_expl": test_var_expl_score, "MAE": test_mae_score,
                               "RMSE": test_rmse_score}
            lm_scores_dict[df_num] = lm_results_dict

            print("Best elastic net performance on test data: \n"
                  " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(test_r2_score,
                                                                                  test_var_expl_score,
                                                                                  test_mae_score,
                                                                                  test_rmse_score),
                  file=open(save_file, "a"))

            # -------------
            # pipeline RF:
            # -------------
            model = "RF"
            print("\nRandom Forest", file=open(save_file, "a"))

            # get best params:
            rf_params = all_params_rf[df_num]
            print("(best) params used: {}".format(rf_params), file=open(save_file, "a"))

            # define X and y:
            y = df["sges"]
            X = df.drop("sges", axis=1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)

            # ---------------------------------------------
            # pipeline steps:
            random_forest = RandomForestRegressor()

            pipe = Pipeline([
                ('imputing', imputer),
                ('scaling', transformer),
                ('regression', random_forest)
            ])

            pipe.set_params(**rf_params)

            # Fit the best model to the training data
            pipe.fit(X_train, y_train)

            # Evaluate/score best out of sample
            y_pred_rf = pipe.predict(X_test)
            test_r2_score = round(metrics.r2_score(y_test, y_pred_rf), decimal_places)
            test_var_expl_score = round(metrics.explained_variance_score(y_test, y_pred_rf), decimal_places)
            test_mae_score = round(metrics.mean_absolute_error(y_test, y_pred_rf), decimal_places)
            test_rmse_score = round(metrics.mean_squared_error(y_test, y_pred_rf, squared=False), decimal_places)

            # save score to dict:
            rf_results_dict = {'r2': test_r2_score, "var_expl": test_var_expl_score, "MAE": test_mae_score,
                               "RMSE": test_rmse_score}
            rf_scores_dict[df_num] = rf_results_dict

            print("Best random forest performance on test data: \n"
                  " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(test_r2_score,
                                                                                  test_var_expl_score,
                                                                                  test_mae_score,
                                                                                  test_rmse_score),
                  file=open(save_file, "a"))

            # --------------------------------------------------------------------------------------------------------
            DV_T1_index = X_train.columns.get_loc(dv_T1)

            # compare to a mean baseline measure
            imputer.fit(X_train)
            X_train = imputer.transform(X_train)

            dummy_regr = DummyRegressor(strategy="mean")
            dummy_regr.fit(X_train, y_train)

            imputer.fit(X_train)
            X_test = imputer.transform(X_test)
            dummy_y_pred = dummy_regr.predict(X_test)

            dummy_r2_score = round(metrics.r2_score(y_test, dummy_y_pred), decimal_places)
            dummy_expl_score = round(metrics.explained_variance_score(y_test, dummy_y_pred), decimal_places)
            dummy_mae_score = round(metrics.mean_absolute_error(y_test, dummy_y_pred), decimal_places)
            dummy_rmse_score = round(metrics.mean_squared_error(y_test, dummy_y_pred, squared=False), decimal_places)

            print("\n----------Comparisons----------", file=open(save_file, "a"))
            print("Dummy model performance on test data: \n"
                  " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(dummy_r2_score,
                                                                                  dummy_expl_score,
                                                                                  dummy_mae_score,
                                                                                  dummy_rmse_score),
                  file=open(save_file, "a"))
            # save score to dict:
            dummy_results_dict = {'r2': dummy_r2_score, "var_expl": dummy_expl_score, "MAE": dummy_mae_score,
                                  "RMSE": dummy_rmse_score}
            dummy_scores_dict[df_num] = dummy_results_dict
            # --------------------------------------------------------------------------------------------------------
            # Compare to DV_T1 only as baseline:

            lin_regr = LinearRegression()
            lin_regr.fit(X_train[:, DV_T1_index].reshape(-1, 1), y_train)

            lin_regr_y_pred = lin_regr.predict(X_test[:, DV_T1_index].reshape(-1, 1))

            lin_regr_r2_score = round(metrics.r2_score(y_test, lin_regr_y_pred), decimal_places)
            lin_regr_expl_score = round(metrics.explained_variance_score(y_test, lin_regr_y_pred), decimal_places)
            lin_regr_mae_score = round(metrics.mean_absolute_error(y_test, lin_regr_y_pred), decimal_places)
            lin_regr_rmse_score = round(metrics.mean_squared_error(y_test, lin_regr_y_pred, squared=False), decimal_places)

            print("\n----------Comparisons----------", file=open(save_file, "a"))
            print("DV_T1 model performance on test data: \n"
                  " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(lin_regr_r2_score,
                                                                                  lin_regr_expl_score,
                                                                                  lin_regr_mae_score,
                                                                                  lin_regr_rmse_score),
                  file=open(save_file, "a"))

            # save score to dict:
            DV_T1_results_dict = {'r2': lin_regr_r2_score, "var_expl": lin_regr_expl_score,
                                     "MAE": lin_regr_mae_score, "RMSE": lin_regr_rmse_score}

            DV_T1_scores_dict[df_num] = DV_T1_results_dict

    # save dicts as dataframes:

    dummy_results_df = pd.DataFrame.from_dict(dummy_scores_dict, orient='index').reset_index()
    dummy_results_df.to_csv(results_path + "Dummy_{}.csv".format(start_string))

    DV_T1_results_df = pd.DataFrame.from_dict(DV_T1_scores_dict, orient='index').reset_index()
    DV_T1_results_df.to_csv(results_path + "DV_T1_{}.csv".format(start_string))

    lm_results_df = pd.DataFrame.from_dict(lm_scores_dict, orient='index').reset_index()
    lm_results_df.to_csv(results_path + "ENet_{}.csv".format(start_string))

    rf_results_df = pd.DataFrame.from_dict(rf_scores_dict, orient='index').reset_index()
    rf_results_df.to_csv(results_path + "RF_{}.csv".format(start_string))

    # plot prediction results ---------------------------------------------------------------------------------

    np.random.seed(global_seed)
    datetime = start_string
    Include_T5 = False

    rf_results = pd.read_csv(results_path + "RF_" + datetime + ".csv", index_col="Unnamed: 0")
    enet_results = pd.read_csv(results_path + "ENet_" + datetime + ".csv", index_col="Unnamed: 0")
    dum_results = pd.read_csv(results_path + "Dummy_" + datetime + ".csv", index_col="Unnamed: 0")

    if Include_T5 == False:
        rf_results.drop(rf_results.loc[rf_results["index"] == 5].index, inplace=True)
        enet_results.drop(enet_results.loc[enet_results["index"] == 5].index, inplace=True)
        dum_results.drop(dum_results.loc[dum_results["index"] == 5].index, inplace=True)
        xticks = ["5 --> 6", "6 --> 7", "7 --> 8", "8 --> 9"]
    else:
        xticks = ['5 --> 6', '6 --> 7', '7 --> 8', '8 --> 9', '9 --> 10']

    rf_results['Model'] = "Random Forest"
    enet_results['Model'] = "Elastic Net"
    dum_results['Model'] = "Mean Baseline"

    if Choose_drop_DV_T1 == False:
        # compare to sges_T1 only when sges is not dropped (for now)
        dv_t1_results = pd.read_csv(results_path + "DV_T1_{}.csv".format(start_string), index_col="Unnamed: 0")
        dv_t1_results['Model'] = "DV Time 1 Baseline"
        if Include_T5 == False:
            dv_t1_results.drop(dv_t1_results.loc[dv_t1_results["index"] == 5].index, inplace=True)

        r2_res2 = pd.concat([rf_results[["index", "r2", "Model"]],
                             enet_results[["index", "r2", "Model"]],
                             dv_t1_results[["index", "r2", "Model"]],
                             dum_results[["index", "r2", "Model"]]], axis=0
                            )
        MAE_res2 = pd.concat([rf_results[["index", "MAE", "Model"]],
                              enet_results[["index", "MAE", "Model"]],
                              dv_t1_results[["index", "MAE", "Model"]],
                              dum_results[["index", "MAE", "Model"]]], axis=0
                             )
        save_name = "R2_DVT1" + datetime
        save_path = results_path + "Plots/"

        plot_results(x="index", y="r2", data=r2_res2, colour='Model',
                     save_path=save_path, save_name=save_name,
                     xlab="Schooling Year (Grade)", ylab="Prediction R Squared",
                     title="Comparison of Predictions", legend_pos="upper left",
                     xaxis_labs=xticks, y_lim=(0, 1))

        save_name = "MAE_DVT1" + datetime
        plot_results(x="index", y="MAE", data=MAE_res2, colour='Model',
                     save_path=save_path, save_name=save_name,
                     xlab="Schooling Year (Grade)", ylab="MAE",
                     title="Comparison of Predictions", legend_pos="upper left",
                     xaxis_labs=xticks)


    r2_res = pd.concat([rf_results[["index", "r2", "Model"]],
                       enet_results[["index", "r2", "Model"]]], axis=0
                       )

    MAE_res = pd.concat([rf_results[["index", "MAE", "Model"]],
                        enet_results[["index", "MAE", "Model"]],
                        dum_results[["index", "MAE", "Model"]]], axis=0
                        )

    save_path = results_path + "Plots/"

    save_name = "R2" + datetime
    plot_results(x="index", y="r2", data=r2_res, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="Prediction R Squared",
                 title="Comparison of Predictions",
                 xaxis_labs=xticks)

    save_name = "MAE" + datetime
    plot_results(x="index", y="MAE", data=MAE_res, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="MAE",
                 title="Comparison of Predictions",
                 xaxis_labs=xticks)

    # SHAP block importance

    execfile('Analysis_2h_SHAP_Importance.py')

print('done!')


