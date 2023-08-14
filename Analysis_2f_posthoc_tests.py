import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import numpy as np
import datetime as dt
import scipy
import joblib

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

from Functions.Plotting import plot_hist
from Functions.RF_preprocess import to_categorical
from RUN_Analysis_2 import start_string, global_seed, analysis_path, version
from Functions.Enet_preprocess import dummy_code
from Fixed_params import cat_vars, drop_list
from Initial_Preprocess import dependent_variable

results_path = analysis_path + "Results/Performance_all/"
save_path = analysis_path + "Results/Post_hoc/"

np.random.seed(global_seed)
dv_T1 = dependent_variable+'_T1'

time_date_string = start_string
time_date_string = "_04_Nov_2022__09.05"

# -------------
start = dt.datetime.now()
np.random.seed(global_seed)

# set save paths and directories
data_path = analysis_path + "Processed_Data/"
directory_bits = os.fsencode(data_path)

# Get best params
rf_param_path = analysis_path + "Results/Best_params/RF/"
enet_param_path = analysis_path + "Results/Best_params/Enet/"

df1_rf_best_params = joblib.load(rf_param_path + 'df{}__{}.pkl'.format('1', time_date_string))
df2_rf_best_params = joblib.load(rf_param_path + 'df{}__{}.pkl'.format('2', time_date_string))
df3_rf_best_params = joblib.load(rf_param_path + 'df{}__{}.pkl'.format('3', time_date_string))
df4_rf_best_params = joblib.load(rf_param_path + 'df{}__{}.pkl'.format('4', time_date_string))
df5_rf_best_params = joblib.load(rf_param_path + 'df{}__{}.pkl'.format('5', time_date_string))

all_params_rf = {'1': df1_rf_best_params, '2': df2_rf_best_params, '3': df3_rf_best_params,
                 '4': df4_rf_best_params, '5': df5_rf_best_params}

df1_enet_best_params = joblib.load(enet_param_path + 'df{}__{}.pkl'.format('1', time_date_string))
df2_enet_best_params = joblib.load(enet_param_path + 'df{}__{}.pkl'.format('2', time_date_string))
df3_enet_best_params = joblib.load(enet_param_path + 'df{}__{}.pkl'.format('3', time_date_string))
df4_enet_best_params = joblib.load(enet_param_path + 'df{}__{}.pkl'.format('4', time_date_string))
df5_enet_best_params = joblib.load(enet_param_path + 'df{}__{}.pkl'.format('5', time_date_string))

all_params_enet = {'1': df1_enet_best_params, '2': df2_enet_best_params, '3': df3_enet_best_params,
                   '4': df4_enet_best_params, '5': df5_enet_best_params}

lm_scores_dict = {}
rf_scores_dict = {}
dummy_scores_dict = {}
lin_regr_scores_dict = {}
folds = 3
train_scoring = 'r2'
decimal_places = 4

if __name__ == "__main__":
    # loop through dfs:
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith("_preprocessed{}.csv".format(version)):
            start_df = dt.datetime.now()
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            # open print file:
            save_file = analysis_path + "Results/Post_hoc/df{}_{}.txt".format(df_num, start_string)
            with pd.option_context('display.float_format', '{:0.2f}'.format):
                print("df{} results, trained using {}-fold cv: {}\n".format(df_num, folds, start_string.replace("_", " ").replace("  ", " ")),
                      file=open(save_file, "w"))

            df = pd.read_csv(data_path+filename)

            # code categorical vars:
            df = to_categorical(df, cat_vars)

            for col in df.columns:
                if col in drop_list:
                    df.drop(col, axis=1, inplace=True)

            # pipeline steps:
            imputer = IterativeImputer(missing_values=np.nan, max_iter=10)
            transformer = StandardScaler()

            # get best params:
            rf_params = all_params_rf[df_num]
            enet_params = all_params_enet[df_num]

            # -------------
            # pipeline Enet:
            # -------------
            model = "Enet"

            # additional preprocess steps:
            df_enet = dummy_code(df=df, vars_to_code=cat_vars)

            # define X and y:
            y = df_enet["sges"]
            X = df_enet.drop("sges", axis=1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}\n".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                          y_test.shape[0], round(y_test.mean(), 2)),
                  file=open(save_file, "a"))

            # ---------------------------------------------
            elastic_net_regression = ElasticNet()

            print("Elastic net", file=open(save_file, "a"))

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

            print("Best elastic net performance on test data: \n"
                  " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(test_r2_score,
                                                                                 test_var_expl_score,
                                                                                 test_mae_score,
                                                                                 test_rmse_score),
                  file=open(save_file, "a"))

            # save score to dict:
            lm_results_dict = {'r2': test_r2_score, "var_expl": test_var_expl_score, "MAE": test_mae_score, "RMSE": test_rmse_score}
            lm_scores_dict[df_num]=lm_results_dict

            # -------------
            # pipeline RF:
            # -------------
            model = "RF"
            # define X and y:
            y = df["sges"]
            X = df.drop("sges", axis=1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}; mean={},\ny test n={}; mean={}\n".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                          y_test.shape[0], round(y_test.mean(), 2)),
                  file=open(save_file, "a"))

            print("Random forest", file=open(save_file, "a"))

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

            print("Best random forest model performance on test data: \n"
                  " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(test_r2_score,
                                                                                 test_var_expl_score,
                                                                                 test_mae_score,
                                                                                 test_rmse_score),
                  file=open(save_file, "a"))

            # save score to dict:
            rf_results_dict = {'r2': test_r2_score, "var_expl": test_var_expl_score, "MAE": test_mae_score,
                            "RMSE": test_rmse_score}
            rf_scores_dict[df_num] = rf_results_dict

            # ===========================
            # Compare to baseline measure
            dummy_regr = DummyRegressor(strategy="mean")
            dummy_regr.fit(X_train, y_train)
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

            # ----------------------------------
            # Compare to DV_T1 only as baseline:
            lin_regr = LinearRegression()
            lin_regr.fit(np.array(X_train[dv_T1]).reshape(-1, 1), y_train)

            lin_regr_y_pred = lin_regr.predict(np.array(X_test[dv_T1]).reshape(-1, 1))

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
            lin_regr_results_dict = {'r2': lin_regr_r2_score, "var_expl": lin_regr_expl_score,
                                  "MAE": lin_regr_mae_score, "RMSE": lin_regr_rmse_score}

            lin_regr_scores_dict[df_num] = lin_regr_results_dict

            # statistically compare errors:
            # todo: qqplot
            err_save_path = "Analysis_2_with_sges_T1/Results/Post_hoc/Err_dist/"
            bins = 25
            err_dum = np.log10(abs(y_test - dummy_y_pred))
            plot = "log10_mae_dummy"
            plot_hist(save_name=plot+"_df{}".format(df_num), x=err_dum, save_path=err_save_path,
                      title=plot, bins=bins)

            err_lin_regr = np.log10(abs(y_test - lin_regr_y_pred))
            plot = "log10_mae_lin_regr_y"
            plot_hist(save_name=plot+"_df{}".format(df_num), x=err_lin_regr, save_path=err_save_path,
                      title=plot, bins=bins)

            err_lm = np.log10(abs(y_test - y_pred_lm))
            plot = "log10_mae_enet"
            plot_hist(save_name=plot+"_df{}".format(df_num), x=err_lm, save_path=err_save_path,
                      title=plot, bins=bins)

            err_rf = np.log10(abs(y_test - y_pred_rf))
            plot = "log10_mae_rf"
            plot_hist(save_name=plot+"_df{}".format(df_num), x=err_rf, save_path=err_save_path,
                      title=plot, bins=bins)

            t_lm, p_lm = scipy.stats.ttest_rel(err_dum, err_lm, axis=0, nan_policy='propagate')
            t_rf, p_rf = scipy.stats.ttest_rel(err_dum, err_rf, axis=0, nan_policy='propagate')
            t_c, p_c = scipy.stats.ttest_rel(err_lm, err_rf, axis=0, nan_policy='propagate')

            t_lr_lm, p_lr_lm = scipy.stats.ttest_rel(err_lin_regr, err_lm, axis=0, nan_policy='propagate')
            t_lr_rf, p_lr_rf = scipy.stats.ttest_rel(err_lin_regr, err_rf, axis=0, nan_policy='propagate')
            t_lr_d, p_lr_d = scipy.stats.ttest_rel(err_dum, err_lin_regr, axis=0, nan_policy='propagate')

            print("t-test dummy and LM: p={}, t={}".format(f'{p_lm:.4f}', f'{t_lm:.4f}'), file=open(save_file, "a"))
            print("t-test dummy and RF: p={}, t={}".format(f'{p_rf:.4f}', f'{t_rf:.4f}'), file=open(save_file, "a"))
            print("t-test LM and RF: p={}, t={}".format(f'{p_c:.4f}', f'{t_c:.4f}'), file=open(save_file, "a"))

            print("t-test {} and LM: p={}, t={}".format(dv_T1, f'{p_lr_lm:.4f}', f'{t_lr_lm:.4f}'), file=open(save_file, "a"))
            print("t-test {} and RF: p={}, t={}".format(dv_T1, f'{p_lr_rf:.4f}', f'{t_lr_rf:.4f}'), file=open(save_file, "a"))
            print("t-test dummy and {}: p={}, t={}".format(dv_T1, f'{p_lr_d:.4f}', f'{t_lr_d:.4f}'), file=open(save_file, "a"))

            df_end_time = dt.datetime.now()
            df_run_time = df_end_time - start_df
            print('\n----------Total df run time----------\n{}'.format(df_run_time), file=open(save_file, "a"))

            # plot dummy:

            continue
        else:
            continue

    overall_save = analysis_path + "Results/Performance_all/"
    dv_t1_results_df = pd.DataFrame.from_dict(lin_regr_scores_dict, orient='index').reset_index()
    dv_t1_results_df.to_csv(overall_save+"DV_T1_{}.csv".format(start_string))

    end_time = dt.datetime.now()
    run_time = end_time - start
    print('done! run time: {}'.format(run_time))


