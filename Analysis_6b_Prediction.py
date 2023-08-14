
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from Functions.Enet_preprocess import dummy_code
from Functions.RF_preprocess import to_categorical
from RUN_Analysis_6 import start_string, global_seed, rf_params, enet_params,\
    analysis_path, version
from Fixed_params import cat_vars, drop_list, cat_vars_agg

# -------------
start = dt.datetime.now()
np.random.seed(global_seed)

# set save paths and directories
data_path = analysis_path + "Processed_Data/Processed_Aggregate_Features/"
directory_bits = os.fsencode(data_path)

lm_scores_dict = {}
rf_scores_dict = {}
dummy_scores_dict = {}
folds = 5
train_scoring = 'r2'
decimal_places = 4

# loop through dfs:
for file in os.listdir(directory_bits):
    filename = os.fsdecode(file)
    if filename.endswith("_preprocessed{}.csv".format(version)):
        start_df = dt.datetime.now()
        print("--> Processing.... {}".format(filename))
        df_num = [s for s in re.findall(r'\d+', filename)][0]

        # open print file:
        save_file = analysis_path + "Results/Prediction/df{}_{}.txt".format(df_num, start_string)
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print("df{} results, trained using {}-fold cv: {}\n".format(df_num, folds, start_string.replace("_", " ").replace("  ", " ")),
                  file=open(save_file, "w"))

        df = pd.read_csv(data_path+filename)

        # code categorical vars:
        df = to_categorical(df, cat_vars_agg)

        for col in df.columns:
            if col in drop_list:
                df.drop(col, axis=1, inplace=True)

        # pipeline steps:
        imputer = IterativeImputer(missing_values=np.nan, max_iter=10)
        # ^note: not converging so in final run set max_iterator higher, and also try with DT (current is ridge reg.)?
        transformer = StandardScaler()

        # -------------
        # pipeline Reg:
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
              file = open(save_file, "a"))

        # save data sets:
        modelling_data_save = analysis_path + "Modelling_Data/"
        X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
        X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
        pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
        pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))

        # ----
        elastic_net_regression = ElasticNet()

        print("\n----------Elastic net----------", file=open(save_file, "a"))
        print("params tried:\n{}\n".format(enet_params), file=open(save_file, "a"))

        pipe = Pipeline([
            ('imputing', imputer),
            ('scaling', transformer),
            ('regression', elastic_net_regression)
        ])

        # Grid search on training data

        grid_start = dt.datetime.now()
        grid_search = GridSearchCV(estimator=pipe,
                                   param_grid=enet_params,
                                   cv=folds,
                                   scoring=train_scoring,
                                   refit=True,
                                   verbose=1,
                                   n_jobs=2)
        grid_search.fit(X_train, y_train)
        param_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
        param_df.to_csv(analysis_path +'Outputs/Modelling/Grid_Search/Enet/' + 'df{}_{}.csv'.format(df_num, start_string))

        print("Linear regression:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.options.display.max_colwidth = 100
            print(param_df[['rank_test_score',
                            'params',
                            'mean_test_score']].iloc[0:10])

        grid_end = dt.datetime.now()
        training_time = grid_end - grid_start
        print("Elastic net training done. Time taken: {}".format(training_time))
        print("Elastic net training time: {}".format(training_time), file=open(save_file, "a"))

        # Get best model on training data
        best_params = grid_search.best_params_
        best_train_score = round(abs(grid_search.best_score_), decimal_places)
        print("Best training {} score: {}. Best model params:\n{}.\n".format(train_scoring, best_train_score, best_params),
              file=open(save_file, "a"))

        # Fit the best model to the training data
        pipe.set_params(**best_params)
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

        # save best params as a pickle:
        params_save = analysis_path +"Results/Best_params/Enet/"
        joblib.dump(best_params, params_save + 'df{}__{}.pkl'.format(df_num, start_string), compress=1)

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

        # save data sets:
        modelling_data_save = analysis_path + "Modelling_Data/"
        X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
        X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
        pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
        pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))

        print("\n----------Random forest----------", file=open(save_file, "a"))
        print("params tried:\n{}\n".format(rf_params), file=open(save_file, "a"))

        random_forest = RandomForestRegressor()
        pipe = Pipeline([
            ('imputing', imputer),
            ('scaling', transformer),
            ('regression', random_forest)
        ])
        # Grid search on training data

        grid_start = dt.datetime.now()
        grid_search = GridSearchCV(estimator=pipe,
                                   param_grid=rf_params,
                                   cv=folds,
                                   scoring=train_scoring,
                                   refit=True,
                                   verbose=1,
                                   n_jobs=2)
        grid_search.fit(X_train, y_train)
        param_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
        param_df.to_csv(analysis_path +'Outputs/Modelling/Grid_Search/RF/'+'df{}_{}.csv'.format(df_num, start_string))

        print("Random forest:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.options.display.max_colwidth = 100
            print(param_df[['rank_test_score',
                            'params',
                            'mean_test_score']].iloc[0:10])

        grid_end = dt.datetime.now()
        training_time = grid_end - grid_start
        print("RF training done. Time taken: {}".format(training_time))
        print("RF training time: {}".format(training_time), file=open(save_file, "a"))

        # Get best model on training data
        best_params = grid_search.best_params_
        best_train_score = round(abs(grid_search.best_score_), decimal_places)

        print("Best training {} score: {}. Best model params:\n{}.\n".format(train_scoring, best_train_score, best_params),
              file=open(save_file, "a"))

        # Fit the best model to the training data
        pipe.set_params(**best_params)
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

        # save best params as a pickle:
        params_save = analysis_path + "Results/Best_params/RF/"
        joblib.dump(best_params, params_save + 'df{}__{}.pkl'.format(df_num, start_string), compress=1)

        # ===========================
        # compare to baseline measure
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

        # statistically compare errors:
        mae_dum = abs(y_test - dummy_y_pred)
        mae_lm = abs(y_test - y_pred_lm)
        mae_rf = abs(y_test - y_pred_rf)

        t_lm, p_lm = scipy.stats.ttest_rel(mae_dum, mae_lm, axis=0, nan_policy='propagate')
        t_rf, p_rf = scipy.stats.ttest_rel(mae_dum, mae_rf, axis=0, nan_policy='propagate')
        t_c, p_c = scipy.stats.ttest_rel(mae_lm, mae_rf, axis=0, nan_policy='propagate')

        print("t-test LM and dummy: p={}, t={}".format(f'{p_lm:.4f}', f'{t_lm:.4f}'), file=open(save_file, "a"))
        print("t-test RF and dummy: p={}, t={}".format(f'{p_rf:.4f}', f'{t_rf:.4f}'), file=open(save_file, "a"))
        print("t-test LM and RF: p={}, t={}".format(f'{p_c:.4f}', f'{t_c:.4f}'), file=open(save_file, "a"))

        df_end_time = dt.datetime.now()
        df_run_time = df_end_time - start_df
        print('\n----------Total df run time----------\n{}'.format(df_run_time), file=open(save_file, "a"))

        continue
    else:
        continue

# save dicts as dataframes:
overall_save = analysis_path + "Results/Performance_all/"
dummy_results_df = pd.DataFrame.from_dict(dummy_scores_dict, orient='index').reset_index()
dummy_results_df.to_csv(overall_save+"Dummy_{}.csv".format(start_string))

lm_results_df = pd.DataFrame.from_dict(lm_scores_dict, orient='index').reset_index()
lm_results_df.to_csv(overall_save+"ENet_{}.csv".format(start_string))

rf_results_df = pd.DataFrame.from_dict(rf_scores_dict, orient='index').reset_index()
rf_results_df.to_csv(overall_save+"RF_{}.csv".format(start_string))

end_time = dt.datetime.now()
run_time = end_time - start
print('done! run time: {}'.format(run_time))