import re
import pandas as pd
import numpy as np
import datetime as dt
import scipy
import joblib
import os
from tqdm import tqdm
from scipy.stats import linregress
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from Fixed_params import parental_degree_expectations, cat_vars, drop_list, imputer_model, imputer_max_iter,\
        drop_p_degree_expect, drop_houspan
from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Check import check_miss
from Functions.Correlations import check_imputed_cors_across_blocks, eval_imputed_track_cors
from Functions.Generic_Fit_Transform import preprocess_transform
from Functions.Plotting import plot_hist
from Functions.Predict import build_pipeline_enet, build_pipeline_rf, build_preprocessor_ns_base
from Functions.RF_preprocess import to_categorical

def predict(directory_bits, analysis_path, data_path,
            start_string, enet_params, rf_params, test,
            use_generic_grid = True,
            block_file = None, drop_df_5=True,
            drop_list=drop_list, check_imputed_cors=False,
            folds=5, version="", compare_DV_T1_baseline =True, addition_pred_checks=False,
            train_scoring= 'r2', decimal_places=4, dependent_variable="sges", compare_non_survey_baseline=True):

    if use_generic_grid == True:
        if test == True:
            from Generic_Grid import test_enet_param_grid, test_rf_param_grid
            enet_params = test_enet_param_grid
            rf_params = test_rf_param_grid
        if test == False:
            from Generic_Grid import enet_param_grid, rf_param_grid
            enet_params = enet_param_grid
            rf_params = rf_param_grid

    corr_print_threshold = 0
    if test == True:
        folds = 2

    if drop_p_degree_expect == True:
        drop_list = drop_list + parental_degree_expectations
    if drop_houspan == True:
        drop_list = drop_list + ["houspan"]
        if "houspan" in cat_vars:
            cat_vars.remove("houspan")

    print("executing predict function")
    dv_T1 = dependent_variable + '_T1'

    lm_scores_dict = {}
    rf_scores_dict = {}
    dummy_scores_dict = {}
    DV_T1_scores_dict = {}
    NS_base_scores_dict = {}

    for file in tqdm(os.listdir(directory_bits)):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            start_df = dt.datetime.now()
            print("--> Processing.... {}".format(filename))

            if addition_pred_checks == False:
                df_num = [s for s in re.findall(r'\d+', filename)][0]
            if addition_pred_checks == True:
                df_num_list = [s for s in re.findall(r'\d+', filename)]
                df_num = ''.join(df_num_list)

            if drop_df_5 == True:
                if df_num == '5':
                    print("skip")
                    continue

            # open print file:
            save_file = analysis_path + "Results/Prediction/df{}_{}.txt".format(df_num, start_string)
            with pd.option_context('display.float_format', '{:0.2f}'.format):
                print("df{} results, trained using {}-fold cv: {}\n".format(df_num, folds,
                                                                            start_string.replace("_", " ").replace("  ",
                                                                                                                   " ")),
                      file=open(save_file, "w"))

            df = pd.read_csv(data_path + filename)

            # code categorical vars:
            df = to_categorical(df, cat_vars)

            for col in df.columns:
                if col in drop_list:
                    df.drop(col, axis=1, inplace=True)

            # if missing, fill houspan (parents) with housechan (child):
            if drop_houspan == False:
                df.loc[df['houspan'].isnull(), 'houspan'] = df['houschn']

            # -------------
            # pipeline Enet:
            # -------------
            model = "Enet"

            # # additional preprocess steps:

            # define X and y:
            y = df[dependent_variable]
            X = df.drop(dependent_variable, axis=1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}, X train p={}; mean={},\ny test n={}, X test p={}; mean={}\n".format(y_train.shape[0], X_train.shape[1],
                                                                                      round(y_train.mean(), 2),
                                                                                      y_test.shape[0], X_test.shape[1],
                                                                                      round(y_test.mean(), 2)),
                  file=open(save_file, "a"))

            # save data sets:
            modelling_data_save = analysis_path + "Modelling_Data/"
            X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
            X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
            pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
            pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))
            # save col list:
            pd.DataFrame(X_train.columns).to_csv(modelling_data_save + "Col_list/df{}_cols.csv".format(df_num))

            # -------
            # pipeline steps:
            # ^note: not converging so in final run set max_iterator higher


            print("\n----------Elastic net----------", file=open(save_file, "a"))
            print("params tried:\n{}\n".format(enet_params), file=open(save_file, "a"))


            # pipeline construction
            pipe, preprocessor = build_pipeline_enet(X=X, imputer_model=imputer_model,
                                                oh_encoder_drop='first',
                                                imputer_max_iter=imputer_max_iter)

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
            param_df.to_csv(
                analysis_path + 'Outputs/Modelling/Grid_Search/Enet/' + 'df{}_{}.csv'.format(df_num, start_string))

            print("Elastic Net:")
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
            print("Best training {} score: {}. Best model params:\n{}.\n".format(train_scoring, best_train_score,
                                                                                 best_params),
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
            lm_results_dict = {'r2': test_r2_score, "var_expl": test_var_expl_score, "MAE": test_mae_score,
                               "RMSE": test_rmse_score}
            lm_scores_dict[df_num] = lm_results_dict

            # save best params as a pickle:
            params_save = analysis_path + "Results/Best_params/Enet/"
            joblib.dump(best_params, params_save + 'df{}__{}.pkl'.format(df_num, start_string), compress=1)

            if check_imputed_cors == True:
                check_imputed_cors_across_blocks(X=X_train, analysis_path=analysis_path,
                                                 block_file=block_file,
                                                 corr_print_threshold=corr_print_threshold,
                                                 df=df, df_num=df_num, imp_method=imputer_model,
                                                 pipe=pipe, model=model, preprocessor=preprocessor)
                # check missing data
                check_miss(X_train, analysis_path, df_num, model)

            preprocessor.fit(X_train)
            X_train_arr = preprocessor.transform(X_train)
            print('number of features after coding = {}'.format(np.shape(X_train_arr)[1]))

            # -------------
            # pipeline RF:
            # -------------
            print("\n----------Random forest----------", file=open(save_file, "a"))
            model = "RF"
            # define X and y:
            y = df[dependent_variable]
            X = df.drop(dependent_variable, axis=1)

            # pipeline construction
            pipe, preprocessor = build_pipeline_rf(X=X, imputer_model=imputer_model,
                                                   oh_encoder_drop='if_binary',
                                                   imputer_max_iter=imputer_max_iter)


            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
            print("y train n={}, X train p={}; mean={},\ny test n={}, X test p={}; mean={}\n".format(y_train.shape[0],
                                                                                      X_train.shape[1],
                                                                                      round(y_train.mean(), 2),
                                                                                      y_test.shape[0], X_test.shape[1],
                                                                                      round(y_test.mean(), 2)),
                  file=open(save_file, "a"))

            # save data sets:
            modelling_data_save = analysis_path + "Modelling_Data/"
            X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
            X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
            pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
            pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))

            print("params tried:\n{}\n".format(rf_params), file=open(save_file, "a"))

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
            param_df.to_csv(
                analysis_path + 'Outputs/Modelling/Grid_Search/RF/' + 'df{}_{}.csv'.format(df_num, start_string))

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

            print("Best training {} score: {}. Best model params:\n{}.\n".format(train_scoring, best_train_score,
                                                                                 best_params),
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

            if check_imputed_cors == True:
                # Get processed RF data:
                X_train_tr = preprocess_transform(X, X_train, pipe, preprocessor)

                # logistic regression and sges_T1, IQ checks
                eval_imputed_track_cors(X_train=X_train_tr, analysis_path=analysis_path,
                                        df_num=df_num, cont_vars=["kftiq", dv_T1],
                                        var_names=["IQ", "Sges Time 1"], X_train_orig=X_train,
                                        xticks=[np.arange(-5, -5, 0.1), np.arange(-10, -10, 0.1)])
            # ===========================
            if compare_DV_T1_baseline == True:
                X_base_train = X_train[dv_T1]
                X_base_test = X_test[dv_T1]
                # DV_T1_index = X_train.columns.get_loc(dv_T1)
            if compare_non_survey_baseline == True:
                block_dict = parse_data_to_dict(block_file)
                all=False
                if all == True:
                    ns_vars = block_dict['Sges_T1'] + block_dict['Grades'] + block_dict['School_Track']\
                              + block_dict['Demographics_and_SES'] + block_dict['IQ']
                else:
                    ns_vars = block_dict['Sges_T1'] + block_dict['School_Track']
                selected_ns_vars = []
                for var in ns_vars:
                    if var in X_train.columns:
                        selected_ns_vars.append(var)
                X_ns_base_train = X_train[selected_ns_vars]
                X_ns_base_test = X_test[selected_ns_vars]

            # compare to mean baseline measure

            dummy_regr = DummyRegressor(strategy="mean")
            dummy_regr.fit(X_train, y_train)

            preprocessor.fit(X_train)
            X_test = preprocessor.transform(X_test)
            # todo: save transformed data here?

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

            # Compare to DV_T1 only as baseline:
            if compare_DV_T1_baseline == True:
                lin_regr = LinearRegression()

                lin_regr.fit(np.array(X_base_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))

                lin_regr_y_pred = lin_regr.predict(np.array(X_base_test).reshape(-1, 1))

                lin_regr_r2_score = round(metrics.r2_score(y_test, lin_regr_y_pred), decimal_places)
                lin_regr_expl_score = round(metrics.explained_variance_score(y_test, lin_regr_y_pred), decimal_places)
                lin_regr_mae_score = round(metrics.mean_absolute_error(y_test, lin_regr_y_pred), decimal_places)
                lin_regr_rmse_score = round(metrics.mean_squared_error(y_test, lin_regr_y_pred, squared=False),
                                            decimal_places)

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

            if compare_non_survey_baseline == True:
                lin_regr_ns = LinearRegression()
                preprocessor = build_preprocessor_ns_base(X=X_ns_base_train,
                                                 oh_encoder_drop='if_binary',
                                                 imputer_max_iter=imputer_max_iter)

                X_ns_base_train = preprocessor.fit_transform(X_ns_base_train)
                lin_regr_ns.fit(X_ns_base_train, y_train)
                # preprocessor.fit(X_ns_base_train)
                X_ns_base_test = preprocessor.transform(X_ns_base_test)
                lin_regr_ns_y_pred = lin_regr_ns.predict(X_ns_base_test)

                lin_regr_r2_score = round(metrics.r2_score(y_test, lin_regr_ns_y_pred), decimal_places)
                lin_regr_expl_score = round(metrics.explained_variance_score(y_test, lin_regr_ns_y_pred), decimal_places)
                lin_regr_mae_score = round(metrics.mean_absolute_error(y_test, lin_regr_ns_y_pred), decimal_places)
                lin_regr_rmse_score = round(metrics.mean_squared_error(y_test, lin_regr_ns_y_pred, squared=False),
                                            decimal_places)

                print("\n----------Comparisons----------", file=open(save_file, "a"))
                print("NS base model performance on test data: \n"
                      " R2: {}\n Variance Explained: {}\n MAE: {} \n RMSE: {}".format(lin_regr_r2_score,
                                                                                      lin_regr_expl_score,
                                                                                      lin_regr_mae_score,
                                                                                      lin_regr_rmse_score),
                      file=open(save_file, "a"))

                # save score to dict:
                NS_base_results_dict = {'r2': lin_regr_r2_score, "var_expl": lin_regr_expl_score,
                                      "MAE": lin_regr_mae_score, "RMSE": lin_regr_rmse_score}

                NS_base_scores_dict[df_num] = NS_base_results_dict
            # ----------------------------------------------------------------------------------------------------------
            # statistically compare errors:
            err_save_path = "{}/Results/Post_hoc/Err_dist/".format(analysis_path)
            bins = 25
            err_dum = np.log10(abs(y_test - dummy_y_pred))
            plot = "log10_mae_dummy"
            plot_hist(save_name=plot + "_df{}".format(df_num), x=err_dum, save_path=err_save_path,
                      title=plot, bins=bins)

            if compare_DV_T1_baseline == True:
                err_lin_regr = np.log10(abs(np.array(y_test).reshape(-1, 1) - lin_regr_y_pred))
                err_lin_regr = pd.Series(err_lin_regr.reshape((err_lin_regr.shape[0],)))
                plot = "log10_mae_lin_regr_y"
                plot_hist(save_name=plot + "_df{}".format(df_num), x=err_lin_regr, save_path=err_save_path,
                          title=plot, bins=bins)

            err_lm = np.log10(abs(y_test - y_pred_lm))
            plot = "log10_mae_enet"
            plot_hist(save_name=plot + "_df{}".format(df_num), x=err_lm, save_path=err_save_path,
                      title=plot, bins=bins)

            err_rf = np.log10(abs(y_test - y_pred_rf))
            plot = "log10_mae_rf"
            plot_hist(save_name=plot + "_df{}".format(df_num), x=err_rf, save_path=err_save_path,
                      title=plot, bins=bins)

            t_lm, p_lm = scipy.stats.ttest_rel(err_dum, err_lm, axis=0, nan_policy='propagate')
            t_rf, p_rf = scipy.stats.ttest_rel(err_dum, err_rf, axis=0, nan_policy='propagate')
            t_c, p_c = scipy.stats.ttest_rel(err_lm, err_rf, axis=0, nan_policy='propagate')

            if compare_DV_T1_baseline == True:
                t_lr_lm, p_lr_lm = scipy.stats.ttest_rel(err_lin_regr, err_lm, axis=0, nan_policy='propagate')
                t_lr_rf, p_lr_rf = scipy.stats.ttest_rel(err_lin_regr, err_rf, axis=0, nan_policy='propagate')
                t_lr_d, p_lr_d = scipy.stats.ttest_rel(err_dum, err_lin_regr, axis=0, nan_policy='propagate')

            print("t-test LM and dummy: p={}, t={}".format(f'{p_lm:.4f}', f'{t_lm:.4f}'), file=open(save_file, "a"))
            print("t-test RF and dummy: p={}, t={}".format(f'{p_rf:.4f}', f'{t_rf:.4f}'), file=open(save_file, "a"))
            print("t-test LM and RF: p={}, t={}".format(f'{p_c:.4f}', f'{t_c:.4f}'), file=open(save_file, "a"))

            if compare_DV_T1_baseline == True:
                print("t-test {} and LM: p={}, t={}".format(dv_T1, f'{p_lr_lm:.4f}', f'{t_lr_lm:.4f}'),
                      file=open(save_file, "a"))
                print("t-test {} and RF: p={}, t={}".format(dv_T1, f'{p_lr_rf:.4f}', f'{t_lr_rf:.4f}'),
                      file=open(save_file, "a"))
                print("t-test dummy and {}: p={}, t={}".format(dv_T1, f'{p_lr_d:.4f}', f'{t_lr_d:.4f}'),
                      file=open(save_file, "a"))

            df_end_time = dt.datetime.now()
            df_run_time = df_end_time - start_df
            print('\n----------Total df run time----------\n{}'.format(df_run_time), file=open(save_file, "a"))

            continue
        else:
            continue

    # save dicts as dataframes:
    overall_save = analysis_path + "Results/Performance_all/"
    dummy_results_df = pd.DataFrame.from_dict(dummy_scores_dict, orient='index').reset_index()
    dummy_results_df.to_csv(overall_save + "Dummy_{}.csv".format(start_string))
    lm_results_df = pd.DataFrame.from_dict(lm_scores_dict, orient='index').reset_index()
    lm_results_df.to_csv(overall_save + "ENet_{}.csv".format(start_string))
    rf_results_df = pd.DataFrame.from_dict(rf_scores_dict, orient='index').reset_index()
    rf_results_df.to_csv(overall_save + "RF_{}.csv".format(start_string))
    if compare_DV_T1_baseline == True:
        dv_t1_results_df = pd.DataFrame.from_dict(DV_T1_scores_dict, orient='index').reset_index()
        dv_t1_results_df.to_csv(overall_save + "DV_T1_{}.csv".format(start_string))
    if compare_non_survey_baseline == True:
        ns_base_results_df = pd.DataFrame.from_dict(NS_base_scores_dict, orient='index').reset_index()
        ns_base_results_df.to_csv(overall_save + "NS_base_{}.csv".format(start_string))

    return












