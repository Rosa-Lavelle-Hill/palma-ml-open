import os
import re
import pandas as pd
import numpy as np
import datetime as dt

from Functions.Correlations import get_top_abs_correlations, get_top_DV_abs_correlations
from Functions.RF_preprocess import to_categorical
from Functions.Plotting import plot_scatt
from RUN_Predict_X_years_ahead import version, original_data_load_path, data_save_path, Additional_pred_checks, analysis_path
from Fixed_params import cat_vars, drop_list
from scipy.stats import pearsonr

dependent_variable = "sges"
correlation_checks = True
multicolinear_threshold = 0.05
original_method = True

# set save paths and directories

if __name__ == "__main__":
    decimal_places = 4
    dv_T1 = dependent_variable+'_T1'

    dfs_dict = {}
    # loop through dfs:
    if original_method == True:
        directory_bits = os.fsencode(original_data_load_path)
        for file in os.listdir(directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("_preprocessed{}.csv".format(version)):
                start_df = dt.datetime.now()
                print("--> Processing.... {}".format(filename))
                df_num = [s for s in re.findall(r'\d+', filename)][0]

                df = pd.read_csv(original_data_load_path+filename)

                # code categorical vars:
                df = to_categorical(df, cat_vars)

                print("df num {} shape: {}".format(df_num, str(df.shape)))

                for col in df.columns:
                    if col in drop_list:
                        df.drop(col, axis=1, inplace=True)

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
            print("df{}: X={}, y={}".format(df_num, str(df_num), str(df_num + 1)))
        else:
            y_future_num = y_future_num + 1
            predict_ahead_dfs_dict[df_num] = {'X': dfs_dict['1']['X'],
                                              'y': dfs_dict[str(y_future_num)]['y']}
            print("df{}: X={}, y={}".format(df_num, str(1), str(y_future_num)))

        X_and_y = pd.concat([predict_ahead_dfs_dict[df_num]['X'],
                             pd.DataFrame(predict_ahead_dfs_dict[df_num]['y'])], axis=1)

        # remove rows with no y (as don't want to impute). In this analysis N will drop off over time.
        X_and_y.dropna(axis=0, inplace=True, subset=[dv_T1, dependent_variable])
        # save
        X_and_y.to_csv(data_save_path+"predict_ahead_df{}.csv".format(str(df_num)))

        # for each df check cor between DV and DV_T1 (grade 5 DV)
        y_dict = {1:6, 2:7, 3:8, 4:9}
        if correlation_checks == True:
            save_path = "{}Outputs/Descriptives/Correlations/DV_Time_cor_plots/".format(analysis_path)
            save_name = "df{}_cor_DVT1_and_DV".format(df_num)
            plot_scatt(x=dv_T1, y=dependent_variable, xlab="Maths Ability Grade 5",
                       ylab="Maths Ability Grade {}".format(y_dict[df_num]),
                       save_path=save_path, save_name=save_name, data=X_and_y)

            # create IV cor matrix:
            df_numeric = X_and_y.drop(cat_vars, axis=1)
            corr_m = round(df_numeric.corr(), 2)
            # save correlation matrix
            save_matrix_path = "{}Outputs/Descriptives/Correlations/All_Ivs/Matrix/".format(analysis_path)
            corr_m.to_csv(save_matrix_path + "correlation_matrix_df_{}.csv".format(df_num))

            # all pairs:
            pairs_save_path = "{}Outputs/Descriptives/Correlations/All_Ivs/".format(analysis_path)
            cors_above_threshold = get_top_abs_correlations(df_numeric, threshold=multicolinear_threshold)
            cors_above_threshold.to_csv(pairs_save_path + "Correlation_pairs_df_{}.csv".format(df_num))

            # check all cors of IV with DV:
            DV_pairs_save_path = "{}Outputs/Descriptives/Correlations/All_Ivs_with_DV/".format(analysis_path)
            DV_cors_above_threshold = get_top_DV_abs_correlations(df_numeric, threshold=multicolinear_threshold,
                                                                  DV=dependent_variable)
            DV_cors_above_threshold.to_csv(DV_pairs_save_path + "DV_Correlation_pairs_df_{}.csv".format(df_num))

            # check maths ability and school type (cat var); and DV and school track
            Track_pairs_save_path = "{}Outputs/Descriptives/Correlations/All_IVs_with_Track/".format(analysis_path)
            X_and_y_factor = X_and_y.copy()
            X_and_y_factor["sctyp"] = pd.factorize(X_and_y_factor["sctyp"])[0]
            Track_cors_above_threshold = get_top_DV_abs_correlations(X_and_y_factor,
                                                                  threshold=multicolinear_threshold,
                                                                  DV="sctyp")
            Track_cors_above_threshold.to_csv(Track_pairs_save_path + "Track_Correlation_pairs_df_{}.csv".format(df_num))

            # Track and DV_T1:
            save_path = "{}Outputs/Descriptives/Correlations/Track_cors/".format(analysis_path)
            save_name = "df{}_cor_Track_and_DVT1".format(df_num)
            plot_scatt(x='sctyp', y=dv_T1, xlab="School Track",
                       ylab="Maths Ability Grade 5", data=X_and_y_factor,
                       save_path=save_path, save_name=save_name)

            # Track and DV
            save_path = "{}Outputs/Descriptives/Correlations/Track_cors/".format(analysis_path)
            save_name = "df{}_cor_Track_and_DV".format(df_num)
            plot_scatt(x='sctyp', y=dependent_variable, data=X_and_y_factor, xlab="School Track",
                       ylab="Maths Ability Grade {}".format(y_dict[df_num]),
                       save_path=save_path, save_name=save_name)
            X_and_y_factor["sctyp"].replace([1, 2, 3], ['1', '2', '3'], inplace=True)

            # Track and Parental_degree_expectations
            save_path = "{}Outputs/Descriptives/Correlations/Track_cors/".format(analysis_path)
            save_name = "df{}_cor_Track_and_parental_deg_exp".format(df_num)
            plot_scatt(x='sctyp', y="parental_expectation_degree", data=X_and_y_factor, xlab="School Track",
                       ylab="Maths Ability Grade {}".format(y_dict[df_num]),
                       save_path=save_path, save_name=save_name)




    if Additional_pred_checks == True:
        # also need data for 6->8, 6->9, 7->9:
        data_save_path = analysis_path + "Processed_Data/Additional_pred_ahead/"
        add_pred_ahead_dfs = {}
        for df_num in [1, 2, 3, 4]:
            if df_num == 2:
                add_pred_ahead_dfs['6->8'] = {'X': dfs_dict[str(df_num)]['X']}
                add_pred_ahead_dfs['6->9'] = {'X': dfs_dict[str(df_num)]['X']}
            if df_num == 3:
                add_pred_ahead_dfs['7->9'] = {'X': dfs_dict[str(df_num)]['X']}
                add_pred_ahead_dfs['6->8']['y'] = dfs_dict[str(df_num)]['y']
            if df_num == 4:
                add_pred_ahead_dfs['6->9']['y'] = dfs_dict[str(df_num)]['y']
                add_pred_ahead_dfs['7->9']['y'] = dfs_dict[str(df_num)]['y']

        for df_name, df in add_pred_ahead_dfs.items():
            X_and_y = pd.concat([add_pred_ahead_dfs[df_name]['X'],
                                 pd.DataFrame(add_pred_ahead_dfs[df_name]['y'])], axis=1)

            # remove rows with no y (as don't want to impute). In this analysis N will drop off over time.
            X_and_y.dropna(axis=0, inplace=True, subset=[dv_T1, dependent_variable])
            print("df {} shape: {}".format(df_name, str(X_and_y.shape)))
            # save
            df_name = df_name.replace('->', '_')
            X_and_y.to_csv(data_save_path + "predict_ahead_df{}.csv".format(str(df_name)))

            if correlation_checks == True:
                # for each df check cor between DV and DV_T1 (grade 5 DV)

                save_path = "Predict_X_years_ahead_with_sges_T1/Outputs/Descriptives/Correlations/DV_Time_cor_plots/"
                save_name = "df{}_cor_DVT1_and_DV".format(df_name)

                plot_scatt(x=X_and_y[dv_T1], y=X_and_y[dependent_variable], xlab= "Maths Ability Grade 5",
                           ylab="Maths Ability Grade {}".format(df_name[1]),
                           save_path=save_path, save_name=save_name)

#

print('data created!')


