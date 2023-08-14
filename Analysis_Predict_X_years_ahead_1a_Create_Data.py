import itertools
import os
import re
import pandas as pd
import numpy as np
import datetime as dt

from Functions.Correlations import get_top_abs_correlations, get_top_DV_abs_correlations
from Functions.Preprocessing import remove_cols
from Functions.RF_preprocess import to_categorical
from Functions.Plotting import plot_scatt
from RUN_Predict_X_years_ahead_New import version, Additional_pred_checks, analysis_path
from Fixed_params import cat_vars, drop_list
from scipy.stats import pearsonr

correlation_checks = False
include_grade_4_vars = False
preprocess_drop_DV_T1 = False
check_DV_descriptives = False

dependent_variable = "sges"
# multicolinear_threshold = 0.05
decimal_places = 4
missing_data_col_cutoff = 50
dv_T1 = dependent_variable+'_T1'

save_new_data_path = "{}Processed_Data/".format(analysis_path)
# reconstruct predict X years ahead data from original waves

if __name__ == "__main__":
    print("Creating data...")
    # load wave data and create a dictionary:
    load_wave_data_path = "Data/Waves/"
    directory_bits = os.fsencode(load_wave_data_path)
    wave_df_dict = {}
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            print("--> Loading.... {}".format(filename))
            wave_num = [s for s in re.findall(r'\d+', filename)][0]

            if int(wave_num) > 5:
                continue

            df = pd.read_csv(load_wave_data_path + filename, index_col=[0])
            wave_df_dict[wave_num] = df

    load_vars_path = "Data/Initial_Preprocess/"
    overlap_cols = pd.read_csv(load_vars_path + "final_variables.csv")
    overlap_cols = list(overlap_cols['0'])

    # create separate df for time constant vars:
    data_path = "Data/"
    construct_save_path = "Data/MetaData/Constructs/"
    time_constant_missing = pd.read_csv(construct_save_path + "Time_Constant_Constructs.csv")
    time_constant_vars = list(time_constant_missing["Variable"])

    # add wave col to each df
    for wave in [1, 2, 3, 4, 5]:
        df = wave_df_dict[str(wave)]
        df['Wave'] = str(wave)
        wave_df_dict[str(wave)] = df

    # add any other cols we want to list here:
    cols = ['Wave'] + overlap_cols

    # load time invariant data:
    df_time_constant = pd.read_csv("Data/Time_Constant_All.csv", index_col=False)
    # save aggregate level vars for later use:
    df_aggregate = df_time_constant[["vpnr", "School_Code", "Class_Code"]]
    # drop those not needed:
    df_time_constant.drop(["School_Code", "Participant_Num", 'Class_Code', 'School_Year'],
                          axis=1, inplace=True)

    # open doc to record droppped vars for each df:
    dropped_vars_missing_path = "{}Outputs/Preprocessing/".format(analysis_path)
    print("Dropped variables due to missing data... ",
          file=open(dropped_vars_missing_path + "Dropped_vars.txt", "w"))

    # create 4 different long dataframes for predicting X years ahead analysis (wave 1+2, 1+3 etc.)
    long_dfs_dict = {}
    for x_num, y_num in zip([1, 2, 3, 4], [2, 3, 4, 5]):
        print("concating df1 and df{}".format(y_num))
        df_long = pd.concat([wave_df_dict['1'][cols], wave_df_dict[str(y_num)][cols]], axis=0)
        df_long.sort_values(by=['vpnr', 'Wave'], inplace=True)
        long_dfs_dict[x_num] = df_long

    # only keep rows where ids in both waves:
    counter = 1
    drop_var_list = []
    for long_df_num, long_df in long_dfs_dict.items():

        # see how many unique individuals in both waves with complete info:
        long_df["Wave"] = long_df["Wave"].astype(int)

        # count where count==2 (have data from both wave 1 and 2)
        counts = long_df.loc[:, ['Wave', 'vpnr']].groupby("vpnr").count()
        counts = counts[counts["Wave"] == 2]
        list_comp_ids = list(counts.index)

        # save dfs with complete ids:
        df = long_df[long_df["vpnr"].isin(list_comp_ids)]
        n_comp = len(df["vpnr"].unique())

        # define X and y:
        x_wave = 1
        y_wave = int(x_wave) + counter

        y = df["sges"][df["Wave"] == y_wave]

        if preprocess_drop_DV_T1 == True:
            Xi = df.drop("sges", axis=1)[df["Wave"] == x_wave]
        if preprocess_drop_DV_T1 == False:
            Xi = df[df["Wave"] == x_wave]
            Xi.rename(columns={dependent_variable: dv_T1}, inplace=True)

        X = Xi.merge(df_time_constant, on="vpnr")

        # drop for now (Meta):
        drop_meta = ["Wave", "filter_$", "agemon", "ageyear", 'index']
        df = remove_cols(df, drop_meta)

        # only keep grade variables that allow forward time prediction:
        drops_list = []
        keep_list = []
        # go through time constant columns:
        for var in df_time_constant.columns:
            if include_grade_4_vars == True:
                # keep in (ignore) if a 4 (primary school grade) or ends with '12345':
                if (var.endswith('4')) or (var.endswith('12345')):
                    keep_list.append(var)
                else:
                    continue
            else:
                # does end with digit?:
                is_digit = var[-1].isdigit()
                # if doesn't end with digit then continue
                if is_digit == False:
                    continue
                else:
                    grade_num = int(var[-1])
                    # only keep vars where final number is ==x_wave +4 (i.e. grade 5 IVs should be from)
                    if grade_num != x_wave + 4:
                        drops_list.append(var)
                    else:
                        keep_list.append(var)
        # drop rest
        X.drop(drops_list, inplace=True, axis=1)

        # drop some IVs due to too much missingness:
        missing_sum = pd.DataFrame(X.isna().sum())
        missing_perc = round((missing_sum / len(X)) * 100, 2)
        missing_summary = pd.concat([missing_sum, missing_perc], axis=1)
        missing_summary.reset_index(inplace=True)
        missing_summary.columns = ["Variable", "Missing_Sum", "Missing_Perc"]
        missing_summary.sort_values(by='Missing_Perc', ascending=False, inplace=True)

        # drop vars with missing data > cutoff
        drop_vars_perc = missing_summary[["Variable", "Missing_Perc"]][
            missing_summary["Missing_Perc"] >= missing_data_col_cutoff]
        drop_vars = missing_summary["Variable"][missing_summary["Missing_Perc"] >= missing_data_col_cutoff]
        drop_var_list.append(list(drop_vars.values))
        X.drop(drop_vars, axis=1, inplace=True)
        print("DF{}: Dropped {} variables due to missing data. {} variables remaining\n"
              "Dropped variables: {}".format(long_df_num, len(drop_vars), X.shape[1], drop_vars_perc),
              file=open(dropped_vars_missing_path + "Dropped_vars.txt", "a"))

        # merge and save X and y
        y = pd.Series(y).reset_index()
        X_and_y = pd.concat([X, y], axis=1)

        # remove more vars
        drop_list = ["Unnamed: 0", "index", "laeng12345", "vpnrg_u", "agemon", "startc", "starty",
                     "laeng12345_C", "laeng12345_S", "filter_$", "ageyear", "Wave", "gbi_jz"]
        for col in X_and_y.columns:
            if col in drop_list:
                X_and_y.drop(col, axis=1, inplace=True)
                print("dropping {}".format(col))
            # rename grade vars (remove number):
            if any(i.isdigit() for i in col):
                if col == "sges_T1":
                    continue
                col_new = ''.join([i for i in col if not i.isdigit()])
                X_and_y.rename({col: col_new}, inplace=True, axis=1)

        # Merge high cardinal feature categories
        X_and_y['houspan'][X_and_y["houspan"] > 9] = 999
        X_and_y['houschn'][X_and_y["houschn"] > 9] = 999

        # initial save
        X_and_y.to_csv(save_new_data_path + "df{}_preprocessed{}.csv".format(long_df_num, version))
        counter = counter + 1
        # --------------------

    # Loop through and if a var dropped for missingness on one df, drop on others (so fair comparisons over time)
    drop_var_list_flat = list(itertools.chain(*drop_var_list))
    drop_var_list_flat = list(set(drop_var_list_flat))
    directory_bits = os.fsencode(save_new_data_path)
    save_path_df_info = "Outputs/Descriptives/".format(analysis_path)
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith("_preprocessed{}.csv".format(version)):
            print("--> Processing... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]
            df = pd.read_csv(save_new_data_path + filename)
            for drop_col in drop_var_list_flat:
                drop_col_start = drop_col.split('_')[0]
                for col in df:
                    if str(col).startswith(drop_col_start):
                        df.drop(col, axis=1, inplace=True)
                    else:
                        continue

            # replace data with new data (with columns dropped due to missings
            print("df{} shape: {}".format(df_num, df.shape))
            df.to_csv(save_new_data_path + "df{}_preprocessed{}.csv".format(df_num, version))

            # save feature df:
            cols = pd.DataFrame(df.columns.values)
            cols.to_csv(analysis_path + save_path_df_info + "IV_Cols/" + "df{}_cols.csv".format(df_num))

            # print n and p for each df:
            save_file = "Descriptives_df{}.txt".format(df_num)
            n = df.shape[0]
            p = int(df.shape[1]) - 1
            print("df {} descriptives :\nn = {}; p = {}".format(df_num, n, p),
                  file=open(analysis_path + save_path_df_info + save_file, "w"))

            # final check of IV cors
            if check_DV_descriptives == True:
                # check sges correlations between DV_T1 and DV:
                grade_IV_dict = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9}
                x_grade = grade_IV_dict[int(df_num)]
                y_grade = x_grade + 1
                print("df{}: x= grade {}; y= grade {}".format(df_num, x_grade, y_grade))
                plot_scatt(x=df[dv_T1],
                           y=df[dependent_variable],
                           save_path="{}Outputs/Descriptives/DV_Cors/".format(analysis_path),
                           save_name="FINAL_Grade_{}_and_grade_{}".format(x_grade, y_grade),
                           xlab="Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                           fontsize=12)

                if df_num == '1':
                    x_grade = 5
                    print("---------------- final checks ------------------")
                    for pred_years_ahead in [1, 2, 3, 4]:
                        # check sges correlations between waves:
                        df_x = df.copy()
                        if pred_years_ahead == 1:
                            y_num = df_num
                            y_grade = x_grade + pred_years_ahead
                            print("x= df{} (grade {}); y= df{} (grade {})".format(df_num, x_grade, y_num, y_grade))
                            print("N={}".format(df_x.shape[0]))
                            plot_scatt(x=df_x[dv_T1],
                                       y=df_x[dependent_variable],
                                       save_path="{}Outputs/Descriptives/Final_DV_cor_check/".format(analysis_path),
                                       save_name="FINAL_Grade_{}_and_grade_{}".format(x_grade, y_grade),
                                       xlab="Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                                       fontsize=12)
                        else:
                            y_num = pred_years_ahead
                            df_y = pd.read_csv(save_new_data_path + "df{}_preprocessed{}.csv".format(y_num, version))
                            y_grade = x_grade + pred_years_ahead
                            x = df_x[dv_T1]
                            y = df_y[dependent_variable]
                            print("x= df{} (grade {}); y= df{} (grade {})".format(df_num, x_grade, y_num, y_grade))
                            if x.shape[0] != y.shape[0]:
                                # remove where both have NAs
                                join_df = df_x[['vpnr', dv_T1]].merge(df_y[['vpnr', dependent_variable]],
                                                                      how='inner', on="vpnr")
                                x = join_df[dv_T1]
                                y = join_df[dependent_variable]
                                print("N={}".format(join_df.shape[0]))

                            plot_scatt(x=x,
                                       y=y,
                                       save_path="{}Outputs/Descriptives/Final_DV_cor_check/".format(analysis_path),
                                       save_name="FINAL_Grade_{}_and_grade_{}".format(x_grade, y_grade),
                                       xlab="Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                                       fontsize=12)
                        print("-----------")


print('data created!')


