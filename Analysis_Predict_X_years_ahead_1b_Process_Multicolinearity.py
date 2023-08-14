import re
import os
import pandas as pd
from Functions.Correlations import get_top_abs_correlations
from Functions.Preprocessing import multicollin_changes
from RUN_Analysis_2_New import multicolinear_threshold
from RUN_Predict_X_years_ahead_New import version, analysis_path, analysis

data_load_path = analysis_path + "Processed_Data/"
directory_bits = os.fsencode(data_load_path)
corr_save_path = '{}/Outputs/Descriptives/Correlations/Full_Correlation_Matrices/'.format(analysis)
data_save_path = analysis_path + "Processed_Data/Processed_Multicollinearity/"

print('Creating correlation matrices...')

# loop through dfs:
for file in os.listdir(directory_bits):
    filename = os.fsdecode(file)
    if filename.endswith("_preprocessed{}.csv".format(version)):
        print("--> Processing.... {}".format(filename))
        df_num = [s for s in re.findall(r'\d+', filename)][0]

        df = pd.read_csv(data_load_path+filename, index_col=False)
        df.drop(["Unnamed: 0"], axis=1)

        print('dealing with correlated variables...')

        df = multicollin_changes(df)

        # create cor matrix
        corr_m = round(df.corr(), 2)

        # save correlation matrix
        corr_m.to_csv(corr_save_path + "correlation_matrix_df_{}.csv".format(df_num))

        # save newly processed data:
        df.to_csv(data_save_path + "df{}_preprocessed{}.csv".format(df_num, version))

        # get correlations left above threshold:
        small_df = df.copy()
        drops = ["Unnamed: 0", "Unnamed: 0.1", "School_Code", "School_Class"]

        for var in small_df.columns:
                if var.endswith('_S') or var.endswith('_C'):
                    small_df.drop(var, inplace=True, axis=1)
                elif var in drops:
                    small_df.drop(var, inplace=True, axis=1)
                else:
                    continue
        pairs_save_path = "{}/Outputs/Descriptives/Correlations/Remaining_Correlation_Pairs/".format(analysis)
        cors_above_threshold = get_top_abs_correlations(small_df, threshold=multicolinear_threshold)
        cors_above_threshold.to_csv(pairs_save_path + "Correlation_pairs_df_{}.csv".format(df_num))

print('done!')