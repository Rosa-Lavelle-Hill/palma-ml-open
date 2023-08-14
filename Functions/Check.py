import pandas as pd


def all_vars_in_block(df, blockdict, print=True):
    vars_not_in = {}
    for col in df.columns:
        in_code = 0
        for dict, list in blockdict.items():
            if col in list:
                in_code = in_code + 1
        vars_not_in[col] = in_code
    if print == True:
        print("**************** variables not in block ****************")
        print(vars_not_in)
    return vars_not_in


def check_miss(X_train, analysis_path, df_num, model):
    count_nonna = X_train.count(axis=0)
    perc_nonna = round((X_train.count(axis=0) / X_train.shape[0]) * 100, 2)
    nona_df = pd.DataFrame(pd.concat([count_nonna, perc_nonna], axis=1)).reset_index()
    nona_df.columns = ["Variable", "Number_Students", "Percentage_Students"]
    nona_df.sort_values(by=["Percentage_Students"], axis=0, ascending=True, inplace=True)
    miss_save_path = analysis_path + "Outputs/Descriptives/Missing_Data/"
    nona_df.to_csv(miss_save_path + "modelling_df{}_{}.csv".format(df_num, model))
    print("xtrain len= {}; count df{} len = {}".format(X_train.shape[1], df_num, nona_df.shape[0]))