import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from Functions.Preprocessing import get_var_names_dict

a2_cols = pd.read_csv("Analysis_2_with_sges_T1/Modelling_Data/Col_list/df1_cols.csv")
predx_cols = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Modelling_Data/Col_list/df1_cols.csv")

a2_list = list(a2_cols['0'])
predx_list = list(predx_cols['0'])

# which a2 not in predx
not_in = list(set(a2_list) - set(predx_list))

# train
a2_train1 = pd.read_csv("Analysis_2_with_sges_T1/Modelling_Data/Enet/df1_X_train.csv")
predx_train1 = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Modelling_Data/Enet/df1_X_train.csv")

# transformed
predx_train1_tr = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Modelling_Data/Enet/Transformed_preprocessor/df1_X_test_transformed.csv")

# double check if 5->6 dfs are the same:

# a) before imputed (train)
predx_col_order = list(predx_train1.columns)
a2_train1 = a2_train1[predx_col_order]
a2_train1.fillna(-9999)
predx_train1.fillna(-9999)
are_equal = a2_train1.equals(predx_train1)
if are_equal == True:
    print('dfs are the same')
else:
    print("dfs not the same")

# b) after imputed (test)
predx_test1 = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Modelling_Data/Enet/Transformed_preprocessor/df1_X_test_transformed.csv", index_col=[0])
a2_test1 = pd.read_csv("Analysis_3_with_sges_T1/Modelling_Data/Enet/Transformed_preprocessor/df1_X_test_transformed.csv", index_col=[0])
print(list(predx_test1.columns))
print(list(a2_test1.columns))

predx_col_order = list(predx_test1.columns)
a2_test1 = a2_test1[predx_col_order]
a2_test1.fillna(-9999)
predx_test1.fillna(-9999)
are_equal = round(a2_test1, 2).equals(round(predx_test1, 2))
if are_equal == True:
    print('dfs are the same')
else:
    print("dfs not the same")
# ^ checked and basically the same but due to imputation and scaling, still minor differences -> 99% same to 1 dp

# check how many features before dummy coding = 88
print(predx_train1.shape)
print(a2_train1.shape)

# substitute short names for long names in block df for supplementary materials
blocked_vars = pd.read_csv("Data/MetaData/Variable_blocks/final_blocked_vars.csv", index_col=[0])
blocked_vars = blocked_vars.T

names_df = pd.read_csv("Data/MetaData/variable_names.csv")
names_dict = get_var_names_dict(names_df)
var_names_df = pd.DataFrame.from_dict(names_dict, orient="index").reset_index()
var_names_df.columns = ['short_name', 'long_name']

cols = list(blocked_vars.columns)
for index, row in blocked_vars.iterrows():
    for col in cols:
        new_row = row.replace(names_dict)
        blocked_vars.at[index] = new_row

blocked_vars.rename(columns=names_dict, inplace=True)
blocked_vars.to_csv("Data/MetaData/Variable_blocks/final_blocked_vars_new_names.csv")

print('done')