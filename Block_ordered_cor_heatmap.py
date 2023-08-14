import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Plotting import annotate_yranges, plot_corr, plot_corr_blocks, plot_corr_blocks_2col, plot_corr_blocks_HH
from RUN_Analysis_2_New import final_data_path
from Fixed_params import block_file, drop_p_degree_expect

dependent_variable="sges"

cor_dict = {}
dfs = [1, 2, 3, 4]
for df_num in dfs:
    df = pd.read_csv(final_data_path + "df{}_preprocessed_with_sges_T1.csv".format(df_num), index_col=[0])

    if df_num ==5:
        df.drop("pcb_jz", axis=1, inplace=True)

    y = df[dependent_variable]
    X = df.drop(dependent_variable, axis=1)

    # invert grades:
    grades = ["gsp_jz", "gmu_jz", "gbi_jz", "gma_jz", "gde_jz", "gla_jz", "sctyp"]
    for grade in grades:
        reversed = max(X[grade]) - X[grade]
        X[grade] = reversed
    if drop_p_degree_expect == True:
        X.drop("parental_expectation_degree", inplace=True, axis=1)

    # find cor:
    df_cor = X.corr()
    cor_dict[df_num] = df_cor

mean_cors = pd.concat([cor_dict[1], cor_dict[2], cor_dict[3], cor_dict[4]]).groupby(level=0).mean()
# reorder vars according to blocks (create a list): df[['col2', 'col3', 'col1']]

block_dict = parse_data_to_dict(block_file)
col_order_list = []
new_block_dict = {}
for block_name, block_vars in block_dict.items():
    new_block_vars = []
    for var in block_vars:
        if var in list(mean_cors.columns):
            new_block_vars.append(var)
            col_order_list.append(var)
    new_block_dict[block_name] = new_block_vars

# save new block dict
final_grouped_vars = pd.DataFrame.from_dict(new_block_dict, orient='index')
final_grouped_vars.to_csv("Data/MetaData/Variable_blocks/final_blocked_vars.csv")

mean_cors = mean_cors[col_order_list]
mean_cors = mean_cors.reindex(col_order_list)
mean_cors_abs = abs(mean_cors)

# plot heatmap
lines = []
line_int = 0
for block, vars_list in new_block_dict.items():
    block_len = len(vars_list)
    line_int = line_int + block_len
    lines.append(line_int)

label_geo_list=[]
line_int = 0
for block, vars_list in new_block_dict.items():
    block_len = len(vars_list)
    if block_len > 1:
        midpoint = block_len/2
    else:
        midpoint = 1
    line_geo = line_int + midpoint
    label_geo_list.append(line_geo)
    line_int = line_int + block_len

group_labels = ["Prior Ach.", "Grades", "School Track", "Demo. & SES",
                "IQ", "Motiv. & Emot.", "Fam. Context (S,P)",
                "Cog. Strat.", "Class Context (S)", "Class Context (T)"]

save_path = "Data/MetaData/Block_Correlations_Heatmaps/"
plot_corr_blocks(cor=mean_cors_abs, save_path=save_path,
                 save_name="ordered_mean_cors_REV.png",
                 title="", lines=lines, label_geo=label_geo_list,
                 group_labels=group_labels)

plot_corr_blocks_2col(cor=mean_cors, save_path=save_path,
                 save_name="ordered_mean_cors_2col_REV.png",
                 title="", lines=lines, label_geo=label_geo_list,
                 group_labels=group_labels)

plot_corr_blocks_HH(cor=mean_cors, save_path=save_path,
                 save_name="ordered_mean_cors_2in1_REV.png",
                 title="", lines=lines, label_geo=label_geo_list,
                 group_labels=group_labels)

# todo: add with pos/neg

#todo: create a plot with the just features >0.7?

print('done')