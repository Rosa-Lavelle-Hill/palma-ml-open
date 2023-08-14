import random

import pandas as pd
import numpy as np

from Functions.Preprocessing import remove_col

np.set_printoptions(suppress=True)

def block_importance(pipe, X, y, random_state, df_num, save_path, model, block_file):
    np.random.seed(random_state)
    # parse block data
    block_dict = parse_data_to_dict(block_file)
    total_vars = 0

    # fit pipeline to test data (note: will be over-optimistic eval compared to OOS)
    pipe.fit(X, y)
    # evaluate score on test data (without perms)
    full_score = round(float(pipe.score(X, y)), 5)

    perm_dict = {}
    for block_name, block_vars in block_dict.items():
        X_perm = X.copy()
        actual_block_vars = []
        # find block vars in df:
        for var in block_vars:
            var = var.split('.')[0]
            if var in X_perm.columns:
                actual_block_vars.append(var)
            else:
                continue

        # permute all columns in block
        num_vars = len(actual_block_vars)
        total_vars = total_vars + num_vars

        X_perm[actual_block_vars] = X_perm[actual_block_vars].sample(frac=1, axis=0, random_state=93).values

        # evaluate fit (with perm)
        perm_score = round(float(pipe.score(X_perm, y)), 5)
        if perm_score <0:
            score_reduction = round(abs(perm_score - full_score), 5)
        else:
            score_reduction = round(full_score - perm_score, 5)
        perm_dict[block_name] = score_reduction
    df_imp = pd.DataFrame.from_dict(perm_dict, orient='index')
    df_imp.columns = ["Importance"]
    df_imp.sort_values(by="Importance", inplace=True, axis=0, ascending=False)
    df_imp.reset_index(inplace=True)
    df_imp.columns = ["Feature_Block", "Importance"]
    df_imp.to_csv(save_path + "block_importance_{}_df{}.csv".format(model, df_num))
    print("total vars: " + str(total_vars))
    return df_imp


def parse_data_to_dict(block_file):
    block_df = pd.read_csv(block_file)
    block_df = remove_col(block_df, "Unnamed: 0")
    block_dict = {}
    for col in block_df.columns:
        block = block_df[col]
        block.dropna(inplace=True)
        if block.shape[0] > 0:
            block = block.str.split('(', n=1, expand=True)[0]
            block = block.str.strip()
            block_dict[col] = list(block)
    return block_dict


def add_aggregates_to_blockfile(block_file, save_path, save_name):
    block_df = pd.read_csv(block_file)
    block_dict = {}
    for col in block_df.columns:
        block = block_df[col]
        block.dropna(inplace=True)
        if block.shape[0] > 0:
            block = block.str.split('(', n=1, expand=True)[0]
            block = block.str.strip()
            agg_block = []
            for row in block:
                agg_row = str(row) + '_C'
                agg_block.append(agg_row)
            block_expanded = pd.DataFrame(block.append(pd.Series(agg_block), ignore_index=True))
            block_dict[col] = list(block_expanded[0])
    new_block_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in block_dict.items()]))
    new_block_df.to_csv(save_path+save_name)
    return






