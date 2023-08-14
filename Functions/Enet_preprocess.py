import pandas as pd

def dummy_code(df, vars_to_code):
    cols_to_code = []
    for var in vars_to_code:
        if var in df.columns.to_list():
            cols_to_code.append(var)

    df_dum = pd.get_dummies(df,
                            columns=cols_to_code,
                            drop_first=True)
    return df_dum

def remove_special_char_colnames(df, char):
    df.columns = df.columns.str.split(char).str[0]
    return df
