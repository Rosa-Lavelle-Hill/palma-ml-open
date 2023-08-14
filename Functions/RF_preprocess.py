import pandas as pd

def to_categorical(df, cat_vars):
    for col in df.columns:
        if col in cat_vars:
            df[col] = df[col].astype("category")
    return df
