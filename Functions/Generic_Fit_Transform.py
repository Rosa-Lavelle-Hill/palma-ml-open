from Fixed_params import cat_vars
import numpy as np
import pandas as pd

def preprocess_transform(X, X_train, pipe, preprocessor):
    """returns transfrormed dataframes with colum names"""
    preprocessor.fit(X_train)
    X_train_arr = preprocessor.transform(X_train)
    numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
    dum_names = list(pipe.named_steps['preprocessor'].transformers_[1][1].named_steps['oh_encoder']
                     .get_feature_names(cat_vars))
    num_names = list(X[numeric_features].columns.values)
    names = num_names + dum_names
    X_train_tr = pd.DataFrame(X_train_arr, columns=names)
    print('number of features after coding = {}'.format(np.shape(X_train_arr)[1]))
    return X_train_tr



def get_preprocessed_col_names(X, pipe):
    numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
    dum_names = list(pipe.named_steps['preprocessor'].transformers_[1][1].named_steps['oh_encoder']
                     .get_feature_names(cat_vars))
    num_names = list(X[numeric_features].columns.values)
    names = num_names + dum_names
    return names