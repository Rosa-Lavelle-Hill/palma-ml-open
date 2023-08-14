from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from Fixed_params import cat_vars
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def build_pipeline_enet(X,
                   imputer_max_iter,
                   oh_encoder_drop,
                   imputer_model="BR",
                   random_state=93
                   ):
    elastic_net_regression = ElasticNet()
    simple_imputer = SimpleImputer(strategy="most_frequent", fill_value='missing')

    if imputer_model == "RF":

        imp_iter_num = IterativeImputer(estimator=RandomForestRegressor(),
                                        initial_strategy='mean',
                                        missing_values=np.nan,
                                        max_iter=1,
                                        random_state=random_state)
        imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                        initial_strategy='most_frequent',
                                        missing_values=np.nan,
                                        max_iter=1,
                                        random_state=random_state,
                                        )

        oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
        categorical_transformer = Pipeline(
            steps=[("imputing", imp_iter_cat),
                   ("oh_encoder", oh_encoder),
                   ]
        )
        numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
        numeric_transformer = Pipeline(
            steps=[("imputing", imp_iter_num)]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_vars),
            ]
        )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("regression", elastic_net_regression)]
        )

    if imputer_model == "HH":
        # this is the original method with cat vars as "most common", could also use RF classifier for these
        imputer_model = BayesianRidge()
        transformer = StandardScaler()

        imp_iter_num = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                   random_state=random_state, estimator=imputer_model)

        imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                        initial_strategy='most_frequent',
                                        missing_values=np.nan,
                                        max_iter=1,
                                        random_state=random_state,
                                        )
        oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
        categorical_transformer = Pipeline(
            steps=[("imputing", imp_iter_cat),
                   ("oh_encoder", oh_encoder),
                   ]
        )
        numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
        numeric_transformer = Pipeline(
            steps=[("imputing", imp_iter_num),
                   ('scaling', transformer)]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_vars),
            ]
        )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("regression", elastic_net_regression)]
        )

    if imputer_model == "BR":
        # this is the original method with cat vars as "most common", could also use RF classifier for these
        imputer_model = BayesianRidge()
        transformer = StandardScaler()

        imputer = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                   random_state=random_state, estimator=imputer_model)
        oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
        categorical_transformer = Pipeline(
            steps=[("imputing", simple_imputer),
                   ("oh_encoder", oh_encoder),
                   ]
        )
        numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
        numeric_transformer = Pipeline(
            steps=[("imputing", imputer),
                   ('scaling', transformer)]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_vars),
            ]
        )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("regression", elastic_net_regression)]
        )

    return pipe, preprocessor



def build_pipeline_rf(X,
                   imputer_max_iter,
                   oh_encoder_drop,
                   imputer_model="BR",
                   random_state=93
                   ):
    simple_imputer = SimpleImputer(strategy="most_frequent", fill_value='missing')
    random_forest = RandomForestRegressor()
    if imputer_model == "RF":

        imp_iter_num = IterativeImputer(estimator=RandomForestRegressor(),
                                        initial_strategy='mean',
                                        missing_values=np.nan,
                                        max_iter=1,
                                        random_state=random_state)
        imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                        initial_strategy='most_frequent',
                                        missing_values=np.nan,
                                        max_iter=1,
                                        random_state=random_state,
                                        )

        oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
        categorical_transformer = Pipeline(
            steps=[("imputing", imp_iter_cat),
                   ("oh_encoder", oh_encoder),
                   ]
        )
        numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
        numeric_transformer = Pipeline(
            steps=[("imputing", imp_iter_num)]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_vars),
            ]
        )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("regression", random_forest)]
        )

    if imputer_model == "HH":
        imputer_model = BayesianRidge()
        transformer = StandardScaler()

        imp_iter_num = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                   random_state=random_state, estimator=imputer_model)

        imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                        initial_strategy='most_frequent',
                                        missing_values=np.nan,
                                        max_iter=1,
                                        random_state=random_state,
                                        )
        oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
        categorical_transformer = Pipeline(
            steps=[("imputing", imp_iter_cat),
                   ("oh_encoder", oh_encoder),
                   ]
        )
        numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
        numeric_transformer = Pipeline(
            steps=[("imputing", imp_iter_num),
                   ('scaling', transformer)]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_vars),
            ]
        )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("regression", random_forest)]
        )

    if imputer_model == "BR":
        # this is the original method with cat vars as "most common", could also use RF classifier for these
        imputer_model = BayesianRidge()
        transformer = StandardScaler()

        imputer = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                   random_state=random_state, estimator=imputer_model)
        oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
        categorical_transformer = Pipeline(
            steps=[("imputing", simple_imputer),
                   ("oh_encoder", oh_encoder),
                   ]
        )
        numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
        numeric_transformer = Pipeline(
            steps=[("imputing", imputer),
                   ('scaling', transformer)]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_vars),
            ]
        )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("regression", random_forest)]
        )
    return pipe, preprocessor

def build_preprocessor_ns_base(X,
                   imputer_max_iter,
                   oh_encoder_drop,
                   random_state=93):

    imputer_model = BayesianRidge()
    transformer = StandardScaler()

    imp_iter_num = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                    random_state=random_state, estimator=imputer_model)

    imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                    initial_strategy='most_frequent',
                                    missing_values=np.nan,
                                    max_iter=1,
                                    random_state=random_state,
                                    )
    oh_encoder = OneHotEncoder(handle_unknown='error', drop=oh_encoder_drop)
    categorical_transformer = Pipeline(
        steps=[("imputing", imp_iter_cat),
               ("oh_encoder", oh_encoder),
               ]
    )
    cat_vars_selected = []
    for var in cat_vars:
        if var in X.columns:
            cat_vars_selected.append(var)
    numeric_features = X.drop(cat_vars_selected, inplace=False, axis=1).columns
    numeric_transformer = Pipeline(
        steps=[("imputing", imp_iter_num),
               ('scaling', transformer)]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, cat_vars_selected),
        ]
    )
    return preprocessor


