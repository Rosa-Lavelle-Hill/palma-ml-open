enet_param_grid = {"regression__tol": [0.01],
               "regression__max_iter": [1000],
               "regression__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
               "regression__alpha": [0.5, 1, 2, 3]
               }

test_enet_param_grid = {"regression__tol": [0.01],
               "regression__max_iter": [1000],
               "regression__l1_ratio": [0],
               "regression__alpha": [1]
               }

rf_param_grid = {"regression__min_samples_split": [2, 4, 6],
             "regression__max_depth": [10, 20, 30],
             "regression__n_estimators": [200, 500],
             "regression__random_state": [93],
             "regression__max_features": [0.2, 0.3, 0.4]
             }

test_rf_param_grid = {"regression__min_samples_split": [4],
             "regression__max_depth": [5],
             "regression__n_estimators": [50],
             "regression__random_state": [93],
             "regression__max_features": [0.3]
             }