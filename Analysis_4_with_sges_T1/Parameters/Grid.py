enet_param_grid = {"regression__tol": [0.001],
               "regression__max_iter": [5000],
               "regression__l1_ratio": [0.2, 0.4, 0.6, 0.8, 0.9, 1],
               "regression__alpha": [0.2, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2]
               }

rf_param_grid = {"regression__min_samples_split": [2, 3, 4, 5, 6],
                 "regression__max_depth": [7, 10, 12, 15, 20, 25],
                 "regression__n_estimators": [250, 500, 750],
                 "regression__random_state": [93],
                 "regression__max_features": [0.2, 0.3, 0.4, 0.5]}

# best grid for 5->6 (l1 ratio 1)
test_enet_param_grid = {"regression__tol": [0.001],
               "regression__max_iter": [1000],
               "regression__l1_ratio": [1, 2],
               "regression__alpha": [0.5]
               }

# best grid for 5->6
test_rf_param_grid = {'regression__max_depth': 25,
                      'regression__max_features': 0.4,
                      'regression__min_samples_split': 3,
                      'regression__n_estimators': 250,
                      'regression__random_state': 93}