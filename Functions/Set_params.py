def set_params_do_initial_preprocess(Do_initial_preprocess, Choose_drop_DV_T1,
                                     preprocess_drop_DV_T1, dependent_variable):
    version = None
    if Do_initial_preprocess == True:
        if preprocess_drop_DV_T1 == True:
            print("Running initial preprocess without Sges_T1")
            version = "_without_{}_T1".format(dependent_variable)
        if preprocess_drop_DV_T1 == False:
            print("Running initial preprocess with Sges_T1")
            version = "_with_{}_T1".format(dependent_variable)
    elif Do_initial_preprocess == False:
        if Choose_drop_DV_T1 == True:
            version = "_without_{}_T1".format(dependent_variable)
        if Choose_drop_DV_T1 == False:
            version = "_with_{}_T1".format(dependent_variable)
    else:
        print("add True or False as param")
        breakpoint()
    return version


def set_params_testrun(test, test_rf_param_grid, test_enet_param_grid,
                       rf_param_grid, enet_param_grid):
    if test == True:
        rf_params = test_rf_param_grid
        enet_params = test_enet_param_grid
        t = "_test"
    elif test == False:
        rf_params = rf_param_grid
        enet_params = enet_param_grid
        t = ""
    else:
        print("add True or False as param")
        t=None
        enet_params = None
        rf_params = None
        breakpoint()
    params = {'Enet': enet_params, "RF": rf_params}
    return t, params