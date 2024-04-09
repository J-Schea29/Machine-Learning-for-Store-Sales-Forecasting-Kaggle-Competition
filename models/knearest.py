# Data Preprocessing 
from cuml.dask.preprocessing import OneHotEncoder, LabelEncoder
from cuml.preprocessing import MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder, OneHotEncoder
from cuml.compose import make_column_transformer
from statsmodels.tsa.deterministic import CalendarFourier
from sklearn.pipeline import Pipeline
import pickle

# Cross-Validation
from sklearn.model_selection import TimeSeriesSplit

# Models
from cuml.linear_model import LinearRegression
from xgboost import XGBRegressor
from cuml.metrics import mean_squared_error, mean_squared_log_error

#Hyperparameter Optimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# My functions
from data_wrangling import Prepare_data
from hybrid_timeseries import Hybrid_Pipeline, Time_Series_CV


class obj_funct:
    
    def __init__(self, Boosted, X_C, y):
        self.Boosted = Boosted.copy()
        self.X_C = X_C.copy()
        self.y = y.copy()
        
    def optimization(onpromotion, total_other_promo_store, total_other_city_promo, holiday,
                               holiday_description, event, time_since_quake, time_since_quake_sq, state, city, 
                               dcoilwtico, day, month, workday, payday, onpromotion_lag_1, type, dayofyear, year,
                               onpromotion_lag_2, onpromotion_lag_3, onpromotion_lag_4, onpromotion_lag_5,
                               onpromotion_lag_6, onpromotion_lag_7, dcoilwtico_lag_1, dcoilwtico_lag_2, 
                               dcoilwtico_lag_3, dcoilwtico_lag_4, dcoilwtico_lag_5, dcoilwtico_lag_6, 
                               dcoilwtico_lag_7, sales_one_year_lag, Change_in_oil_prices, promo_last_7_days):

        # Convert non-integer arguments to integers
        variable_list = [int(round(Change_in_oil_prices)), int(round(city)),int(round(day)), int(round(dayofyear)), int(round(dcoilwtico)),
                            int(round(dcoilwtico_lag_1)), int(round(dcoilwtico_lag_2)), int(round(dcoilwtico_lag_3)), int(round(dcoilwtico_lag_4)),
                            int(round(dcoilwtico_lag_5)), int(round(dcoilwtico_lag_6)), int(round(dcoilwtico_lag_7)), int(round(event)),
                            int(round(holiday)), int(round(holiday_description)), int(round(month)), int(round(onpromotion)),int(round(onpromotion_lag_1)),
                            int(round(onpromotion_lag_2)), int(round(onpromotion_lag_3)), int(round(onpromotion_lag_4)), int(round(onpromotion_lag_5)),
                            int(round(onpromotion_lag_6)), int(round(onpromotion_lag_7)), int(round(payday)), int(round(promo_last_7_days)),
                            int(round(sales_one_year_lag)), int(round(state)), int(round(time_since_quake)), int(round(time_since_quake_sq)),
                            int(round(total_other_city_promo)), int(round(total_other_promo_store)), int(round(type)), int(round(workday)), int(round(year))]
        
        X_C = self.X_C
        y = self.y
        column_to_remove = []
        for i in range(3, X_C.shape[1]):
            if variable_list[i-3]==0:
                column_to_remove.append(X_C.columns[i])
        
        X_C.drop(column_to_remove, axis=1, inplace=True)
    
        model_2 = KNeighborsRegressor()
    
        model_1 = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
        
        list1 = ["time_since_quake", "time_since_quake_sq"]
        list2 = X_C.columns
    
        # Convert lists to sets
        set1 = set(list1)
        set2 = set(list2)
    
        # Find the values in set1 that are not in set2
        uncommon_values = set1 - set2
    
        # Remove the uncommon values from list1
        list1 = [value for value in list1 if value not in uncommon_values]
        
        numeric_transformer = ["float", StandardScaler()]
        categorical_transformer = ["uint8", OneHotEncoder(sparse=False, handle_unknown='ignore')]
        data_preprocessor = Prepare_data(list1, [numeric_transformer, categorical_transformer])    
        
        model = Hybrid_Pipeline(data_preprocessor, model_1, model_2, Boosted=self.Boosted, to_tensor=False)
    
        return -Time_Series_CV(model, X_C, self.y)

class hyp_funct:
    
    def __init__(self, Boosted, X_C, y):
        self.Boosted = Boosted
        self.X_C = X_C.copy()
        self.y = y.copy()
        
    def optimization(n_neighbors, weights, algorithm, leaf_size, metric):

        model_1 = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
        
        if metric>1:
            if metric>2:
                metric = 'euclidean'
            else:
                metric = 'manhattan'
        else:
            metric = 'minkowski'
    
    
        if algorithm>1:
            if algorithm>2:
                algorithm = 'auto'
            else:
                if algorithm>3:
                    algorithm = 'ball_tree'
                else:
                    algorithm = 'kd_tree'
        else:
            algorithm = 'brute'
    
    
        if weights>.5:
            weights = 'uniform'
        else:
            weights = 'distance'
        
        param = {
        'n_neighbors': int(round(n_neighbors)),
        'weights': weights,
        'algorithm': algorithm,
        'leaf_size': int(round(leaf_size)),
        'metric': metric}
    
        model_2 = KNeighborsRegressor(**params)

        list1 = ["time_since_quake", "time_since_quake_sq"]
        list2 = X_C.columns
    
        # Convert lists to sets
        set1 = set(list1)
        set2 = set(list2)
    
        # Find the values in set1 that are not in set2
        uncommon_values = set1 - set2
    
        # Remove the uncommon values from list1
        list1 = [value for value in list1 if value not in uncommon_values]
    
        numeric_transformer = ["float", StandardScaler()]
        categorical_transformer = ["uint8", OneHotEncoder(sparse=False, handle_unknown='ignore')]
        data_preprocessor = Prepare_data(list1, [numeric_transformer, categorical_transformer])
    
        model = Hybrid_Pipeline(data_preprocessor, model_1, model_2, Boosted=self.Boosted, to_tensor=False)
    
        return -Time_Series_CV(model, X_C, y)
            
def knearest(Boosted):
    
    if Boosted:
        text = "b"
    else:
        text = "s"
        
    model_name = f"lr_knearest_{text}"
    # Read the pickle file into a cuDF DataFrame
    X_C = cudf.read_pickle('../input/X.pkl')
    X_test = cudf.read_pickle('../input/X_test.pkl')
    y = cudf.read_pickle('../input/y.pkl')
        
    print(f"******** Linear Regression, K-NN Regressor {text} ********")
    
    X_C["family"] = X_C["family"].cat.codes
    X_C["store_nbr"] = X_C["store_nbr"].cat.codes
    X_C["holiday"] = X_C["holiday"].cat.codes
    X_C["event"] = X["event"].cat.codes
    X_C["city"] = X_C["city"].cat.codes
    X_C["state"] = X_C["state"].cat.codes
    X_C["type"] = X_C["type"].cat.codes
    X_C["payday"] = X_C["payday"].cat.codes
    X_C["workday"] = X_C["workday"].cat.codes
    X_C["holiday_description"] = X_C["holiday_description"].cat.codes
    
    X_C = X_C[["id", "store_nbr", "family"] + sorted(set(X_C.columns)-set(["id", "store_nbr", "family"]))]
    
    X_C = X_C.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])
    y = y.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])
    
    X_test_C = X_test.copy()
    X_test_C = X_test_C[["id", "store_nbr", "family"] + sorted(set(X_test_C.columns)-set(["id", "store_nbr", "family"]))]
    X_test_C = X_test_C.reset_index().sort_values(["store_nbr", "family", "date"])
    X_test_C = X_test_C.set_index(["date"])
    
    # Convert float64 columns to float32
    float_cols = X_test_C.select_dtypes(include=['float64']).columns
    X_test_C[float_cols] = X_test_C[float_cols].astype('float32')
    
    # Define the parameter space for Bayesian optimization (each feature is a parameter)
    params = {X_C.columns[i]: (0, 1) for i in range(3, X_C.shape[1])}
    
    obj_funct = obj_funct(Boosted, X_C, y)
    
    # Initialize the Bayesian optimizer
    optimizer = BayesianOptimization(
        f=obj_funct.optimization,
        pbounds=params,
        random_state=1,  # For reproducibility
    )
    
    logger = JSONLogger(path=f"./Logs/logs.{model_name}_v")
    
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    try:
        load_logs(optimizer, logs[f"./Logs/logs.{model_name}_v"])
    except:
        pass
    
    # Perform the optimization
    optimizer.maximize(init_points=30, n_iter=70)
    
    variables = list(optimizer.max["params"].values())
    variables = [True, True, True] + [x>0.5 for x in variables]
    X_C = X_C[X_C.columns[variables]]
    
    list1 = ["time_since_quake", "time_since_quake_sq"]
    list2 = X_C.columns
    
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    
    # Find the values in set1 that are not in set2
    uncommon_values = set1 - set2
    
    # Remove the uncommon values from list1
    list1 = [value for value in list1 if value not in uncommon_values]
    
    numeric_transformer = ["float", StandardScaler()]
    categorical_transformer = ["uint8", OneHotEncoder(sparse=False, handle_unknown='ignore')]
    data_preprocessor = Prepare_data(list1, [numeric_transformer, categorical_transformer])    
    
    param_bounds = {
        'n_neighbors': (1, 20),
        'weights': (0, 1),
        'algorithm': (0, 4),
        'leaf_size': (10, 50),
        'metric': (0, 3),
    }
    
    hyp_funct = hyp_funct(Boosted, X_C, y)
    
    optimizer = BayesianOptimization(
        f=hyp_funct.optimization,
        pbounds=parambounds,
        random_state=1,
    )
    
    logger = JSONLogger(path=f"./Logs/logs.{model_name}_p")
    
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    try:
        load_logs(optimizer, logs[f"./Logs/logs.{model_name}_p"])
    except:
        pass
    
    optimizer.maximize(init_points=30, n_iter=70)
    
    params = optimizer.max["params"]
    
    if params[4]>1:
        if params[4]>2:
            metric = 'euclidean'
        else:
            metric = 'manhattan'
    else:
        metric = 'minkowski'
    
    
    if params[2]>1:
        if params[2]>2:
            algorithm = 'auto'
        else:
            if params[2]>3:
                algorithm = 'ball_tree'
            else:
                algorithm = 'kd_tree'
    else:
        algorithm = 'brute'
    
    
    if params[1]>.5:
        weights = 'uniform'
    else:
        weights = 'distance'
    
    param = {'n_neighbors': int(round(params[0])),
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': int(round(params[3])),
            'metric': metric}
    
    knn = KNeighborsRegressor(**knn_params)
    
    lr = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
    
    model = Hybrid_Pipeline(data_preprocessor, lr, knn, Boosted=Boosted, to_tensor=False)
     
    e = 1/Time_Series_CV(model, X_C, y, verbose=True)
    
    # Fit Model
    model.fit(X_C, y)
    
    X_test_C = X_test_C[X_test_C.columns[variables]]

    pred = model.predict(X_test_C)
    
    try:
        cv_scores = pd.read_csv("../Output/cv_score.csv")
    
    except:
        # Define the columns for the new DataFrame
        columns = ["Model", "cv"]

        # Create an empty DataFrame with the specified columns
        cv_scores = pd.DataFrame(columns=columns)
        
    new_row = {"Model": model_name, "cv": e}
    cv_scores = cv_scores.append(new_row, ignore_index=True)
    cv_scores.to_csv('../Output/cv_score.csv', index=False)
    
    # Save predictions to pickle file
    pred.to_pickle(f'../Output/{model_name}.pkl')

def main():
    
    # Call your functions here
    knearest(True)
    knearest(False)

if __name__ == "__main__":
    main()