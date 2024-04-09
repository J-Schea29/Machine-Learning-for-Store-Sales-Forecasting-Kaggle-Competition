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
        self.Boosted = Boosted
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
                
        xgb_params = {
            'tree_method': 'gpu_hist',  # Specify GPU usage
            'predictor': 'gpu_predictor',
            'enable_categorical': True,}
    
        model_2 = XGBRegressor(**xgb_params)
    
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
        data_preprocessor = Prepare_data(list1, [numeric_transformer])
        
        model = Hybrid_Pipeline(data_preprocessor, model_1, model_2, Boosted=Boosted, to_tensor=False)
        
        return -Time_Series_CV(model, X_C, self.y)

class hyp_funct:
    
    def __init__(self, Boosted, X_C, y):
        self.Boosted = Boosted
        self.X_C = X_C.copy()
        self.y = y.copy()
        
    def optimization(n_estimators, gamma, subsample, max_depth, learning_rate):
        X_C = self.X_C
        y = self.y
        model_1 = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
        
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)
        
        params = {
        'tree_method': 'gpu_hist',  # Specify GPU usage
        'predictor': 'gpu_predictor',
        'enable_categorical': True,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'gamma': gamma,
        'subsample': subsample}
    
        model_2 = XGBRegressor(**params)
    
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
        data_preprocessor = Prepare_data(list1, [numeric_transformer])
      
        model = Hybrid_Pipeline(data_preprocessor, model_1, model_2, Boosted=self.Boosted, to_tensor=False)
    
        return -Time_Series_CV(model, X_C, y)
        
def xgboost(Boosted):
    
    if Boosted:
        text = "b"
    else:
        text = "s"
        
    model_name = f"lr_xgb_{text}"
    
    # Read the pickle file into a cuDF DataFrame
    X_C = cudf.read_pickle('../input/X.pkl')
    X_test = cudf.read_pickle('../input/X_test.pkl')
    y = cudf.read_pickle('../input/y.pkl')
        
    column_list = ["time_since_quake", "time_since_quake_sq"]
    
    print(f"******** Linear Regression, XGBoost {text} ********")
    
    X_C = X_C[["id", "store_nbr", "family"] + sorted(set(X_C.columns)-set(["id", "store_nbr", "family"]))]
    
    X_C = X_C.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])
    y = y.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])
        
    X_test_C = X_test.copy()
    X_test_C = X_test_C[["id", "store_nbr", "family"] + sorted(set(X_test_C.columns)-set(["id", "store_nbr", "family"]))]
    X_test_C = X_test_C.reset_index().sort_values(["store_nbr", "family", "date"])
    X_test_C = X_test_C.set_index(["date"])
        
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
    data_preprocessor = Prepare_data(list1, [numeric_transformer])
    
    parambounds = {
        'learning_rate': (0.00001, 1),
        'n_estimators': (0, 1000),
        'max_depth': (3,12),
        'subsample': (0, 1.0),  
        'gamma': (1, 10),
        
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
    
    xgb_params = {
        'tree_method': 'hist',  # Specify GPU usage
        'device': "cuda",
        'enable_categorical': True,
        'max_depth': int(params["max_depth"]),
        'learning_rate': params["learning_rate"],
        'n_estimators': int(params["n_estimators"]),
        'gamma': params["gamma"],
        'subsample': params["subsample"]
    }
    
    xgb = XGBRegressor(**xgb_params)
    
    lr = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
    
    model = Hybrid_Pipeline(data_preprocessor, lr, xgb, Boosted=Boosted, to_tensor=False)
    
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
    xgboost(True)
    xgboost(False)

if __name__ == "__main__":
    main()