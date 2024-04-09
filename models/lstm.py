# Data Preprocessing 
from cuml.dask.preprocessing import OneHotEncoder, LabelEncoder
from cuml.preprocessing import MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder, OneHotEncoder
from cuml.compose import make_column_transformer
from statsmodels.tsa.deterministic import CalendarFourier
from sklearn.pipeline import Pipeline

# Cross-Validation
from sklearn.model_selection import TimeSeriesSplit

# Models
from sklearn.dummy import DummyRegressor
from cuml.linear_model import LinearRegression
from xgboost import XGBRegressor
from cuml.neighbors import KNeighborsRegressor
from cuml.ensemble import RandomForestRegressor
from cuml.metrics import mean_squared_error, mean_squared_log_error

# Hyperparameter Optimization
# from bayes_opt import BayesianOptimization
# from bayes_opt.logger import JSONLogger
# from bayes_opt.event import Events

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# My functions
from data_wrangling import Prepare_data
from hybrid_timeseries import Hybrid_Pipeline, Time_Series_CV

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss().to(device)
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred+1), torch.log(actual + 1))
    
    
class LSTMModel(nn.Module):
    def __init__(self, input_layer, n_hidden_1, n_hidden_2, drop):
        super(LSTMModel, self).__init__()
        
        self.input_layer = input_layer
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        
        # Layers: Linear, LSTM, Linear
        self.linear1 = nn.Linear(input_layer, n_hidden_1)
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(n_hidden_1, n_hidden_2, batch_first=True)
        self.linear2 = nn.Linear(n_hidden_2, 1)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.dropout(x)
        
        output, (h_t, c_t) = self.lstm(x)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.ReLU(output)
        return output


class LSTMRegressor():
    def __init__(self, n_hidden=50, n_hidden_2=20, drop=0.2, epochs=100, early_stop=5, lr=0.01, Boosted=False, verbose=False):
        
        self.n_hidden = n_hidden
        self.n_hidden_2 = n_hidden_2
        self.drop = drop
        if Boosted:
            self.criterion = nn.MSELoss().to(device)
        else: 
            self.criterion = MSLELoss()
            
        self.early_stop = early_stop 
        self.epochs = epochs 
        self.lr = lr
        self.min_val_loss = float('inf')
        self.min_val_loss_2 = float('inf')
        self.verbose = verbose
    def train(self, train_loader):
        
        self.model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch, y_batch

            self.optimizer.zero_grad()
            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()


    def pred(self, test_loader, valid=False, epoch=0):
        
        self.model.eval()
        if valid:
             
            val_losses = 0
            num = 0
            
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch
                    outputs = self.model(x_batch)

                    loss = self.criterion(outputs, y_batch)
                    val_losses=+loss.item()

                    num=+1

            val_loss = val_losses/num

            if val_loss<self.min_val_loss:
            
                self.min_val_loss = val_loss
                self.early_stop_count = 0
            else:
                self.early_stop_count+=1
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Validation score of {np.sqrt(val_loss):.4f}")
            if self.early_stop_count>=self.early_stop:
                if self.verbose:
                    print(f"early stopping at Validation Score of {np.sqrt(self.min_val_loss):.4f}")
                    print()
                self.stop = True
            
        else:
            
            with torch.no_grad():
                predictions = []
                for x_batch in test_loader:
                    x_batch = x_batch.to(device)
                    outputs = self.model(x_batch)
                    predictions.append(outputs.cpu().numpy())

                return np.concatenate(predictions)
                
    def fit(self, X, y):
        
        
        if isinstance(X, list):
            X_train, y_train = X[0], y[0]
            self.model = LSTMModel(X_train.shape[1], n_hidden_1=self.n_hidden, n_hidden_2= self.n_hidden_2, drop=self.drop).to(device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            train_loader = DataLoader(TensorDataset(X_train.to(device), y_train.to(device)), batch_size=31, shuffle=False)
            
            X_valid, y_valid = X[1], y[1]
            test_loader = DataLoader(TensorDataset(X_valid.to(device), y_valid.to(device)), batch_size=31, shuffle=False)
            
            self.stop=False
            self.early_stop_count =0 
            
            for epoch in range(self.epochs):
                self.train(train_loader)
                
                self.pred(test_loader, valid=True, epoch=epoch)
                if self.stop:
                    break

        else:
                X_train, y_train = X, y
                self.model = LSTMModel(X_train.shape[1], n_hidden_1=self.n_hidden, n_hidden_2= self.n_hidden_2, drop=self.drop).to(device)
                
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                train_loader = DataLoader(TensorDataset(X_train.to(device), y_train.to(device)), batch_size=31, shuffle=False)
                
                for epoch in range(self.epochs):
                    self.train(train_loader)
    
    def predict(self, X):
        
        test_loader = DataLoader(X.to(device), batch_size=31, shuffle=False)
        
        outputs = self.pred(test_loader)
        
        return outputs

# Define the preprocessing steps
numeric_transformer = ["float", StandardScaler()]
categorical_transformer = ["category", OneHotEncoder(sparse=False, handle_unknown='ignore')]

column_list = ["time_since_quake", "time_since_quake_sq"]

#data_preprocessor = Prepare_data(column_list, [numeric_transformer, categorical_transformer])
data_preprocessor = Prepare_data(column_list, [numeric_transformer])


print("******** Linear Regression, LSTM Regressor, Boosted ********")

# X_C = X.copy()

# X_C["family"] = X_C["family"].cat.codes
# X_C["store_nbr"] = X_C["store_nbr"].cat.codes
# X_C["holiday"] = X_C["holiday"].cat.codes
# X_C["event"] = X["event"].cat.codes
# X_C["city"] = X_C["city"].cat.codes
# X_C["state"] = X_C["state"].cat.codes
# X_C["type"] = X_C["type"].cat.codes
# X_C["payday"] = X_C["payday"].cat.codes
# X_C["workday"] = X_C["workday"].cat.codes
# X_C["holiday_description"] = X_C["holiday_description"].cat.codes

# X_C = X_C[["id", "store_nbr", "family"] + sorted(set(X_C.columns)-set(["id", "store_nbr", "family"]))]

# X_C = X_C.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])
# y = y.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])

# X_test_C = X_test.copy()
# X_test_C = X_test_C[["id", "store_nbr", "family"] + sorted(set(X_test_C.columns)-set(["id", "store_nbr", "family"]))]
# X_test_C = X_test_C.reset_index().sort_values(["store_nbr", "family", "date"])
# X_test_C = X_test_C.set_index(["date"])

# def objective_function(onpromotion, total_other_promo_store, total_other_city_promo, holiday,
#                        holiday_description, event, time_since_quake, time_since_quake_sq, state, city, 
#                        dcoilwtico, day, month, workday, payday, onpromotion_lag_1, type, dayofyear, year,
#                        onpromotion_lag_2, onpromotion_lag_3, onpromotion_lag_4, onpromotion_lag_5,
#                        onpromotion_lag_6, onpromotion_lag_7, dcoilwtico_lag_1, dcoilwtico_lag_2, 
#                        dcoilwtico_lag_3, dcoilwtico_lag_4, dcoilwtico_lag_5, dcoilwtico_lag_6, 
#                        dcoilwtico_lag_7, sales_one_year_lag, Change_in_oil_prices, promo_last_7_days, X_C=X_C, y=y):
    
#     # Convert non-integer arguments to integers
#     variable_list = [int(round(Change_in_oil_prices)), int(round(city)),int(round(day)), int(round(dayofyear)), int(round(dcoilwtico)),
#                         int(round(dcoilwtico_lag_1)), int(round(dcoilwtico_lag_2)), int(round(dcoilwtico_lag_3)), int(round(dcoilwtico_lag_4)),
#                         int(round(dcoilwtico_lag_5)), int(round(dcoilwtico_lag_6)), int(round(dcoilwtico_lag_7)), int(round(event)),
#                         int(round(holiday)), int(round(holiday_description)), int(round(month)), int(round(onpromotion)),int(round(onpromotion_lag_1)),
#                         int(round(onpromotion_lag_2)), int(round(onpromotion_lag_3)), int(round(onpromotion_lag_4)), int(round(onpromotion_lag_5)),
#                         int(round(onpromotion_lag_6)), int(round(onpromotion_lag_7)), int(round(payday)), int(round(promo_last_7_days)),
#                         int(round(sales_one_year_lag)), int(round(state)), int(round(time_since_quake)), int(round(time_since_quake_sq)),
#                         int(round(total_other_city_promo)), int(round(total_other_promo_store)), int(round(type)), int(round(workday)), int(round(year))]
    
#     X_C = X_C.copy()
#     column_to_remove = []
#     for i in range(3, X_C.shape[1]):
#         if variable_list[i-3]==0:
#             column_to_remove.append(X_C.columns[i])
    
#     X_C.drop(column_to_remove, axis=1, inplace=True)

#     model_2 = LSTMRegressor(Boosted=True)

#     model_1 = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
#     # Use time series split for cross validation. 
#     cv_split = TimeSeriesSplit(n_splits = 4)
    
#     # Create lists to append MSLE scores.
#     valid_msle = []
    
#     # Dates to index through. 
#     dates = X_C.index.drop_duplicates()
    
    
#     list1 = ["time_since_quake", "time_since_quake_sq"]
#     list2 = X_C.columns

#     # Convert lists to sets
#     set1 = set(list1)
#     set2 = set(list2)

#     # Find the values in set1 that are not in set2
#     uncommon_values = set1 - set2

#     # Remove the uncommon values from list1
#     list1 = [value for value in list1 if value not in uncommon_values]
    
#     numeric_transformer = ["float", StandardScaler()]
#     categorical_transformer = ["uint8", OneHotEncoder(sparse=False, handle_unknown='ignore')]
#     data_preprocessor = Prepare_data(list1, [numeric_transformer, categorical_transformer])    
    
#     # Perform Cross-Validation to determine how model will do on unseen data.
#     for train_index, valid_index in cv_split.split(dates):

#         model = Hybrid_Pipeline(data_preprocessor, model_1, model_2, Boosted=True, to_tensor=True)

#         # Index dates.
#         date_train, date_valid = dates[train_index], dates[valid_index]

#         # Selecting data for y_train and y_valid.
#         y_train = y.loc[date_train]
#         y_valid = y.loc[date_valid]

#         # Selecting data for X_train and X_valid.
#         X_train = X_C.loc[date_train]
#         X_valid = X_C.loc[date_valid]

#         X_train = X_train.reset_index().sort_values(["store_nbr", "family", "date"])
#         X_valid = X_valid.reset_index().sort_values(["store_nbr", "family", "date"])
#         X_train = X_train.set_index(["date"])
#         X_valid = X_valid.set_index(["date"])

#         y_train = y_train.reset_index().sort_values(["store_nbr", "family", "date"])
#         y_valid = y_valid.reset_index().sort_values(["store_nbr", "family", "date"])
#         y_train = y_train.set_index(["date"])
#         y_valid = y_valid.set_index(["date"])


#         # Fitting model.
#         model.fit(X_train, y_train)

#         # Create predictions for Trainning and Validation.
#         pred = model.predict(X_valid)

#         # MSE for trainning and validation. 
#         valid_msle.append(float(mean_squared_log_error(y_valid["sales"], pred["sales"])))


#     return -float(np.sqrt(np.mean(valid_msle)))

# # Define the parameter space for Bayesian optimization (each feature is a parameter)
# params = {X_C.columns[i]: (0, 1) for i in range(3, X_C.shape[1])}

# # Initialize the Bayesian optimizer
# optimizer = BayesianOptimization(
#     f=objective_function,
#     pbounds=params,
#     random_state=1,  # For reproducibility
# )

# # Perform the optimization
# optimizer.maximize(init_points=30, n_iter=100)

# variables = list(optimizer.max["params"].values())
# variables = [True, True, True] + [x>0.5 for x in variables]
# X_C = X_C[X_C.columns[variables]]

# list1 = ["time_since_quake", "time_since_quake_sq"]
# list2 = X_C.columns

# # Convert lists to sets
# set1 = set(list1)
# set2 = set(list2)

# # Find the values in set1 that are not in set2
# uncommon_values = set1 - set2

# # Remove the uncommon values from list1
# list1 = [value for value in list1 if value not in uncommon_values]

# numeric_transformer = ["float", StandardScaler()]
# categorical_transformer = ["uint8", OneHotEncoder(sparse=False, handle_unknown='ignore')]
# data_preprocessor = Prepare_data(list1, [numeric_transformer, categorical_transformer])    
# def hyperparameter_optimization(n_hidden, n_hidden_2, drop, epochs, lr):
    
    
#     model_1 = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)
    
    
#     model_2 = LSTMRegressor(n_hidden=n_hidden, n_hidden_2=n_hidden_2, drop=drop, epochs=epochs, lr=lr, Boosted=True)
#     # Use time series split for cross validation. 
#     cv_split = TimeSeriesSplit(n_splits = 4)
    
#     # Create lists to append MSLE scores.
#     valid_msle = []

#     # Dates to index through. 
#     dates = X_C.index.drop_duplicates()

    
#     # Perform Cross-Validation to determine how model will do on unseen data.
#     for train_index, valid_index in cv_split.split(dates):

#         model = Hybrid_Pipeline(data_preprocessor, model_1, model_2, Boosted=True, to_tensor=True)

#         # Index dates.
#         date_train, date_valid = dates[train_index], dates[valid_index]

#         # Selecting data for y_train and y_valid.
#         y_train = y.loc[date_train]
#         y_valid = y.loc[date_valid]

#         # Selecting data for X_train and X_valid.
#         X_train = X_C.loc[date_train]
#         X_valid = X_C.loc[date_valid]

#         X_train = X_train.reset_index().sort_values(["store_nbr", "family", "date"])
#         X_valid = X_valid.reset_index().sort_values(["store_nbr", "family", "date"])
#         X_train = X_train.set_index(["date"])
#         X_valid = X_valid.set_index(["date"])

#         y_train = y_train.reset_index().sort_values(["store_nbr", "family", "date"])
#         y_valid = y_valid.reset_index().sort_values(["store_nbr", "family", "date"])
#         y_train = y_train.set_index(["date"])
#         y_valid = y_valid.set_index(["date"])


#         # Fitting model.
#         model.fit(X_train, y_train)

#         # Create predictions for Trainning and Validation.
#         pred = model.predict(X_valid)

#         # MSE for trainning and validation. 
#         valid_msle.append(float(mean_squared_log_error(y_valid["sales"], pred["sales"])))


#     return -float(np.sqrt(np.mean(valid_msle)))

# param_bounds = {
#     'n_hidden': (1, 20),
#     'n_hidden_2': (0, 1),
#     'drop': (0, 1),
#     'epochs': (10, 500),
#     'lr': (0, 1),
# }
    
# optimizer = BayesianOptimization(
#     f=hyperparameter_optimization,
#     pbounds=parambounds,
#     random_state=1,
# )

# optimizer.maximize(init_points=30, n_iter=100,)
# print(optimizer.max)

# params = optimizer.max["params"]

# if params[4]>1:
#     if params[4]>2:
#         metric = 'euclidean'
#     else:
#         metric = 'manhattan'
# else:
#     metric = 'minkowski'


# if params[2]>1:
#     if params[2]>2:
#         algorithm = 'auto'
#     else:
#         if params[2]>3:
#             algorithm = 'ball_tree'
#         else:
#             algorithm = 'kd_tree'
# else:
#     algorithm = 'brute'


# if params[1]>.5:
#     weights = 'uniform'
# else:
#     weights = 'distance'

# param = {'n_neighbors': int(round(params[0])),
#         'weights': weights,
#         'algorithm': algorithm,
#         'leaf_size': int(round(params[3])),
#         'metric': metric}

# knn = KNeighborsRegressor(**knn_params)


# model_1 = LinearRegression(fit_intercept=False, algorithm="svd", copy_X=True)

# # Use time series split for cross validation. 
# cv_split = TimeSeriesSplit(n_splits = 4)

# # Create lists to append MSE scores. 
# train_msle = []
# valid_msle = []

# # Dates to index through. 
# dates = X_C.index.drop_duplicates()
# a = 0
# # Perform Cross-Validation to determine how model will do on unseen data.
# for train_index, valid_index in cv_split.split(dates):
#     a = a+1
#     print(f"Fold {a}:") 
#     model = Hybrid_Pipeline(data_preprocessor, lr, knn, Boosted=True, to_tensor=True)
    
#     # Index dates.
#     date_train, date_valid = dates[train_index], dates[valid_index]

#     # Selecting data for y_train and y_valid.
#     y_train = y.loc[date_train]
#     y_valid = y.loc[date_valid]
    
#     # Selecting data for X_train and X_valid.
#     X_train = X_C.loc[date_train]
#     X_valid = X_C.loc[date_valid]
    
#     X_train = X_train.reset_index().sort_values(["store_nbr", "family", "date"])
#     X_valid = X_valid.reset_index().sort_values(["store_nbr", "family", "date"])
#     X_train = X_train.set_index(["date"])
#     X_valid = X_valid.set_index(["date"])

#     y_train = y_train.reset_index().sort_values(["store_nbr", "family", "date"])
#     y_valid = y_valid.reset_index().sort_values(["store_nbr", "family", "date"])
#     y_train = y_train.set_index(["date"])
#     y_valid = y_valid.set_index(["date"])


#     # Fitting model.
#     model.fit(X_train, y_train)

#     # Create predictions for Trainning and Validation.
#     fit = model.predict(X_train)
#     pred = model.predict(X_valid)
    
#     # MSE for trainning and validation. 
#     train_msle.append(float(mean_squared_log_error(y_train["sales"], fit["sales"])))
#     valid_msle.append(float(mean_squared_log_error(y_valid["sales"], pred["sales"])))
    
#     print(f"Training RMSLE: {cp.sqrt(mean_squared_log_error(y_train.sales, fit.sales)):.3f}, Validation RMSLE: {cp.sqrt(mean_squared_log_error(y_valid.sales, pred.sales)):.3f}")

# # Returns the square root of the average of the MSE.
# print("Average Across Folds")
# print(f"Training RMSLE:{np.sqrt(np.mean(train_msle)):.3f}, Validation RMSLE: {np.sqrt(np.mean(valid_msle)):.3f}")

# e_6 = 1/np.sqrt(np.mean(valid_msle))
# # Fit Model
# model = Hybrid_Pipeline(data_preprocessor, lr, xfvsgb, Boosted=True, to_tensor=False)
# model.fit(X_C, y)

# X_test_C = X_test_C[X_test_C.columns[variables]]
# pred_7 = model.predict(X_test_C)

def lstm(Boosted):
    
    if Boosted:
        text = "b"
    else:
        text = "s"
        
    model_name = f"lr_xgb_{text}"

    print(f"******** Linear Regression, LSTM Regressor, {text} ********")
    
    # Read the pickle file into a cuDF DataFrame
    X_C = cudf.read_pickle('../input/X.pkl')
    X_test = cudf.read_pickle('../input/X_test.pkl')
    y = cudf.read_pickle('../input/y.pkl')   

    
    model = Hybrid_Pipeline(data_preprocessor, lr, xfvsgb, Boosted=Boosted, to_tensor=True)
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
    r_forest_boosted()
    r_forest_stacked()

if __name__ == "__main__":
    main()
    