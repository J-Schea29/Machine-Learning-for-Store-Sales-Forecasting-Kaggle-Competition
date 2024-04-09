import cudf
import cupy as cp
import pandas
import numpy as np
from cuml.metrics import mean_squared_error, mean_squared_log_error
# Cross-Validation
from sklearn.model_selection import TimeSeriesSplit

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class Hybrid_Time_Series_ML:
    '''
    A Class for creating either boosted or stacked Hybrid Machine Learning Models for Time Series. 
    
    Parameters:
        model_1: 1st Machine Learning model to be used (usually linear),
        model_2: 2nd Machine Learning Model to be used (usually non-linear),
        Boosted: A boolean variable which, when True, trains 2nd model on the residuals of the 1st model. If false, 
                 then it adds the predictions of the first model to the 2nd dataset. Default is Boosted.
                 
    Attributes:
        change_model: Change input either model
        fit(X_1, X_2, y): Fits Hybrid model to the data.
        predict(X_1, X_2): Generates Prediction for Hybrid model.
        name(): Gives the names of both models. 
    '''
    
    def __init__(self, model_1, model_2, Boosted=True, to_tensor=False):    
        '''
        Initializes the Hybrid_Time_Series_ML class.
        
        Parameters:
            model_1: 1st Machine Learning model to be used (usually linear),
            model_2: 2nd Machine Learning Model to be used (usually non-linear),
            Boosted: A boolean variable which, when True, trains 2nd model on the residuals of the 1st model. If false 
                     then it adds the predictions of the first model to the 2nd dataset. Default is Boosted.
        '''

        #Defining instance variables
        self.model_1 = model_1
        self.model_2 = model_2
        self.boosted = Boosted
        self.to_tensor = to_tensor

    def change_model(self, model, first=True):
        '''
        Change input either model
        
        Parameters:
            model: new model, 
            first: determines which model to change.
        '''
        
        if first:
            self.model_1 = model
        
        else:
            self.model_2 = model

            
    def fit(self, X_1, X_2, y):
        '''
        Fits Hybrid model to the data.
        
        Parameters:
            X_1: The data to use for training the 1st model,
            X_2: The data to use for training the 2nd model,
            y: The target variable.
        '''

        y_c = y.copy()
        y_c.set_index(["store_nbr", "family"], append=True, inplace=True)
        
        unstack_y = y_c.unstack(["store_nbr", "family"])

        self.unstack_y = unstack_y
        
        self.model_1.fit(X_1, unstack_y)

        y_fit = self.model_1.predict(X_1)

        y_fit.columns = unstack_y.columns
        y_fit.index = X_1.index
        
        if self.boosted:

            y_resid = unstack_y - y_fit
            
            y_resid.columns = y_fit.columns
            y_resid.index = X_1.index
            
            y_resid = y_resid.stack(["store_nbr", "family"]).reset_index().sort_values(["store_nbr", "family", "date"])

            if self.to_tensor:
                
                X_2 = torch.from_dlpack(X_2.astype("float32").values).to(device)
                y_t = torch.from_dlpack(y_resid[["sales"]].astype("float32").values).to(device)
                self.model_2.fit(X_2, y_t)
                
            else:
                
                self.model_2.fit(X_2, y_resid.reset_index()[["sales"]])
                

        else:

            y_fit = y_fit.stack(["store_nbr", "family"]).reset_index().sort_values(["store_nbr", "family", "date"])
            X_2 = X_2.copy()
            X_2["fit"] = y_fit["sales"]

            if self.to_tensor:
            
                X_2 = torch.from_dlpack(X_2.astype("float32").values).to(device)
                y_t = torch.from_dlpack(y_c.reset_index()[["sales"]].astype("float32").values).to(device)

                self.model_2.fit(X_2, y_t)
            
            else:
            
                self.model_2.fit(X_2, y_c.reset_index()["sales"])
            
    def predict(self, X_1, X_2):
        '''
        Generates Prediction for Hybrid model.
        
        Parameters:
            X_1: The data to use for making predictions with the 1st model,
            X_2: The data to use for making predictions with  the 2nd model.
        
        Returns: Hybrids Models Predictions for data X_1 and X_2.
        '''

        y_pred_1 = self.model_1.predict(X_1)
        
        y_pred_1.columns = self.unstack_y.columns
        y_pred_1.index = X_1.index

        y_pred_1 = y_pred_1.stack(["store_nbr", "family"]).reset_index().sort_values(["store_nbr", "family", "date"])
        
        if self.boosted:

            if self.to_tensor:

                X_2 = torch.from_dlpack(X_2.astype("float32").values).to(device)

            y_pred_2 = cudf.DataFrame(self.model_2.predict(X_2))

            y_pred_1 = y_pred_1.set_index(["date", "store_nbr", "family"])
            
            y_pred_2.index = y_pred_1.index
            y_pred_2.columns = y_pred_1.columns
            
            y_pred_boosted = y_pred_2 + y_pred_1
            
            return y_pred_boosted.clip(0)
            
        else:

            X_2 = X_2.copy()
            X_2["fit"] = y_pred_1["sales"]

            if self.to_tensor:
                
                X_2 = torch.from_dlpack(X_2.astype("float32").values).to(device)
                
            y_pred_stacked = cudf.DataFrame(self.model_2.predict(X_2))
            
            y_pred_1 = y_pred_1.set_index(["date", "store_nbr", "family"])
            
            y_pred_stacked.index = y_pred_1.index
            y_pred_stacked.columns = y_pred_1.columns
            
            return y_pred_stacked.clip(0)     
            
    def name(self):
        '''
        Gives the names of both models. 
        '''
        return f"Model 1: {self.model_1.__class__.__name__}, Model 2: {self.model_2.__class__.__name__}"
    

class Hybrid_Pipeline:
    '''
    A class for creating a pipeline for Hybrid Machine Learning Models for Time Series.

    Parameters:
        Data_Preprocess: Takes the Prepare_Data Class.
        model1: The first model to run.
        model2: The second model to run.
        Boosted: Determines whether the second model is fitted on the residual of the first, or if False, whether the predictions of the first are added to the second dataset.
        to_tensor: Determines if X_2 gets converted to tensor.

    Attributes:
        fit(X, y): Transforms data and then fits the model to it.
        predict(X): Transforms data and then makes predictions with it.
    '''

    def __init__(self, Data_Preprocess, model1, model2, Boosted=True, to_tensor=True):
        '''
        Initializes the Hybrid_Pipeline class.

        Parameters:
            Data_Preprocess: A class for transforming data.
            model1: The first model to run.
            model2: The second model to run.
            Boosted: Determines whether the second model is fitted on the residual of the first, or if False, whether the predictions of the first are added to the second dataset.
            to_tensor: Determines if X_2 gets converted to tensor.
        '''
        # Defining instance variables
        self.preprocess = Data_Preprocess

        # Calls Hybrid_Series Class
        self.models = Hybrid_Time_Series_ML(model1, model2, Boosted=Boosted, to_tensor=to_tensor)

    def fit(self, X, y):
        '''
        Transforms data and then fits the model to it.

        Parameters:
            X: Independent variable(s) for training, or a list with validation and training data.
            y: The dependent variable(s) for training, or a list with validation and training data.
        '''

        # Exclude ID column
        X_C = X.copy()
        X_C.drop("id", axis=1, inplace=True)

        X_1, X_2 = self.preprocess.fit_transform(X_C)
        # Fit hybrid models
        self.models.fit(X_1, X_2, y)

    def predict(self, X):
        '''
        Transforms data and then makes predictions with it.

        Parameters:
            X: The independent variable(s).
            
        Returns:
            DataFrame: Predicted values.
        '''
        # Exclude ID column
        X_C = X.copy()
        X_id = X_C["id"]
        X_C.drop("id", axis=1, inplace=True)
        X_1, X_2 = self.preprocess.transform(X_C)

        pred = self.models.predict(X_1, X_2)
        pred = pred.set_index(X_id)
        # return predictions
        return pred

    
def Time_Series_CV(model, X_C, y, splits=4, verbose=False):
    # Use time series split for cross validation. 
    cv_split = TimeSeriesSplit(n_splits = splits)
    
    # Create lists to append MSLE scores.
    valid_msle = []
    train_msle = []
    
    # Dates to index through. 
    dates = X_C.index.drop_duplicates()
    a = 0
    
    # Perform Cross-Validation to determine how model will do on unseen data.
    for train_index, valid_index in cv_split.split(dates):

        # Index dates.
        date_train, date_valid = dates[train_index], dates[valid_index]

        # Selecting data for y_train and y_valid.
        y_train = y.loc[date_train]
        y_valid = y.loc[date_valid]

        # Selecting data for X_train and X_valid.
        X_train = X_C.loc[date_train]
        X_valid = X_C.loc[date_valid]

        X_train = X_train.reset_index().sort_values(["store_nbr", "family", "date"])
        X_valid = X_valid.reset_index().sort_values(["store_nbr", "family", "date"])
        X_train = X_train.set_index(["date"])
        X_valid = X_valid.set_index(["date"])

        y_train = y_train.reset_index().sort_values(["store_nbr", "family", "date"])
        y_valid = y_valid.reset_index().sort_values(["store_nbr", "family", "date"])
        y_train = y_train.set_index(["date"])
        y_valid = y_valid.set_index(["date"])


        # Fitting model.
        model.fit(X_train, y_train)

        # Create predictions for Trainning and Validation.
        pred = model.predict(X_valid)

        # MSE for trainning and validation. 
        valid_msle.append(float(mean_squared_log_error(y_valid["sales"], pred["sales"])))
        
        if verbose:
            # Create predictions for Trainning and Validation.
            fit = model.predict(X_train)
        
            # MSE for trainning and validation. 
            train_msle.append(float(mean_squared_log_error(y_train["sales"], fit["sales"])))
            
            a = a+1
            print(f"Fold {a}:") 
            print(f"Training RMSLE: {cp.sqrt(mean_squared_log_error(y_train.sales, fit.sales)):.3f}, Validation RMSLE: {cp.sqrt(mean_squared_log_error(y_valid.sales, pred.sales)):.3f}")
        
    if verbose:
        # Returns the square root of the average of the MSE.
        print("Average Across Folds")
        print(f"Training RMSLE:{np.sqrt(np.mean(train_msle)):.3f}, Validation RMSLE: {np.sqrt(np.mean(valid_msle)):.3f}")
        
    return float(np.sqrt(np.mean(valid_msle)))
