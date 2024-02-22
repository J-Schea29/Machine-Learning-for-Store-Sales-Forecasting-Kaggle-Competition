import cudf
import cupy as cp
import pandas
import numpy as np

# Data Preprocessing 
from cuml.preprocessing import MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder, OneHotEncoder
from cuml.compose import make_column_transformer
from statsmodels.tsa.deterministic import CalendarFourier

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class DeterministicProcess_gpu:
    def __init__(self, order, constant=True, fourier=[]):
                 
        self.fourier = fourier
        self.order = order
        self.constant = constant
                 
    def in_sample(self, data):
                 
        data = data.copy()   
                 
        if self.constant:
            data["constant"]=1
        
        data["trend"] = cp.arange(1,len(data.index)+1)
        
        self.last_value =len(data.index)+1         
        
        for i in range(2, self.order+1):
                 data[f"trend_{i}"] = data["trend"]**i
                 
        if self.fourier:    
            cf = cudf.DataFrame.from_pandas(CalendarFourier(freq=self.fourier[0], order=self.fourier[1]).in_sample(data.index.to_pandas()))
                 
            data = data.merge(cf, on="date")         
        return data.sort_index()
    def out_of_sample(self, data):
        
        data = data.copy()   
                 
        if self.constant:
            data["constant"]=1
        
        data["trend"] = cp.arange(self.last_value,len(data.index)+self.last_value)
                 
        for i in range(2, self.order+1):
                 data[f"trend_{i}"] = data["trend"]**i
                 
        if self.fourier:    
            cf = cudf.DataFrame.from_pandas(CalendarFourier(freq=self.fourier[0], order=self.fourier[1]).in_sample(data.index.to_pandas()))
                 
            data = data.merge(cf, on="date")         
        return data.sort_index()



class ColTransformer_gpu:
    def __init__(self, transformers):
        self.transformers = transformers
    
    def fit_transform(self, X):
        # Initialize an empty list to store transformed features
        X = X.copy()
        transformed_features = []
        
        num = 0
        
        # Iterate over each transformer and apply fit_transform
        for dtype, transformer in self.transformers:
            try: 
                X_transformed = cudf.DataFrame(transformer.fit_transform(X[X.columns[X.dtypes==dtype]]))

                prev_num = num
                num += len(X_transformed.columns)

                # Rename columns to ensure uniqueness
                X_transformed.columns = [f"{i}" for i in range(prev_num, num)]

                transformed_features.append(X_transformed)  # Append transformed feature DataFrame

                X  = X.drop(X.columns[X.dtypes==dtype], axis=1)
            except:
                break

        transformed_features.append(X)
        
        # Concatenate transformed features horizontally
        X_transformed_concat = cudf.concat(transformed_features, axis=1)
        X_transformed_concat.columns = [f"{i}" for i in range(len(X_transformed_concat.columns))]
        
        return X_transformed_concat
    
    def transform(self, X):
        X = X.copy()
        # Initialize an empty list to store transformed features
        transformed_features = []
        
        num = 0
        
        # Iterate over each transformer and apply transform
        for dtype, transformer in self.transformers:
            try:
                X_transformed = cudf.DataFrame(transformer.transform(X[X.columns[X.dtypes==dtype]]))

                prev_num = num
                num += len(X_transformed.columns)

                # Rename columns to ensure uniqueness
                X_transformed.columns = [f"{i}" for i in range(prev_num, num)]

                transformed_features.append(X_transformed)  # Append transformed feature DataFrame

                X  = X.drop(X.columns[X.dtypes==dtype], axis=1)
            except:
                break

        transformed_features.append(X)
        
        # Concatenate transformed features horizontally
        X_transformed_concat = cudf.concat(transformed_features, axis=1)
        X_transformed_concat.columns = [f"{i}" for i in range(len(X_transformed_concat.columns))]
        
        return X_transformed_concat



class Prepare_data:
    '''
    A Class for preparing data for Hybrid Models. 
    
    Parameters:
        X_1_column: Columns to be used in the 1st Machine Learning model,
        unwanted_columns: Columns to not be used for the 2nd Machine Learning Model.
                 
    Attributes:
        X_1(X): Creates data for the first model,
        X_2(X): Creates data for the second model,
        preprocessor_2: Preprossesing for data of type oj=bject, catergory, and ,
        transform(X): Transform data performing X_1(X) and X_2(X)  
    '''
    
    def __init__(self, X_1_column, transformer_list, unwanted_columns=[], to_tensor=False):
        '''
        Initializes the Prepare_data class.
        
        Parameters:
            X_1_column: Columns to be used in the 1st Machine Learning model,
            unwanted_columns: Columns to not be used for the 2nd Machine Learning Model.
        '''
        
        #Defining instance variables
        self.column_list = X_1_column
        self.unwanted_columns = unwanted_columns
        self.to_tensor = to_tensor
        self.ct = ColTransformer_gpu(transformer_list)
        self.dpg = DeterministicProcess_gpu(1, fourier = ["W", 2])
        
    def X_1_fit_transform(self, X): 
        '''
        Transform data into X_1. Expects linear 1st model so uses Deterministic Process. 
        
        Parameters:
            X: Data for the model. 
        '''
        
        # return deterministic process
        return self.dpg.in_sample(X[self.column_list][~X.index.duplicated(keep='last')])

    def X_1_transform(self, X): 
        '''
        Transform data into X_1. Expects linear 1st model so uses Deterministic Process. 
        
        Parameters:
            X: Data for the model. 
        '''
        
        return self.dpg.out_of_sample(X[self.column_list][~X.index.duplicated(keep='last')])

    def X_2_fit_transform(self, X):
        '''
        Transform data into X_2.
        
        Parameters:
            X: Data for the model. 
        '''
        X_2 = X.reset_index()
        X_2 = X_2.drop(self.column_list + self.unwanted_columns + ["date"], axis=1)
        
        X_2 = self.ct.fit_transform(X_2)
        return X_2
        
    def X_2_transform(self, X):
        '''
        Transform data into X_2.
        
        Parameters:
            X: Data for the model. 
        '''
        X_2 = X.reset_index()
        X_2 = X_2.drop(self.column_list + self.unwanted_columns + ["date"], axis=1)
        X_2 = self.ct.transform(X_2)
        
        return X_2
    
    def fit_transform(self, X):
        '''
        Transform data into X_1 and X_2.
        
        Parameters:
            X: Data for the model. 
        '''
        X_1, X_2 = self.X_1_fit_transform(X), self.X_2_fit_transform(X)
        if self.to_tensor:
            
            X_1 = torch.tensor(X_1.values).to(device)
            X_2 = torch.from_numpy(X_2.astype("float")).to(device)
        
        return X_1, X_2
        
    def transform(self, X, to_tensor=False):
        '''
        Transform data into X_1 and X_2.
        
        Parameters:
            X: Data for the model.
            to_tensor: If true changes data to tensors on the GPU.
        '''
        X_1, X_2 = self.X_1_transform(X), self.X_2_transform(X)
        
        if self.to_tensor:
            
            X_1 = torch.tensor(X_1.values).to(device)
            X_2 = torch.from_numpy(X_2.astype("float")).to(device)
            
        return X_1, X_2
    
    

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
                
                self.model_2.fit(X_2, y_resid.reset_index()["sales"])
                

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
