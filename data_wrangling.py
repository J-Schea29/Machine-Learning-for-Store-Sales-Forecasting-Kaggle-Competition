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
            for f_list in self.fourier:
                
                cf = cudf.DataFrame.from_pandas(CalendarFourier(freq=f_list[0], order=f_list[1]).in_sample(data.index.to_pandas()))
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
            for f_list in self.fourier:
                
                cf = cudf.DataFrame.from_pandas(CalendarFourier(freq=f_list[0], order=f_list[1]).in_sample(data.index.to_pandas()))
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
    
    def __init__(self, X_1_column, transformer_list, fourier = [["W", 2]], unwanted_columns=[], to_tensor=False):
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
        self.dpg = DeterministicProcess_gpu(1, fourier = fourier)
        
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
    
    
