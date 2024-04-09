#Imports for data wrangling 
import cudf
import cupy as cp
import pandas as pd
import numpy as np
import gc
import pickle

def make_lags(data, column, lags):
    '''Takes Data and creates lagged features for every catergory'''
    for k in range(1, lags+1):
        new_cloumn = data.reset_index().groupby(["store_nbr", "family"])[column].shift(k)
        new_cloumn.index = data.index
        data[f"{column}_lag_{k}"] = new_cloumn

def make_one_year_lag(data, column):
    '''Takes Data and retrieves the values from the previous year'''
    new_cloumn = data.reset_index().groupby(["store_nbr", "family", "dayofyear"])[column].shift(1)
    new_cloumn.index = data.index
    data[f"{column}_one_year_lag"] = new_cloumn
    
    # Any after a year is just the result of the store being closed
    data[f"{column}_one_year_lag"] = data[f"{column}_one_year_lag"].fillna(0)
        
def wrangling():
    
    # import warnings
    # warnings.filterwarnings('ignore')
    print("******** BEGINNING DATA WRANGLING ********")
    
    #Importing given data
    train = cudf.read_csv("./input/train.csv", parse_dates=['date'])
    
    test = cudf.read_csv("./input/test.csv", parse_dates=['date'])
    
    oil = cudf.read_csv("./input/oil.csv", parse_dates=['date'])
    
    holiday = cudf.read_csv("./input/holidays_events.csv")
    
    store = cudf.read_csv("./input/stores.csv")
    
    # Converting dates to datetime
    holiday["date"] = cudf.to_datetime(holiday["date"], format='%Y-%m-%d')
    holiday = holiday.set_index("date")
    
    # Keeping only celbrated holidays
    holiday = holiday.loc[(holiday["transferred"]!=True)].drop("transferred", axis=1)
    holiday.loc[holiday["type"]=="Transfer", "type"] = "Holiday"
    
    # Bridged days are day where there is no work
    bridge = holiday.loc[holiday["type"]=="Bridge"]
    bridge["bridge"] = True
    bridge = bridge[["bridge"]]
    
    # Special events
    event = holiday.loc[holiday["type"]=="Event"][["description"]]
    
    # Keeping only holidays
    holiday = holiday.loc[holiday["type"]=="Holiday"]
    
    # Holidays celerbated localy 
    loc_hol = holiday.loc[holiday["locale"]=="Local"][["locale_name", "description"]]
    
    # Holidays celerbrated regionally
    reg_hol = holiday.loc[holiday["locale"]=="Regional"][["locale_name", "description"]]
    
    #Holidays celberbrated nationally
    nat_hol = holiday.loc[holiday["locale"]=="National"][["description"]]
    
    del holiday
    gc.collect()
    
    # Recording days Earthquake
    quake = event.loc[event["description"].str.find("Terremoto Manabi")!=-1]
    quake["time_since_quake"] = cp.arange(1,len(quake.index)+1)
    quake.drop("description", axis=1, inplace=True)
    
    # Removing Earthquake and adding Sporting Events
    event = event.loc[event["description"].str.find("Terremoto Manabi")==-1]
    event.loc[event["description"].str.find("futbol")!=-1, "description"]= "Sports"
    
    # Ensure proper format
    train["store_nbr"] = train["store_nbr"].astype(int)
    
    # Merging
    X = train.merge(store, on="store_nbr", how="left")
    X.drop("cluster", axis=1, inplace=True)
    
    
    # Converting dates to datetime
    X["date"] = cudf.to_datetime(X["date"], format='%Y-%m-%d')
    
    # Creating feature measuring the total in store promotions.
    total_other_promo_store = X[["date", "store_nbr", "onpromotion"]].groupby(['date', 'store_nbr']).sum()["onpromotion"].reset_index()
    total_other_promo_store = total_other_promo_store.rename(columns={'onpromotion': 'total_other_promo_store',})
    
    # Creating feature measuring the total promotions in each town for similar products.
    total_other_city_promo = X[["date", "onpromotion", "family", "city"]].groupby(['date', 'city', 'family']).sum()["onpromotion"].reset_index()
    total_other_city_promo = total_other_city_promo.rename(columns={'onpromotion': 'total_other_city_promo',})
    
    # Adding new features
    X = X.merge(total_other_promo_store, on=['date', 'store_nbr'], how="left")
    X = X.merge(total_other_city_promo, on=['date', 'city', 'family'], how="left")
    
    # Ensure proper format
    store["store_nbr"] = store["store_nbr"].astype(int)
    test["store_nbr"] = test["store_nbr"].astype(int)
    
    # Merging
    X_test = test.merge(store, on="store_nbr", how="left")
    X_test.drop("cluster", axis=1, inplace=True)
    del total_other_promo_store, total_other_city_promo
    gc.collect()
    
    # Converting dates to datetime
    X_test["date"] = cudf.to_datetime(X_test["date"], format='%Y-%m-%d')
    
    # Creating feature measuring the total in store promotions.
    total_other_promo_store = X_test[["date", "store_nbr", "onpromotion"]].groupby(['date', 'store_nbr']).sum()["onpromotion"].reset_index()
    total_other_promo_store = total_other_promo_store.rename(columns={'onpromotion': 'total_other_promo_store',})
    
    # Creating feature measuring the total promotions in each town for similar products.
    total_other_city_promo = X_test[["date", "onpromotion", "family", "city"]].groupby(['date', 'city', 'family']).sum()["onpromotion"].reset_index()
    total_other_city_promo = total_other_city_promo.rename(columns={'onpromotion': 'total_other_city_promo',})
    
    # Adding new features
    X_test = X_test.merge(total_other_promo_store, on=['date', 'store_nbr'], how="left")
    X_test = X_test.merge(total_other_city_promo, on=['date', 'city', 'family'], how="left")
    
    del store, total_other_promo_store, total_other_city_promo
    gc.collect()
    
    X = X.set_index("date")
    X_test = X_test.set_index("date")
    
    # Adding national holidays
    X = X.merge(nat_hol, on="date", how="left")
    
    # Bridge days
    X = X.merge(bridge, on="date", how="left")
    
    # Adding local holdays
    X = X.merge(loc_hol, left_on=["date", "city"],
                right_on=["date", "locale_name"],
                suffixes=(None, '_l'), how="left"
               )
    X.drop("locale_name", axis=1, inplace=True)
    
    # Adding regional holidays
    X = X.merge(reg_hol, left_on=["date", "state"],
                right_on=["date", "locale_name"], 
                suffixes=(None, '_r'),how="left"
               )
    X.drop("locale_name", axis=1, inplace=True)
    
    # True if holiday that Day
    X["holiday"] = (((X["descriptionNone"].isnull()==False) | (X["description_l"].isnull()==False)) | (X["description"].isnull()==False))
    
    X["holiday_description"] = X['descriptionNone'].fillna('') + X['description_l'].fillna('') + X['description'].fillna('')
    
    # Combine Holiday descriptions
    X.drop("descriptionNone", axis=1, inplace=True)
    X.drop("description_l", axis=1, inplace=True)
    X.drop("description", axis=1, inplace=True)
    
    #Events
    X = X.merge(event, on="date", how="left")
    X = X.rename(columns={'description': 'event',})
    X["event"] = X["event"].fillna("none")
    
    # Adding Quake data
    X = X.merge(quake, on="date", how="left")
    X["time_since_quake"] = X["time_since_quake"].fillna(0)
    
    #To model a diminishing marginal effect on the economy by the earthquake
    X["time_since_quake_sq"] = X["time_since_quake"]**2
    
    # Adding national holidays
    X_test = X_test.merge(nat_hol, on="date", how="left")
    del nat_hol
    gc.collect()
    
    # Bridge days
    X_test = X_test.merge(bridge, on="date", how="left")
    del bridge
    gc.collect()
    
    # Adding local holdays
    X_test = X_test.merge(loc_hol, left_on=["date", "city"],
                right_on=["date", "locale_name"],
                suffixes=(None, '_l'), how="left"
               )
    X_test.drop("locale_name", axis=1, inplace=True)
    del loc_hol
    gc.collect()
    
    # Adding regional holidays
    X_test = X_test.merge(reg_hol, left_on=["date", "state"],
                right_on=["date", "locale_name"], 
                suffixes=(None, '_r'),how="left"
               )
    X_test.drop("locale_name", axis=1, inplace=True)
    del reg_hol
    gc.collect()
    
    # True if holiday that Day
    X_test["holiday"] = (((X_test["descriptionNone"].isnull()==False) | (X_test["description_l"].isnull()==False)) | (X_test["description"].isnull()==False))
    
    X_test["holiday_description"] = X_test['descriptionNone'].fillna('') + X_test['description_l'].fillna('') + X_test['description'].fillna('')
    
    # Combine Holiday descriptions
    X_test.drop("descriptionNone", axis=1, inplace=True)
    X_test.drop("description_l", axis=1, inplace=True)
    X_test.drop("description", axis=1, inplace=True)
    
    #Events
    X_test = X_test.merge(event, on="date", how="left")
    X_test = X_test.rename(columns={'description': 'event',})
    X_test["event"] = X_test["event"].fillna("none")
    del event
    gc.collect()
    
    # Adding Quake data
    X_test = X_test.merge(quake, on="date", how="left")
    X_test["time_since_quake"] = X_test["time_since_quake"].fillna(0)
    del quake
    gc.collect()
    
    #To model a diminishing marginal effect on the economy by the earthquake
    X_test["time_since_quake_sq"] = X_test["time_since_quake"]**2
    
    oil["date"] = cudf.to_datetime(oil["date"], format='%Y-%m-%d')
    oil = oil.set_index("date")
    X = X.merge(oil, on="date", how="left")
    X_test = X_test.merge(oil, on="date", how="left")
    
    del oil
    gc.collect()
    
    # There is no price of oil on days that the market is closed so we interpolate to get next value.
    X["dcoilwtico"]= X["dcoilwtico"].interpolate(method="linear", limit_direction="both")
    X_test["dcoilwtico"]= X_test["dcoilwtico"].interpolate(method="linear", limit_direction="both")
    
    # I just to do a rolling average to smooth out any problems with the empty values,
    # and to capture any effect of changes. 
    X["dcoilwtico"] = X["dcoilwtico"].rolling(
        window=30,       
        min_periods=1,  
    ).mean()
    
    X_test["dcoilwtico"] = X_test["dcoilwtico"].rolling(
        window=30,       
        min_periods=1,  
    ).mean()
    
    # Time variables
    X["day"] = X.index.dayofweek
    X["dayofyear"] = X.index.dayofyear
    X["month"] = X.index.month
    X["year"] = X.index.year
    
    # This varible says whether it is a workday.
    X["workday"] = (((X.bridge.isnull()) & (X.holiday==False)) & ((X["day"]!=5) & (X["day"]!=6)))
    X.drop("bridge", axis=1, inplace=True)
    
    # In Ecudor, people get paid on the 15 and the last day of the month
    X["payday"] = ((X.index.day==15) | (X.index.day==X.index.to_series().dt.days_in_month)) 
    
    # Time variables
    X_test["day"] = X_test.index.dayofweek
    X_test["dayofyear"] =X_test.index.dayofyear
    X_test["month"] = X_test.index.month
    X_test["year"] = X_test.index.year
    
    # This varible says whether it is a workday.
    X_test["workday"] = (((X_test.bridge.isnull()) & (X_test.holiday==False)) & ((X_test["day"]!=5) & (X_test["day"]!=6)))
    X_test.drop("bridge", axis=1, inplace=True)
    
    # In Ecudor, people get paid on the 15 and the last day of the month
    X_test["payday"] = ((X_test.index.day==15) | (X_test.index.day==X_test.index.to_series().dt.days_in_month)) 
    
    # Fixing data type
    X_test = X_test.reset_index()
    X_test = X_test.set_index("date")
    
    X_test["onpromotion"] = X_test["onpromotion"].astype('float')
    X_test["total_other_promo_store"] = X_test["total_other_promo_store"].astype('float')
    X_test["total_other_city_promo"] = X_test["total_other_city_promo"].astype('float')
    X_test["holiday"] = X_test["holiday"].astype('float')
    
    X_test["family"] = X_test["family"].astype('category')
    X_test["store_nbr"] = X_test["store_nbr"].astype('category')
    X_test["holiday"] = X_test["holiday"].astype('category')
    X_test["event"] = X_test["event"].astype('category')
    X_test["city"] = X_test["city"].astype('category')
    X_test["state"] = X_test["state"].astype('category')
    X_test["type"] = X_test["type"].astype('category')
    X_test["workday"] = X_test["workday"].astype('category')
    X_test["payday"] = X_test["payday"].astype('category')
    X_test["holiday_description"] = X_test["holiday_description"].astype('category')
    
    X = X.reset_index()
    X = X.set_index("date")
    
    X["onpromotion"] = X["onpromotion"].astype('float')
    X["total_other_promo_store"] = X["total_other_promo_store"].astype('float')
    X["total_other_city_promo"] = X["total_other_city_promo"].astype('float')
    X["holiday"] = X["holiday"].astype('float')
    
    X["family"] = X["family"].astype('category')
    X["store_nbr"] = X["store_nbr"].astype('category')
    X["holiday"] = X["holiday"].astype('category')
    X["event"] = X["event"].astype('category')
    X["city"] = X["city"].astype('category')
    X["state"] = X["state"].astype('category')
    X["type"] = X["type"].astype('category')
    X["workday"] = X["workday"].astype('category')
    X["payday"] = X["payday"].astype('category')
    X["holiday_description"] = X["holiday_description"].astype('category')
    
    X_lag = cudf.concat([X[["store_nbr", "family", "dayofyear", "onpromotion", "dcoilwtico", "sales"]], X_test[["store_nbr", "family", "onpromotion", "dcoilwtico"]]], axis=0)
    X_lag = X_lag.reset_index().sort_values(["store_nbr", "family", "date"]).set_index(["date"])
    
    X_lag["dayofyear"] = X_lag.index.dayofyear
            
    make_lags(X_lag, "onpromotion", 7)
    make_lags(X_lag, "dcoilwtico", 7)
    
    make_one_year_lag(X_lag, "sales")
    
    X_lag = X_lag.drop(["dayofyear", "onpromotion", "dcoilwtico", "sales"], axis=1)
    
    X = X.merge(X_lag, on=["date", "store_nbr", "family"], how="left")
    X_test = X_test.merge(X_lag, on=["date", "store_nbr", "family"], how="left")
    
    del X_lag
    
    X["Change_in_oil_prices"] = X["dcoilwtico"]-X["dcoilwtico_lag_1"]
    X_test["Change_in_oil_prices"] = X_test["dcoilwtico"]-X_test["dcoilwtico_lag_1"]
    X["Change_in_oil_prices"] = X["Change_in_oil_prices"].astype('float')
    X_test["Change_in_oil_prices"] = X_test["Change_in_oil_prices"].astype('float')
    
    X["promo_last_7_days"] = X[X.columns[X.columns.str.find("onpromotion_lag")==0]].sum(axis=1)
    X_test["promo_last_7_days"] = X_test[X_test.columns[X_test.columns.str.find("onpromotion_lag")==0]].sum(axis=1)
    X["promo_last_7_days"] = X["promo_last_7_days"].astype('float')
    X_test["promo_last_7_days"] = X_test["promo_last_7_days"].astype('float')
    
    y = X[["store_nbr", "family", "sales"]]
    X.drop("sales", axis=1, inplace=True)
    
    # # Removing early time with NaNs
    X = X.loc[X.index >= "2015-07-01"]
    y = y.loc[y.index >= "2015-07-01"]
    
    # X = X.loc[X.index >= "2016-01-01"]
    # y = y.loc[y.index >= "2016-01-01"]

    X.to_pickle('input/X.pkl')
    X_test.to_pickle('input/X_test.pkl')
    y.to_pickle('input/y.pkl')

def main():
    
    wrangling()

if __name__ == "__main__":
    main()
