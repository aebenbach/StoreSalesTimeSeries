#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:04:23 2022

@author: Andrew Ebenbach

This file contains all the code used to build the final model. It is broken
into sections for importing data, feature engineering, training the model,
and testing the model.
  
    
"""
#%%importing the data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#train and test data
path_to_file = "/Users/Andrew1/Desktop/Kaggle/Store Sales/Data/train.csv"
df_train = pd.read_csv(path_to_file, index_col="id", parse_dates=(['date']))

path_to_file = "/Users/Andrew1/Desktop/Kaggle/Store Sales/Data/test.csv"
df_test = pd.read_csv(path_to_file, index_col="id", parse_dates=(['date']))

df = pd.concat([df_train, df_test])

#other datasets
#oil
path_to_file = "/Users/Andrew1/Desktop/Kaggle/Store Sales/Data/oil.csv"
df_oil = pd.read_csv(path_to_file,index_col=("date"),  parse_dates=(['date']))
df_oil = df_oil.reindex(index=df.date.unique())
df_oil["date"] = df_oil.index
df_oil.index = range(len(df_oil.index))
df_oil.rename(columns = {"dcoilwtico":"oil_price"},inplace= True)

#replace missing values in oil data set with the most recent value
while(df_oil.isnull().values.any()):
    if(np.isnan(df_oil.iloc[0,0])):
        df_oil.iloc[0,0] = df_oil.iloc[1,0]
    
    nan_index = df_oil.loc[pd.isna(df_oil["oil_price"]), :].index
    
    df_oil.iloc[nan_index,0] = df_oil.iloc[(nan_index-1),0]
    
#store
path_to_file = "/Users/Andrew1/Desktop/Kaggle/Store Sales/Data/stores.csv"
df_stores = pd.read_csv(path_to_file)

#holiday events
path_to_file = "/Users/Andrew1/Desktop/Kaggle/Store Sales/Data/holidays_events.csv"
df_holiday = pd.read_csv(path_to_file, parse_dates=(['date']))
#%%Functions to be used for feature engineering

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

#create time series feature
def timeSeries(df):
    new_df = df.copy()
    
    date_list = new_df.date.unique()
    dates = pd.DataFrame(index = date_list)
    dates["date"] = date_list
    
    fourier = CalendarFourier(freq = 'A', order=8)
    
    
    
    dp = DeterministicProcess(index=dates.index,
                              constant=False,
                              order=1,
                              seasonal=True,
                              period=7,
                              additional_terms=[fourier],
                              drop=True)
    
    time_series = dp.in_sample()
    
    time_series = time_series.join(dates)
    
    new_df = new_df.merge(time_series, 
                          on= "date",
                          how = "left")
    
    return new_df

#merge dfs on date

def mergeDate(df, add_df):
    new_df = df.copy()
    new_df = new_df.merge(add_df, 
                          on= "date",
                          how = "left")
    
    return new_df

#merge on store number
def mergeStore(df, add_df):
    new_df = df.copy()
    new_df = new_df.merge(add_df, 
                          on= "store_nbr",
                          how = "left")
    return new_df

#add holiday indicators
def addHoliday(df, df_holiday):
    new_df = df.copy()
    holidays = pd.get_dummies(df_holiday.type,prefix="day")
    df_dummy_holiday = df_holiday.join(holidays)[0:328]
    
    
    national = df_dummy_holiday.loc[df_dummy_holiday.locale=="National"]
    regional = df_dummy_holiday.loc[df_dummy_holiday.locale=="Regional"]
    regional.rename(columns={"locale_name":"state"},inplace=True)
    
    local = df_dummy_holiday.loc[df_dummy_holiday.locale=="Local"]
    local.rename(columns={"locale_name":"city"},inplace=True)

    prev_date = pd.to_datetime(1999)
    for index, row in national.iterrows():
        if row["date"] == prev_date:
            national.drop(index,inplace=True)
        prev_date = row["date"]
        
        
    new_df = new_df.merge(national[["date"]+list(holidays.columns)], 
                          on= "date",
                          how = "left").fillna(0)
    
    
    
    
    for index, row in regional.iterrows():
        
        for col in holidays.columns:
            if row[col] == 1:
                new_df.loc[((new_df.state == row["state"]) & (new_df.state == row["state"])),col] = 1
                 
    for index, row in local.iterrows():
        
        for col in holidays.columns:
            if row[col] == 1:
                new_df.loc[((new_df.city == row["city"]) & (new_df.city == row["city"])),col] = 1
    
    
    return new_df

#add indicators for the days when public wages are paid
def addPayday(df):
    new_df = df.copy()
    
    dates = df["date"].unique()
    new_df["PayDay"] = 0
    for date in dates: 
        date = pd.to_datetime(date)
        if (date.day == 15) or date.is_month_end :
            new_df.loc[new_df.date == date, "PayDay"] = 1
            
    return new_df


#add lags for any given feature
           
def makeLags(df, first_lag, last_lag, col):
    new_df = df.copy()
    
    fill_val = new_df[col][0]
    
    new_columns = []
    
    lag_list = range(first_lag, last_lag+1, 1)
    
    for lag in lag_list:
        new_col = "lag" + str(lag) + "_" + col
        new_df[new_col] = 0
        new_columns += [new_col]
    
    #breaks data set down into one store's sales in one product family
    #allowing the shift() function to apply lags properly
    for fam in new_df.family.unique():
        for store in new_df.store_nbr.unique():
            print(fam + str(store))
            sub_index = new_df.loc[((new_df.family == fam)&(new_df.store_nbr == store))].index
            
            for i in range(len(new_columns)):
                lag_col = new_columns[i]
                shift = lag_list[i]
                
                new_df.loc[sub_index,lag_col] = new_df.loc[sub_index,col].shift(shift)
            
        
    return new_df


#add a unique variable for each day of year (this is only an approximation because of leap years)
#not used in model
def dayOfYear(df):
    new_df = df.copy()
    
    new_df["day_of_year"] = new_df.trend % 365
    
    return new_df

#ratio of onpromotion products to oil price inspired by PCA
#not used in model
def oilPromotionRatio(df):
    new_df = df.copy()
    
    new_df["OilPromotionRatio"] = new_df.onpromotion / new_df.oil_price 
    
    return new_df

#add numeric codes for categorical features
features_nom = ["store_nbr","family"]       
def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")



#%% add features

X = df.copy()

X = timeSeries(X)

X = mergeDate(X, df_oil)

X = mergeStore(X, df_stores)

X = addHoliday(X, df_holiday)

X = addPayday(X)


print("making lags")
X = makeLags(X, 1, 16, "sales")

X = X.drop(["date","city","state","type"],1)

encode(X)


X["store_nbr"] = X["store_nbr"].cat.codes
X["family"] = X["family"].cat.codes


#%%copy X because can take a while to load a new X

X_copy = X.copy()

#%%split training from test set

X_train = X.iloc[df_train.index,:]
y_train = X_train.pop("sales")

X_test = X.iloc[df_test.index,:]
X_test = X_test.drop("sales",1)

#%%class for the final model object

import time

class ForecastModel():
    def __init__(self, model_type):
        self.model_type = model_type
    
    #fits a new model for each family of products
    def fit_family(self, X, y, model_param):
        
        self.families = X.family.unique()
        self.stores = X.store_nbr.unique()
        
        self.model_series = pd.Series(index= self.families)
        
        for fam in self.families:
            #print("training on fam "+ str(fam))
            X_fam = X.loc[X.family == fam]
            y_fam = y[X_fam.index]
            
            model_fam = XGBRegressor(**model_param)
            
            self.model_series[fam] = model_fam.fit(X_fam, y_fam)
            
        return self.model_series
    
    #makes predicions for the string of models in each family      
    def predict_family(self, X):
        pred = pd.Series(index = X.index)
        for fam in self.families:
            X_fam = X.loc[X.family == fam]
            
            model_fam = self.model_series[fam]
            
            pred[X_fam.index] = model_fam.predict(X_fam)
        
        return pred
    
    #creates a string of family models for each day on the prediction horizon
    def fit_lag(self, X , y,lags ,model_param):
        
        new_X = X.copy()
        
        self.families = X.family.unique()
        self.stores = X.store_nbr.unique()
        
        self.model_lags = pd.DataFrame(index = self.families)
        
        for lag_no in lags:
            print("sleeping...")
            print(time.gmtime().tm_sec)
            time.sleep(180)
            time.gmtime().tm_sec
            print("training on lag " +str(lag_no))
            new_column = "lag"+str(lag_no)+"_sales"
            self.model_lags[new_column] = self.fit_family(new_X, y, model_param)
            
            new_X = new_X.drop(new_column,1)
            
    #creates the prections for the above fitted models       
    def pred_lag(self, X):
        days = X.trend.unique()
        days_amnt = len(days)
        
        if (len(self.model_lags.columns) != days_amnt):
            print("Lags and sample size out of sync")
            return
        
        pred = pd.Series(index = X.index)
        
        new_X = X.copy()
        
        for i in range(days_amnt):
            day = days[i]
            X_day = new_X.loc[new_X.trend == day]
            
            model_series = self.model_lags.iloc[:,i]
            
            col_name = self.model_lags.columns[i]
            
            for fam in self.families:
                X_fam = X_day.loc[X_day.family == fam]
                
                model_fam = model_series[fam]
                
                pred[X_fam.index] = model_fam.predict(X_fam)
                
            new_X = new_X.drop(col_name,1)
                
        return pred
            
    #creates a model for each combination of family and store   
    #not used in final predictions 
    def fit(self, X, y, model_param):
        self.families = X.family.unique()
        self.stores = X.store_nbr.unique()
        
        self.model_df = pd.DataFrame(index= self.families, columns= self.stores)
        
        for fam in self.families:
            for store in self.stores:
                X_fam = X.loc[((X.family == fam) & (X.store_nbr == store))]
                y_fam = y[X_fam.index]
            
                model_fam = XGBRegressor(**model_param)
            
                self.model_df.loc[fam,store] = model_fam.fit(X_fam, y_fam)
    
    #predictions for the above model
    def predict(self, X):
        pred = pd.Series(index = X.index)
        for fam in self.families:
            for store in self.stores:
                X_fam = X.loc[((X.family == fam) & (X.store_nbr == store))]
                model_fam =self.model_df.loc[fam,store]
                
                pred[X_fam.index] = model_fam.predict(X_fam)
                
        return pred
                


    

#%%train and test the model on a subsample of the training data
from sklearn.metrics import mean_squared_log_error
import math
from xgboost import XGBRegressor

sample = 26740

X_train_valsample = X_train[-sample:]
y_train_valsample = y_train[X_train_valsample.index]

X_train_sample = X_train.drop(X_train_valsample.index)
y_train_sample = y_train[X_train_sample.index]

xgb_params = dict(
    max_depth=6,           # maximum depth of each tree - try 2 to 10
    learning_rate=0.1,    # effect of each tree - try 0.0001 to 0.1
    n_estimators= 100,     # number of trees (that is, boosting rounds) - try 1000 to 8000
    min_child_weight=1,    # minimum number of entries in a leaf - try 1 to 10
    colsample_bytree=1,  # fraction of features (columns) per tree - try 0.2 to 1.0
    subsample=.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
    reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
    num_parallel_tree=1,   # set > 1 for boosted random forests
)




model = ForecastModel(XGBRegressor)


print("training model")
#model.fit(X_train_sample,y_train_sample, xgb_params)
#model.fit_family(X_train_sample,y_train_sample, xgb_params)
model.fit_lag(X_train_sample,y_train_sample,range(1,17,1),xgb_params)

print("making predictions")
#pred = model.predict(X_train_valsample)
#pred = model.predict_family(X_train_valsample)
pred = model.pred_lag(X_train_valsample)

pred.loc[pred<0] = 0

print("scoring")
msle = mean_squared_log_error(y_train_valsample,pred)
rmsle = math.sqrt(msle)
print("rmsle: " + str(rmsle))

#%%make competition predictions

print("training model")
model.fit_lag(X_train,y_train, range(1,17,1),xgb_params)

print("making predictions")
results = model.pred_lag(X_test)
results.loc[results<0] = 0

results = pd.DataFrame(results)
results.rename(columns={0:'sales'}, inplace=True)
results.index.names = ['id']

results.to_csv("/Users/Andrew1/Desktop/Kaggle/Store Sales/model_2_results.csv", index=True) 
