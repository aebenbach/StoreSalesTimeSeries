# StoreSalesTimeSeries
Code used for Kaggle's Store Sales - Time Series Forecasting.

The aim of the project is to predict a day's sales in each product category and store of the ecuadorian grocery chain Corporación Favorita. The training data is over 3 million rows long, and the test data is for the 16 days following the end of the training data.

The overall model is a combination of smaller XGBoost models for each family of products and for each day on the prediction horizon. 

Final predictions placed in top 30% of submissions based on RMSLE: 0.49391 of the test set. 
