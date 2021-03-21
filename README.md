# Forecasting-USD-EUR-Exchange-Rate-using-Arima-GARCH-VAR-and-RNN-models


In this project we use ARIMA，GARCH and VAR(Vector Autoregression Model) models to forecast the trend and value of the USD/EUR exchange rate.


### Data Source

Obtain the data of the exchange rate of USD/EUR from the site of The European Central Bank. 

The data of indicators (money supply (in US and Europe), GDP (in US and Europe),CPI ( in US and Europe) ,Philadelphia Gold and Silver Index(XAU) and West Texas Intermediate(WTI)) are from https://fred.stlouisfed.org/. The time frame of our dataset is one month starting from Jan. 2015 to Feb. 2020.

Taking into account that the rate is noise and non-stationary, we assume that the historical data is the result of incorporation of all the behaviors including inflation, economic growth, changes of interest rates and etc, .


### Methods 

- ARIMA
- RNN(Recurrent Neural Network)
- GARCH 
- VAR(Vector Autoregression Model)


### Results

The best model is selected based on the prediction of the future trend and the accuracy of models with Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).



### Documentation description

1. "LSTM.py" is raw code for LSTM model.
   "rate.csv" is the data of LSTM model in "LSTM.py". 
   "LSTM.html" is the result of "LSTM.py".

2. "models.rmd" is raw code for ARIMA, GARCH,VEC,DCC.
   "rate.csv" is the data of ARIMA, GARCH model. 
   "data.csv" is the data of VEC model.
   "US_part.xlsx" and "DAILY_part.xlsx" is the data of DCC.
   "Models.pdf" is the result of "models.rmd".

3. "VAR.rmd" and "VAR.pdf" are raw code and results of VAR model.
   "data.csv" is the data of VAR model.
