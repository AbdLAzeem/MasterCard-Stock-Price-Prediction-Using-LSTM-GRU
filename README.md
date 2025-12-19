# MasterCard-Stock-Price-Prediction-Using-LSTM-GRU
Comparative Study: Time Series based on Kaggle’s MasterCard stock dataset from May 25, 2006, to Oct 11, 2021, and train the LSTM and GRU models to forecast the stock price.
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c8700e4c-aa64-4bb0-b43e-386b031afbed" />
# Long Short-Term Memory (LSTM):
An advanced type of RNN, which was designed to prevent both decaying and exploding gradient problems. Just like RNN, LSTM has repeating modules, but the structure is different. Instead of having a single layer of tanh, LSTM has four interacting layers that communicate with each other.
# Gated Recurrent Unit (GRU):
A variation of LSTM, as both have design similarities, and in some cases, they produce similar results. GRU uses an update gate and a reset gate to solve the vanishing gradient problem. These gates decide what information is important and pass it to the output. The gates can be trained to store information from long ago, without vanishing over time or removing irrelevant information

Use Kaggle’s MasterCard stock dataset from May 25, 2006, to Oct 11, 2021, and train the LSTM and GRU models to forecast the stock price.

Key Features:
# Date Range:
The dataset includes stock data over an extensive period, allowing for in-depth historical analysis. 
# Variables:
# Open:
The price at which the stock opened on a given day. High: The highest price reached during the trading day. 
# Low:
The lowest price reached during the trading day. 
# Close:
The final price at which the stock traded at the end of the day. 
# Adj Close:
The adjusted closing price accounts for dividends and stock splits.
# Volume:
The number of shares traded during the day.
# Applications:
This dataset is ideal for financial analysis, including time series forecasting, trend analysis, and stock price prediction.

# Work Flow:
# Connect Kaggle to Colab:
Import the selected dataset online
# Data Analysis:
Import the MasterCard dataset by adding the Date column to the index and converting it to DateTime format. We will also drop irrelevant columns from the dataset as we are only interested in stock prices, volume, and date.
# Data Insights:
The minimum stock price is  4.10, and the highest is 400.5. The mean is at  105.9, and the standard deviation is 107.3, which means that stocks have high variance.
# overview graph for dataset:
The train_test_plot function takes three arguments: dataset, tstart, and tend and plots a simple line graph. The tstart and tend are time limits in years. We can change these arguments to analyse specific periods. The **line plot **is divided into two parts: train and test. This will allow us to decide the distribution of the test dataset.
<img width="1296" height="373" alt="image" src="https://github.com/user-attachments/assets/f2c658ec-f19d-44cc-b0a9-5fd95994e328" />
# Discussion:
MasterCard stock prices have been on the rise since 2016. It had a dip in the first quarter of 2020, but it gained a stable position in the latter half of the year. Our test dataset consists of one year, from 2021 to 2022, and the rest of the dataset is used for training
# Data Preprocessing:
The train_test_split function divides the dataset into two subsets: training_set and test_set
# Apply MinMaxScaler function:
standardise our training set, which will help us avoid the outliers or anomalies
working with univariate أحادي المتغير series, so the number of features is one, and we need to reshape the X_train to fit on the LSTM model. The X_train has [samples, timesteps], and we will reshape it to [samples, timesteps, features].
# LSTM Model:
The model consists of a single hidden layer of LSTM and an output layer. You can experiment with the number of units, as more units will give you better results. For this experiment, we will set LSTM units to 125, tanh as activation, and set the input size
# compile the model with an RMSprop optimizer and mean square error as a loss function

# Model Training: 
The model will train on 50 epochs with 32 batch sizes. we can change the hyperparameters to reduce training time or improve the results. The model training was completed with the best possible loss.

# Results:
The model got 7.28 RMSE on the test dataset.
According to the line plot below, the single-layered LSTM model has performed well.
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/9e62b72b-93f2-4478-a60a-e5bfa8960b57" />

# GRU Model:
We are going to keep everything the same and just replace the LSTM layer with the GRU layer to properly compare the results. The model structure contains a single GRU layer with 125 units and an output layer
The model has successfully trained with 50 epochs and a batch size of 32.
# Results:
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/2e70bc1e-5df4-4fef-9103-58c4270f8e62" />
As we can see, the real and predicted values are relatively close. The predicted line chart almost fits the actual values.
# GRU model got 5.85 rmse on the test dataset, which is an improvement from the LSTM model.

# Conclusion:
The results clearly show that the GRU model performed better than LSTM, with a similar structure and hyperparameters
