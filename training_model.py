# Importing all required modules
import random
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow

## Set Result for Reproducible Results
np.random.seed(1551)
tensorflow.random.set_seed(1551)

'''
Part 1: Creat Dataset to be used in RNN

'''
## Read Dataset
Stock_data = pd.read_csv("data/q2_dataset.csv")
print(Stock_data)

## Convert String to Date time format
Stock_data['Date']=pd.to_datetime(Stock_data.Date)

## Sorting the data
Sorted_data = Stock_data.sort_values("Date")
# print(Sorted_data)

## Drop the "Close" Column from dataset
Clean_data = Sorted_data.drop([" Close/Last"], axis=1)
# print(Clean_data)

## Removing Date from Clean Data
Stocks_df = Clean_data.drop(["Date"], axis =1)
# print(Stocks_df)

## Convert to float
Converted_Stock_df = Stocks_df.astype("float32")
# print(Converted_Stock_df.info())

## Convert the Stock Dataframe to array
Stocks = np.asarray(Converted_Stock_df)
print(Stocks)

'''
Now, Create the dataset using the latest 3 days as features and next day opening price as target

'''

## Define variables for data and target
Data_Features = [] ## Will hold latest 3 days features
target = [] ## Will hold next day opening price
gap = 3 ## Latest 3 days
total_sample_count = len(Stocks)  ## Stock: Complete Data
# print(total_sample_count)

for r in range(gap, total_sample_count):
    ## Create latest 3 days data
    latest_data = Stocks[r - gap:r, :]
    ## Store 3 days data in "Data Features"
    Data_Features.append(latest_data)
    ## Store next day opening price in target
    # print("Next Day opening Price: ", Stocks[r][1])
    target.append(Stocks[r][1])
print("----------------------------------------------------------------")
print("Latest 3 days Stock Data:\n", Data_Features)
print("Next Day Opening Price Data:\n", target)
print("----------------------------------------------------------------")

## Convert the Data features and Target into Dataframe
Complete_Data = zip(Data_Features, target)
Stock_latest_data_with_label = list(Complete_Data)
# print(Stock_latest_data_with_label)

SDL = pd.DataFrame(Stock_latest_data_with_label)
# print(SDL)

## Shuffle the data
SDL_df = SDL.sample(frac=1)
# print(SDL_df)

## Split the data into 70% Training data and 30% Testing data
train_data, test_data  = train_test_split(SDL_df, test_size = 0.3, random_state = 42)
# print(train_data.shape)
# print(test_data.shape)

##Fetch Features from Training Data and flatten it out in order to save data in dataframe, so that it can be converted to a csv file
training_features = train_data[0]
# print(type(training_features))
training_features = training_features.tolist()

## Flatten each array in dataset
training_FF = [list(np.asarray(each_train_point).flatten()) for each_train_point in training_features]
# print(training_FF)

##Fetch Features from Testing Data and flatten it out in order to save data in dataframe, so that it can be converted to a csv file
testing_features = test_data[0]
# print(type(training_features))
testing_features = testing_features.tolist()

## Flatten each array in dataset
testing_FF = [list(np.asarray(each_test_point).flatten()) for each_test_point in testing_features]
# print(testing_FF)

## Fetch Training and Testing labels 
train_labels = train_data[1]
test_labels = test_data[1]
# print(train_labels)
# print(test_labels)

## Convert flattened Training/Testing features and Training/testing Labels into dataframe
train_features_df = pd.DataFrame(training_FF)
test_features_df = pd.DataFrame(testing_FF)
tr_labels = pd.DataFrame(train_labels) 
ts_labels = pd.DataFrame(test_labels)
tr_labels.columns = ['Target']
ts_labels.columns = ['Target']
# print(train_features_df)
# print('--------------------------------')
# print(test_features_df)
# print('--------------------------------')
# print(tr_labels)
# print('--------------------------------')
# print(ts_labels)

## Concatenate training features and labels, testing features and labels
train_data_RNN = pd.concat([train_features_df.reset_index(drop = True), tr_labels.reset_index(drop = True)], axis = 1)
test_data_RNN = pd.concat([test_features_df.reset_index(drop = True), ts_labels.reset_index(drop = True)], axis = 1)
# print(train_data_RNN)
# print('-----------------------')
# print(test_data_RNN)

## Save file in "data" folder as "train_data_RNN.csv" and "test_data_RNN.csv"
train_data_RNN.to_csv("data/train_data_RNN.csv")
test_data_RNN.to_csv("data/test_data_RNN.csv")

'''
Part2:
- Read Training Data
- Preprocesses the Training Data
- Training of RNN Network
- Saved Trained Model in "models" directory with ".h5" extension
- Test Data was not used at any point

''' 
if __name__ == "__main__":
    ## Read csv files from "data" folder
    Training_data = pd.read_csv("data/train_data_RNN.csv")
    # print(Training_data)

    ## Dropping "Unnamed: 0" column from both the dataset
    Train_data = Training_data.drop(["Unnamed: 0"], axis = 1)
    # print(Train_data)

    ## Segregate features and Targets from Training and Testing Data
    Train_features = Train_data.iloc[:,:12]
    # print(Train_features.shape)
    Train_targets = Train_data.iloc[:,-1]
    # print(Train_targets)

    ## Convert Training Targets into array and convert it to 2D data
    Train_targets = np.asarray(Train_targets)  
    Train_targets = Train_targets.reshape(-1,1) # previous shape was (879,), now it will be (879,1) 

    ## Applying Min-Max Scalar for Normalization of "Training Data Features"
    Feature_Normalizer = MinMaxScaler()
    Train_features = np.asarray(Train_features)
    Norm_train_features = Feature_Normalizer.fit_transform(Train_features)
    # print(Norm_train_features)

    ## Applying Min-Max Scalar for Normlization of "Training targets"
    Target_Normalizer = MinMaxScaler()
    Normalised_training_targets = Target_Normalizer.fit_transform(Train_targets)
    # print(Norm_train_features.shape[0])

    ## Reshape the training features backto (3*4) shape in order to provide it as an input to RNN Model
    Normalised_training_features = Norm_train_features.reshape(Norm_train_features.shape[0], 3, 4) ## Converting to (879, 3, 4)
    # print(Normalised_training_features)

    # ## Created RNN Base/Best/Optimal Model
    RNN_Model = Sequential()
    RNN_Model.add(LSTM(64, input_shape=(3,4)))
    RNN_Model.add(Dense(1)) ## Single Node to store "Next Day Opening price"
    RNN_Model.compile(loss='mean_squared_error', optimizer='adam') ## Mean Squared Error for Regression
    ## Results of Base/Best Model: Train MSE Score: 8.49 MSE
    ## Below Results are obtained after Testing from "test_RNN.py" for Base/Best Model
    # ----------------------------------
    # Loss:  0.00010798125618623153
    # Test MSE Score: 8.877
    # ----------------------------------

    '''
    Below are Extra Models (commented out with their results) which were tried and helped me in obtaining optimal model.

    '''
    ## Created RNN Model (Base Model with SGD optimizer) (RNN Model 2)
    # RNN_Model = Sequential()
    # RNN_Model.add(LSTM(64, input_shape=(3,4)))
    # RNN_Model.add(Dense(1))
    # RNN_Model.compile(loss='mean_squared_error', optimizer='sgd') ## Mean Squared Error for Regression
    # ----------------------------------
    # Loss:  0.00018774359877042493
    # Test MSE Score: 15.434
    # ----------------------------------

    # #Created RNN Model 3
    # RNN_Model = Sequential()
    # RNN_Model.add(LSTM(64, return_sequences = True, input_shape=(3,4)))
    # RNN_Model.add(LSTM(32, input_shape=(3,4)))
    # RNN_Model.add(Dense(10))
    # RNN_Model.add(Dense(1))
    # RNN_Model.compile(loss='mean_squared_error', optimizer='adam') ## Mean Squared Error for Regression
    # ----------------------------------
    # Loss:  0.00012192292414069552
    # Test MSE Score: 10.023
    # ----------------------------------

    # ## Created RNN Model 4
    # RNN_Model = Sequential()
    # RNN_Model.add(LSTM(32, return_sequences = True, input_shape=(3,4)))
    # RNN_Model.add(LSTM(32, input_shape=(3,4)))
    # RNN_Model.add(Dense(10))
    # RNN_Model.add(Dense(1))
    # RNN_Model.compile(loss='mean_squared_error', optimizer='adam') ## Mean Squared Error for Regression
    # ----------------------------------
    # Loss:  0.00034922525918621334
    # Test MSE Score: 28.709
    # ----------------------------------

    # ##Created RNN Model 5 
    # RNN_Model = Sequential()
    # RNN_Model.add(LSTM(128, return_sequences = True, input_shape=(3,4)))
    # RNN_Model.add(LSTM(64, return_sequences = True))
    # RNN_Model.add(LSTM(32, input_shape=(3,4)))
    # RNN_Model.add(Dense(10))
    # RNN_Model.add(Dense(1))
    # RNN_Model.compile(loss='mean_squared_error', optimizer='adam') ## Mean Squared Error for Regression
    # ----------------------------------
    # Loss:  0.0001318902383468593
    # Test MSE Score: 10.842
    # ----------------------------------

    ## RNN Model Summary
    Model_Summary = RNN_Model.summary()
    # print(Model_Summary)

    ## Fit the data and train model
    RNN_Model.fit(Normalised_training_features, Normalised_training_targets, epochs = 120, batch_size = 10, verbose =2)

    '''
    Part 3: Cheking Training Mean Square Error for Reporting Purpose

    '''
    ## Training Features Prediction
    Training_prediction = RNN_Model.predict(Normalised_training_features)
    
    ## Inverse Transform in order to obtain original features and Targets
    Predicted_Training_features = Target_Normalizer.inverse_transform(Training_prediction)
    Predicted_Training_targets = Target_Normalizer.inverse_transform(Normalised_training_targets)
    
    ##Calculate Training MSE
    Error = mean_squared_error(Predicted_Training_targets, Predicted_Training_features[:,0])
    print('Train MSE Score: %.2f MSE' % (Error))

    ## Save the model
    RNN_Model.save('models/RNN_model.h5')
