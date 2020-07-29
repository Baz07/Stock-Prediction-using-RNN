# Importing all the rquired modules
import random
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout

'''
PART 3:
- Load RNN Trained Model from "model" directory
- Read RNN Test Data from "data" directory
- Run Predcition Model
- Prints Loss on Test Data at every epoch
- Plot of True vs Predicted Values
- Tested (both train_RNN.py and test_RNN.py are running from Command Line)

'''

if __name__ == "__main__":
    ## Load Model
    Model = load_model("models/20842555_RNN_model.h5")

    ## Read Data File
    Testing_data = pd.read_csv("data/test_data_RNN.csv")
    # print(Testing_data)

    ## Drop 'Unnamed: 0' Column
    Clean_testing_data = Testing_data.drop(["Unnamed: 0"], axis = 1)
    # print(Clean_testing_data)

    ## Segregate data into features and targets
    Test_features = Clean_testing_data.iloc[:, :12]
    Test_targets = Clean_testing_data.iloc[:,-1]
    # print(Test_features)
    # print(Test_targets)

    ## Reading Training Data, removing Unnamed Column, segregating Features and Targets 
    Training_data = pd.read_csv("data/train_data_RNN.csv")
    Train_data = Training_data.drop(["Unnamed: 0"], axis = 1)
    Train_features = Train_data.iloc[:,:12]
    Train_targets = Train_data.iloc[:,-1]

    #Convert Training Targets into array and convert it to 2D data
    Train_targets = np.asarray(Train_targets)
    Train_targets = Train_targets.reshape(-1,1) # previous shape was (879,), now it will be (879,1) 

    ## Applying Min-Max Scalar for Normalization of Training Data Features
    Test_Feature_Normalizer = MinMaxScaler()
    Train_features = np.asarray(Train_features)
    Test_Feature_Normalizer.fit_transform(Train_features)

    ## Applying Min-Max Scalar for Normlization of Training Targets
    Test_Target_Normalizer = MinMaxScaler()
    Test_Target_Normalizer.fit_transform(Train_targets)

    ## Applying Min Max Scalar for Normalization of Testing Data Features and Targets
    Norm_Testing_features = Test_Feature_Normalizer.transform(np.asarray(Test_features))
    Norm_Testing_targets = Test_Target_Normalizer.transform(np.asarray(Test_targets).reshape(-1, 1))
    # print(Norm_Testing_features.shape)
    # print(Norm_Testing_targets.shape)

    ## Reshape Testing Features for prediction
    Reshaped_Norm_Test_features = Norm_Testing_features.reshape(Norm_Testing_features.shape[0], 3, 4)
    # print(Reshaped_Norm_Test_features.shape)

    ## Prediction of Testing Features with the help of loaded model (PREDICTION MODEL)
    Predicted_test_features = Model.predict(Reshaped_Norm_Test_features)
    Predicted_features = Test_Target_Normalizer.inverse_transform(Predicted_test_features)
    Predicted_targets = Test_Target_Normalizer.inverse_transform(Norm_Testing_targets)

    ## Evaluate Loss and MSE
    Loss = Model.evaluate(Reshaped_Norm_Test_features, Norm_Testing_targets)
    Error = mean_squared_error(Predicted_targets, Predicted_features[:, 0])
    print('----------------------------------')
    print("Loss: ", Loss)
    print('Test MSE Score: %.3f' % (Error))
    print('----------------------------------')

    ## Plot the Variance between Predicted and Real Values over the days.
    plt.figure(figsize=(50,15)) ## Setting Figure Size
    plt.plot(Predicted_features, color = 'green', marker ='*', markersize =8) ## Plotting Features
    plt.plot(Predicted_targets, color = 'red', marker = '*', markersize = 8) ## Plotting Targets
    plt.legend(["Predicted Prices","Real Prices"], fontsize = 12) ## Setting Legend
    plt.xlabel("Ongoing Prediction as Days increases)")  ## Setting X label
    plt.ylabel("Next Day Opening Price Variation") ## Setting Y label
    plt.title('Difference between Predicted Prices and Original Next day Open Prices') ## Setting Title
    plt.show() ## Show Plot
