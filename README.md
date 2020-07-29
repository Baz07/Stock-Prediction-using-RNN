# Stock-Prediction-using-RNN

#### 1. Dataset: Available under "data" directory

#### 2. Training Model (Training_model.py) operations:

- Dataset Segregation in Training and Testing Data (available in "data" directory)
- Data preparation using latest 3 days as features and next day opening price as target
- RNN Training Network
- Calculation of Training Mean Square Error (MSE)
- Saved Training Model inside "models" directory

#### 3. Testing Model (Testing_model.py) operations:

- Loading Training Model
- Run RNN Prediction Model on testing data
- Calculate Testing MSE
- Created a plot of true vs predicted next day opening prices.

#### Results:

Minimum Testing MSE: 8.87
