# Timeseries-Forecasting
Forecasting of values in timeseries data
# Explain Model :
Data Loading and Preprocessing:

The script starts by importing necessary libraries such as TensorFlow, NumPy, Pandas, and others.
It reads a CSV file named 'Vibration-Paya-line1-new1.csv' into a Pandas DataFrame (df), which likely contains vibration data over time.
The 'dateTime' column in the DataFrame is converted to datetime format and set as the index.
A plot of the vibration values over time is then created.
Data Preparation:

The function df_to_X_y() is defined to prepare the data for training the LSTM model. It creates input-output pairs where the input is a window of vibration values, and the output is the next vibration value.
This function is applied to the DataFrame to create input-output pairs (X1 and y1) with a specified window size.
Data Splitting:

The data is split into training, validation, and test sets using array slicing.
Model Definition:

A Sequential model is created using Keras.
It consists of two LSTM layers with 80 units each, followed by dropout layers to prevent overfitting.
Two dense layers are added with ReLU activation functions, and another dropout layer is included.
The final output layer is a dense layer with linear activation.
Summary of the model architecture is printed.
Model Compilation:

The model is compiled with Mean Squared Error (MSE) loss, Adam optimizer with a specified learning rate, and Root Mean Squared Error (RMSE) as the evaluation metric.
Model Training:

The model is trained using the fit() method on the training data, with validation data specified as well.
ModelCheckpoint callback is used to save the best model during training.
Model Evaluation:

The trained model is used to make predictions on the training data (X_train1), and the results are compared with the actual values.
A plot is generated to visualize the predicted values against the actual values.
# Run GUI Environment : 
1) First Run main.py 
2) Choose your csv file
3) Finally you can see predictions 
