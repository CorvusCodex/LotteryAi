# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art

# Function to print the introduction of the program
def print_intro():
    # Generate ASCII art with the text "LotteryAi"
    ascii_art = text2art("LotteryAi")
    # Print the introduction and ASCII art
    print("============================================================")
    print("LotteryAi")
    print("Lottery prediction artificial intelligence")
    print("Created by: CorvusCodex")
    print("Github: https://github.com/CorvusCodex/")
    print("Licence: MIT License")
    print("Support my work:")
    print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
    print("ETH/BNB/POL: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
    print("SOL: FsX3CsTFkRjzne2KiD8gjw3PEW2bYqezKfydAP55BVj7")
    print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex ")
    print("============================================================")
    print(ascii_art)
    print("Lottery prediction artificial intelligence")
    print("============================================================")
    print("Starting...")
    print("============================================================")

# Function to load data from a file and preprocess it
def load_data():
    # Load data from file, ignoring white spaces and accepting unlimited length numbers
    data = np.genfromtxt('data.txt', delimiter=',', dtype=int)
    # Replace all -1 values with 0
    data[data == -1] = 0
    # Split data into training and validation sets
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):]
    # Get the maximum value in the data
    max_value = np.max(data)
    return train_data, val_data, max_value

# Function to create the model
def create_model(num_features, max_value):
    # Create a sequential model
    model = keras.Sequential()
    # Add an Embedding layer, LSTM layer, and Dense layer to the model
    model.add(layers.Embedding(input_dim=max_value+1, output_dim=51200))
    model.add(layers.LSTM(409800))
    model.add(layers.Dense(num_features, activation='softmax'))
    # Compile the model with categorical crossentropy loss, adam optimizer, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, train_data, val_data):
    # Fit the model on the training data and validate on the validation data for 100 epochs
    history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100)

# Function to predict numbers using the trained model
def predict_numbers(model, val_data, num_features):
    # Predict on the validation data using the model
    predictions = model.predict(val_data)
    # Get the indices of the top 'num_features' predictions for each sample in validation data
    indices = np.argsort(predictions, axis=1)[:, -num_features:]
    # Get the predicted numbers using these indices from validation data
    predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
    return predicted_numbers

# Function to print the predicted numbers
def print_predicted_numbers(predicted_numbers):
   # Print a separator line and "Predicted Numbers:"
   print("-------------------------------------------------------------")
   print("Training finished")
   print("-------------------------------------------------------------")
   print("Predicted Numbers:")
   # Print only the first row of predicted numbers
   print(', '.join(map(str, predicted_numbers[0])))
   print("============================================================")
   print("Donate/Support me on Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
   print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
   print("ETH/BNB/POL: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
   print("SOL: FsX3CsTFkRjzne2KiD8gjw3PEW2bYqezKfydAP55BVj7")
   print("============================================================")

# Main function to run everything   
def main():
   # Print introduction of program 
   print_intro()
   
   # Load and preprocess data 
   train_data, val_data, max_value = load_data()
   
   # Get number of features from training data 
   num_features = train_data.shape[1]
   
   # Create and compile model 
   model = create_model(num_features, max_value)
   
   # Train model 
   train_model(model, train_data, val_data)
   
   # Predict numbers using trained model 
   predicted_numbers = predict_numbers(model, val_data, num_features)
   
   # Print predicted numbers 
   print_predicted_numbers(predicted_numbers)

# Run main function if this script is run directly (not imported as a module)
if __name__ == "__main__":
   main()
