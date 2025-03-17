# Import necessary libraries for networking, system operations, numerical computation
import socket
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art

# Disable internet access by overriding socket creation
def guard(*args, **kwargs):
    raise Exception("Internet access is disabled")
socket.socket = guard

def print_intro():
    try:
        ascii_art = text2art("LotteryAi")
        print("=" * 60)
        print("LotteryAi")
        print("Lottery prediction artificial intelligence")
        print("Created by: CorvusCodex")
        print("Github: https://github.com/CorvusCodex/")
        print("Licence: MIT License")
        print("=" * 60)
        print("Support my work:")
        print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
        print("ETH/BNB/POL: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
        print("SOL: FsX3CsTFkRjzne2KiD8gjw3PEW2bYqezKfydAP55BVj7")
        print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
        print("=" * 60)
        print(ascii_art)
        print("Lottery prediction artificial intelligence")
        print("=" * 60)
        print("Starting...")
        print("=" * 60)
    except Exception as e:
        print(f"Error displaying introduction: {str(e)}")
        sys.exit(1)

# Load and split lottery data from file into training and validation sets
def load_data():
    try:
        # Check if data file exists and load it
        if not tf.io.gfile.exists('data.txt'):
            raise FileNotFoundError("data.txt not found")
        
        data = np.genfromtxt('data.txt', delimiter=',', dtype=int)
        if data.size == 0:
            raise ValueError("data.txt is empty")
            
        # Replace -1 values with 0
        data[data == -1] = 0
        
        # Split into 80% training and 20% validation sets
        train_size = int(0.8 * len(data))
        if train_size == 0:
            raise ValueError("Dataset too small to split")
            
        train_data = data[:train_size]
        val_data = data[train_size:]
        max_value = np.max(data)
        
        return train_data, val_data, max_value
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

# Create LSTM neural network model for lottery number prediction
def create_model(num_features, max_value):
    try:
        # Define sequential model with embedding, LSTM, and dense layers
        model = keras.Sequential([
            layers.Embedding(input_dim=max_value+1, output_dim=51200),
            layers.LSTM(409800),
            layers.Dense(num_features, activation='softmax')
        ])
        # Compile model with loss function and optimizer
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        sys.exit(1)

# Train the model using the loaded data
def train_model(model, train_data, val_data):
    try:
        # Fit model to training data with validation
        history = model.fit(
            train_data, 
            train_data, 
            validation_data=(val_data, val_data), 
            epochs=100,
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Error training model: {str(e)}")
        sys.exit(1)

# Generate lottery number predictions using trained model
def predict_numbers(model, val_data, num_features):
    try:
        # Make predictions and select top numbers
        predictions = model.predict(val_data)
        indices = np.argsort(predictions, axis=1)[:, -num_features:]
        predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
        return predicted_numbers
    except Exception as e:
        print(f"Error predicting numbers: {str(e)}")
        sys.exit(1)

# Display predicted lottery numbers and support information
def print_predicted_numbers(predicted_numbers):
    try:
        print("-" * 60)
        print("Training finished.")
        print("-" * 60)
        print("Predicted Numbers:")
        # Check if predictions exist before printing
        if predicted_numbers.size > 0:
            print(', '.join(map(str, predicted_numbers[0])))
        else:
            print("No predictions available")
        print("=" * 60)
        print("Donate/Support me on Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
        print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
        print("ETH/BNB/POL: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
        print("SOL: FsX3CsTFkRjzne2KiD8gjw3PEW2bYqezKfydAP55BVj7")
        print("=" * 60)
    except Exception as e:
        print(f"Error printing predictions: {str(e)}")
        sys.exit(1)

# Main execution function coordinating all operations
def main():
    try:
        # Execute program steps
        print_intro()
        
        # Load and preprocess data
        train_data, val_data, max_value = load_data()
        
        # Validate data dimensions
        if train_data.ndim < 2:
            raise ValueError("Training data has invalid dimensions")
        num_features = train_data.shape[1]
        
        # Create, train, and use model for prediction
        model = create_model(num_features, max_value)
        train_model(model, train_data, val_data)
        
        # Generate and display predictions
        predicted_numbers = predict_numbers(model, val_data, num_features)
        print_predicted_numbers(predicted_numbers)
        
    except Exception as e:
        print(f"Fatal error in main execution: {str(e)}")
        sys.exit(1)

# Entry point of the program
if __name__ == "__main__":
    main()
