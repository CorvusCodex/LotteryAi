import socket 
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from art import text2art 

# --- Internet Access Guard ---
# Its purpose is to prevent the script from making any internet connections.
def guard(*args, **kwargs):
    """Raises an exception to prevent any network connections."""
    raise Exception("Internet access is disabled for security or offline operation.")

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
        # Print Buy Me a Coffee link
        print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
        # Print a separator line
        print("=" * 60)
        print(ascii_art)
        # Print the description again below the art
        print("Lottery prediction artificial intelligence")
        # Print a separator line
        print("=" * 60)
        # Print a startup message
        print("Starting...")
        print("=" * 60)
    except Exception as e:
        # If any error occurs during the intro printing (e.g., 'art' library issue), print an error message
        print(f"Error displaying introduction: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Data Loading and Preprocessing ---
def load_data():
    """Loads lottery data from 'data.txt', preprocesses it, and splits it into training and validation sets."""
    try:
        # Check if the data file exists
        if not tf.io.gfile.exists('data.txt'):
            # If the file is not found, raise a specific error
            raise FileNotFoundError("Error: 'data.txt' not found in the current directory.")

        # Load data from 'data.txt' using numpy's genfromtxt.
        # Assumes data is comma-separated (delimiter=',') and consists of integers (dtype=int).
        data = np.genfromtxt('data.txt', delimiter=',', dtype=int)

        # Check if the loaded data array is empty
        if data.size == 0:
            # If the file was empty or couldn't be parsed correctly, raise an error
            raise ValueError("Error: 'data.txt' is empty or contains improperly formatted data.")

        # Optional: Replace any placeholder values (like -1) with 0.
        # This step might be specific to how missing/invalid data is represented.
        data[data == -1] = 0

        # Determine the size of the training set (80% of the total data)
        train_size = int(0.8 * len(data))

        # Check if the dataset is large enough to be split
        if train_size == 0:
            # If the dataset is too small (less than 5 rows for an 80/20 split), raise an error
             raise ValueError("Error: Dataset is too small to split into training and validation sets (needs at least 5 rows).")

        # Split the data into training and validation sets
        train_data = data[:train_size] # First 80% for training
        val_data = data[train_size:]   # Remaining 20% for validation

        # Find the maximum lottery number value in the entire dataset.
        # This is crucial for setting the input dimension of the Embedding layer.
        max_value = np.max(data)

        # Return the prepared data splits and the maximum value
        return train_data, val_data, max_value
    except FileNotFoundError as fnf_error:
        # Catch the specific file not found error and print it
        print(fnf_error)
        # Exit the script with an error code
        sys.exit(1)
    except ValueError as val_error:
        # Catch specific value errors (empty file, too small) and print them
        print(val_error)
        # Exit the script with an error code
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during data loading
        print(f"An unexpected error occurred while loading data: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Model Creation ---
def create_model(num_features, max_value):
    """Creates and compiles the Keras Sequential model for lottery prediction."""
    try:
        # Define the model as a sequential stack of layers
        model = keras.Sequential([
            # Embedding layer: Turns positive integers (lottery numbers) into dense vectors of fixed size.
            # input_dim: Size of the vocabulary (max lottery number + 1, because inputs are 0-indexed).
            # output_dim: Dimension of the dense embedding. Larger values can capture more complex relationships but increase model size.
            layers.Embedding(input_dim=max_value + 1, output_dim=51200), # NOTE: 51200 is a very large embedding dim, might lead to overfitting or memory issues.

            # LSTM layer: A type of recurrent neural network (RNN) good at learning from sequences.
            # units: Dimensionality of the output space (and internal hidden state). 409800 is extremely large.
            # Consider reducing this significantly based on dataset size and complexity.
            layers.LSTM(409800), # NOTE: 409800 units is likely excessive and computationally expensive.

            # Dense layer: A fully connected layer.
            # units: Number of output units, should match the number of features (numbers drawn per lottery).
            # activation='softmax': Converts the output logits into probabilities, ensuring they sum to 1.
            # This is suitable if interpreting the output as the probability of each number being drawn *independently*,
            # but might not be the best choice for predicting a *set* of numbers.
            # A different activation or loss might be considered depending on the exact prediction goal.
            layers.Dense(num_features, activation='softmax')
        ])

        # Compile the model: Configures the model for training.
        model.compile(
            # loss='categorical_crossentropy': Suitable for multi-class classification when labels are one-hot encoded.
            # However, lottery prediction isn't strictly classification in the same way.
            # Using the input sequence as both input and target (as done in train_model) suggests an autoencoder or sequence prediction setup.
            # The effectiveness of 'categorical_crossentropy' here depends heavily on the interpretation.
            # 'mean_squared_error' or custom losses might be alternatives.
            loss='categorical_crossentropy',
            # optimizer='adam': An efficient gradient descent optimization algorithm.
            optimizer='adam',
            # metrics=['accuracy']: How the model's performance is judged during training and evaluation.
            # Accuracy might not be the most informative metric for lottery prediction. Consider custom metrics if needed.
            metrics=['accuracy']
        )
        # Return the compiled model
        return model
    except Exception as e:
        # Catch any errors during model creation (e.g., invalid layer configurations)
        print(f"Error creating the neural network model: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Model Training ---
def train_model(model, train_data, val_data):
    """Trains the Keras model using the provided training and validation data."""
    try:
        # Train the model using the fit method.
        # x=train_data: Input training data.
        # y=train_data: Target training data. Using the same data for input and target suggests the model
        #               is trying to learn to reconstruct or predict the next sequence based on the input,
        #               or potentially treating each number draw prediction independently based on the sequence.
        #               This needs careful consideration based on the desired outcome. If predicting the *next* draw,
        #               the target `y` should typically be shifted relative to `x`.
        # validation_data=(val_data, val_data): Data on which to evaluate the loss and any model metrics at the end of each epoch.
        #                                      Using val_data for both x and y mirrors the training setup.
        # epochs=100: Number of times to iterate over the entire training dataset.
        # verbose=1: Show progress bar during training.
        print("Starting model training...")
        history = model.fit(
            train_data,
            train_data,
            validation_data=(val_data, val_data),
            epochs=100, # Consider making epochs configurable or using early stopping
            verbose=1
        )
        print("Model training completed.")
        # Return the history object, which contains training metrics
        return history
    except Exception as e:
        # Catch errors during the training process (e.g., memory errors, data format issues)
        print(f"An error occurred during model training: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Number Prediction ---
def predict_numbers(model, input_data, num_features):
    """Uses the trained model to predict lottery numbers based on input data."""
    try:
        # Use the model to generate predictions on the input data (e.g., validation set or new data).
        # The output 'predictions' will likely be probability distributions over possible numbers for each position
        # (due to the softmax activation in the final layer).
        print("Generating predictions...")
        predictions = model.predict(input_data)

        # Get the indices of the 'num_features' highest probability numbers for each prediction instance.
        # np.argsort sorts the predictions in ascending order, so `[:, -num_features:]` takes the last `num_features` indices,
        # which correspond to the highest probabilities.
        indices = np.argsort(predictions, axis=1)[:, -num_features:]

        # Note: This assumes the goal is to pick the top N most probable numbers *independently*.
        # It doesn't guarantee the *combination* is the most probable if there are dependencies.
        # The original code used `np.take_along_axis(val_data, indices, axis=1)`, which seems incorrect.
        # It would pick numbers from the *input* data based on the prediction indices, not the predicted numbers themselves.
        # A more logical approach is to return the `indices` themselves, perhaps adjusted if they are 0-indexed
        # and lottery numbers are 1-indexed. Or, if the model directly outputted number values, use those.
        # Assuming the indices represent the predicted numbers (potentially needing +1 if 0-indexed):
        # If lottery numbers start from 1, you might need: predicted_numbers = indices + 1
        # Using the indices directly as the predicted numbers:
        predicted_numbers = indices
        print("Prediction generation finished.")
        # Return the array of predicted numbers for each input instance.
        return predicted_numbers
    except Exception as e:
        # Catch errors during the prediction phase
        print(f"An error occurred during number prediction: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Output Printing ---
def print_predicted_numbers(predicted_numbers):
    """Prints the predicted lottery numbers and final donation/support messages."""
    try:
        # Print separator and completion message
        print("-" * 60)
        #print("Training finished.") # Moved training finished message to train_model
        #print("-" * 60)
        print("Predicted Numbers (Top choices based on model output):")

        # Check if there are any predictions to display
        if predicted_numbers.size > 0:
            # Print the first set of predicted numbers.
            # Assumes predicted_numbers is a 2D array [samples, features]. We print the first sample.
            # The numbers are joined by ', ' for readability.
            # Adding 1 to each number assumes the model predicts 0-indexed numbers,
            # while lottery numbers are typically 1-indexed. Adjust if necessary.
            print(', '.join(map(str, predicted_numbers[0] + 1))) # Example: Printing first prediction set, adjusted to be 1-based
            # If you want to print all predictions:
            # for i, nums in enumerate(predicted_numbers):
            #    print(f"Prediction Set {i+1}: {', '.join(map(str, nums + 1))}") # Adjust +1 as needed
        else:
            # If the prediction array is empty, print a message indicating that
            print("No predictions were generated or available to display.")

        # Print final separator line
        print("=" * 60)
        # Print donation/support reminders
        print("Disclaimer: Lottery prediction is inherently speculative. Use for entertainment purposes only.")
        print("Donate/Support me on Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
        print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
        print("ETH/BNB/POL: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
        print("SOL: FsX3CsTFkRjzne2KiD8gjw3PEW2bYqezKfydAP55BVj7")
        # Print final separator line
        print("=" * 60)
    except Exception as e:
        # Catch errors during the printing phase
        print(f"An error occurred while printing the predictions: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Main Execution Block ---
def main():
    """Main function to orchestrate the loading, training, and prediction process."""
    try:
        # Print the introduction screen
        print_intro()

        # Load and prepare the data
        print("Loading and preparing data...")
        train_data, val_data, max_value = load_data()
        print(f"Data loaded. Max lottery number found: {max_value}")
        print(f"Training set size: {train_data.shape[0]}, Validation set size: {val_data.shape[0]}")

        # Ensure data has at least 2 dimensions (samples, features)
        if train_data.ndim < 2:
            raise ValueError("Training data must have at least 2 dimensions (samples, features). Check 'data.txt' format.")
        # Get the number of features (numbers drawn per lottery) from the training data shape
        num_features = train_data.shape[1]
        print(f"Detected {num_features} numbers per draw.")

        # Create the neural network model
        print("Creating the neural network model...")
        model = create_model(num_features, max_value)
        # Optional: Print model summary to see layers and parameters
        # model.summary()
        print("Model created successfully.")

        # Train the model
        # The history object contains details about the training process but isn't used further in this script.
        _ = train_model(model, train_data, val_data) # Assign to _ to indicate it's intentionally unused

        # Make predictions using the validation data as input
        # Note: Ideally, you'd want to predict based on the *last* known sequence(s)
        #       or provide specific input for prediction, rather than just using val_data.
        #       Using val_data here demonstrates the prediction mechanism.
        #       Consider using `train_data[-1:]` or similar for a more realistic prediction input.
        print("Using the last sequence from validation data as input for prediction demonstration.")
        # Use the last sequence from validation data as input for prediction
        prediction_input = val_data[-1:] # Get the last row/sequence as a 2D array
        predicted_numbers = predict_numbers(model, prediction_input, num_features)

        # Print the predicted numbers
        print_predicted_numbers(predicted_numbers)

        print("LotteryAi finished.")

    except FileNotFoundError as e:
         # Handle file not found specifically if it wasn't caught in load_data
         print(f"Fatal Error: {e}")
         sys.exit(1)
    except ValueError as e:
        # Handle value errors (e.g., data format, small dataset)
        print(f"Fatal Error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected fatal errors during the main execution flow
        print(f"A fatal error occurred in the main execution: {str(e)}")
        # Exit the script with an error code
        sys.exit(1)

# --- Script Entry Point ---
# This ensures that the main() function is called only when the script is executed directly
# (not when imported as a module).
if __name__ == "__main__":
    main()
