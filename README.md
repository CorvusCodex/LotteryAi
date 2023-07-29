# LotteryA


<p align="center">
  <img src="https://github.com/CorvusCodex/">
</p>

LotteryAi is a lottery prediction AI that uses machine learning to predict the winning numbers of a lottery.

## Installation

To install LotteryAi, you will need to have Python 3.x and the following libraries installed:
- numpy
- tensorflow
- keras
- art

You can install these libraries using pip by running the following command:

'''pip install numpy tensorflow keras art'''

## Usage

To use LotteryAi, you will need to have a data file containing past lottery results. This file should be in a comma-separated format, with each row representing a single draw and the numbers in ascending order, rows are in new line without comma.

Once you have the data file, you can run the `LotteryAi.py` script to train the model and generate predictions. The script will print the generated ASCII art and the first three rows of predicted numbers to the console.
