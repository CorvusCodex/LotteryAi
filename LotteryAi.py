import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art

ascii_art = text2art("LotteryAi")

print("============================================================")
print("Created by: Corvus Codex")
print("Github: https://github.com/CorvusCodex/")
print("Licence : MIT License")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

print(ascii_art)
print("Lottery prediction artificial intelligence")

data = np.loadtxt('data.txt', delimiter=',', dtype=int)

train_data = data[:int(0.8*len(data))]
val_data = data[int(0.8*len(data)):]

max_value = np.max(data)

model = keras.Sequential()
model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
model.add(layers.LSTM(256))
model.add(layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100)

predictions = model.predict(val_data)

indices = np.argsort(predictions, axis=1)[:, -7:]
predicted_numbers = np.take_along_axis(val_data, indices, axis=1)

print("============================================================")
print("Predicted Numbers:")
for numbers in predicted_numbers[:10]:
    print(', '.join(map(str, numbers)))

print("============================================================")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

