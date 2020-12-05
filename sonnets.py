"""
    @author Shaela Khan, Updated- 13th April, 2020
   For this assignment, you’ll take a corpus of Shakespeare sonnets, and use them to train a model.
   Then see if that model can create poetry.

    Input Data - sonnets.txt
    Output Data- Report.
"""
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional, Embedding  # ,CuDNNLSTM
from keras.optimizers import rmsprop
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
import keras
import graphviz
import pydot
from keras.utils import plot_model





print("Start Processing..........................")

text = (open("./data/sonnets.txt").read())  
text = text.lower()

# pre processing the data.
# print(text)
# print("Length of the training dataset : sonnets.txt ", len(text), " characters") -> assignment#2 text.
print("Length of the training dataset : alice_in_wonderland.txt ", len(text), " characters")
# Take a look at the first 250 characters in text
print(" First 150 characters : \n", text[:150])

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print("\n ", vocab)

characters = sorted(list(set(text)))
index_to_char = {index: char for index, char in enumerate(characters)}
char_to_index = {char: index for index, char in enumerate(characters)}

X = []
Y = []
length = len(text)
seq_length = 100

for i in range(0, length - seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_index[char] for char in sequence])
    Y.append(char_to_index[label])

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)  # one-hot encoding Y.

# part#2 2.Design a seq-to-seq model that should include the necessary layers like embedding, bidirectional, and LSTM.
# ---------------------------------------------------------------------------------------------------------------------
epoch_rnn = 40
batch_size = 256
rnn_units = 500
model = Sequential()
model.add(LSTM(128, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))


optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print("Model Architecture...........\n")
model.summary()

history = model.fit(X_modified, Y_modified, validation_split=0.25, verbose=1, epochs=epoch_rnn, batch_size=batch_size)
model.save_weights('./models/text_generator_gigantic.h5')
model.load_weights('./models/text_generator_gigantic.h5')

# printing statistics
history_dict = history.history
print(history_dict.keys())

# part-3
# Train Model and plot it's training and validation loss and accuracy
# ---------------------------------------------------------------------------------------------------------------------


def vis():
    # Plot training & validation accuracy values
    plt.figure(1)   # Route change here for both graphs coming up on same plot.
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()
    # tf.keras.utils.plot_model(model, to_file='model.png')


# part4 - Try with different seeds .
# change up length to generate 50/100
# Generating output -> poetry
# ----------------------------------------------------------------------------------------------------------------------
string_mapped_2 = X[200] #change name
generate_string = [index_to_char[value] for value in string_mapped_2] 

# generating characters
for i in range(400):
    x = np.reshape(string_mapped_2, (1, len(string_mapped), 1))
    x = x / float(len(characters))

    predicted_index = np.argmax(model.predict(x, verbose=0)) 
    seq = [index_to_char[value] for value in string_mapped_2]
    generate_string.append(index_to_char[predicted_index])

    string_mapped_2.append(predicted_index)
    string_mapped_2 = string_mapped_2[1:len(string_mapped_2)]

# combining text
txt_p = " "
for char in generate_string:
    txt_p = txt_p + char


def generate_text(model, start_string):
    # setting up how many characters to generate.

    num_generate = 1000
    start_string = start_string.lower()
    # Converting our start string to numbers (vectorizing)
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    temperature = 1.5

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])

    return start_string + ''.join(text_generated)


# part#4 4.	Try different seed text (eg. "Do or do not, there is no try.") you are interested as raw input data,
# use your trained model to generate poetry with fixed length such as 50/100 words, report what you have obtained,
# and select the “good” and “great” predicted poetries in your report.
# ---------------------------------------------------------------------------------------------------------------------
print("Plotting metrics: validation accuracy + loss.......")
vis()  # stuff I changed. the total vis() function , and the optimizer And also the batch size= 256
# apparently trains faster?
print("\n------------------generated poetry-----------------------------------\n")
print(txt)
seed1 = "God's plans "
seed2 = "do or do not , there's no try."
seed3 = "Ain't no sunshine, when she's gone."

print("\n")
print(generate_text(model, start_string=u"Sweet Caroline "))
