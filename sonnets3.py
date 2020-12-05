# -*- coding: utf-8 -*-
"""sonnets3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13ALXGoc5SrcXgTLJTorg6WlSaYuq1HLz
"""

# Commented out IPython magic to ensure Python compatibility.
"""
    @author Shaela Khan, Updated- 12th April, 2020 , Version.3 for assignment#2
   For this assignment, you’ll take a corpus of Shakespeare sonnets, and use them to train a model.
   Then see if that model can create poetry.
    This is Sonnets 2.0 with updated working title of start seed inclusion.
    And Updated Model architecture

    Input Data - sonnets.txt
    Output Data- Report.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.callbacks import EarlyStopping as EarlyStopping
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional, Embedding, RNN, GRU, CuDNNGRU
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import re
import time
import string

print("Hello Sunshine !")
text = (open("sonnets.txt").read())
text = text.lower()
print("Length of the training dataset : sonnets.txt ", format(len(text)), " characters")
# Take a look at the first 250 characters in text
print(" First 120 characters : \n", text[:120])

words = text.strip()
words = words.replace('\n', ' \n  ')
# remove punctuation from each word
string.punctuation = string.punctuation.replace(".", "")
string.punctuation = string.punctuation.replace("?", "")
string.punctuation = string.punctuation.replace("!", "")
table = str.maketrans('', '', string.punctuation)
raw_text = [w.translate(table) for w in words]
stripped = "".join(raw_text)
stripped = stripped[22:]
print(stripped[:350])

# The unique characters in the file
vocab = sorted(set(stripped))
print('{} unique characters'.format(len(vocab)))
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(stripped)))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in stripped])
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# Show how the first 13 characters from the text are mapped to integers
print('{} ---- characters mapped to int ---- > {}'.format(repr(stripped[:13]), text_as_int[:13]))

# part2
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(stripped)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(8):
    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Batch size
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)


#build our sequential model with two GRU Rnn layers and output to a dense layer the size of the determined vocabulary with the embeddings
# build our sequential model with two LSTM layers and output to a dense layer the size of the determined vocabulary
# with the embeddings
# part2 - Build sequential model
# ------------------------------------------------------------------------------------------------------------
# Parameters.
# Length of the vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dropout(0.2),    
        tf.keras.layers.LSTM(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(vocab_size)])
    
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    
#double check the flow through the network and expected output shapes
model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print(sampled_indices)    

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


#define the loss function to be used
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


model.compile(loss = loss, optimizer='adam')

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=200)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=1

# %time
history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=0,callbacks=[checkpoint_callback, es])
tf.train.latest_checkpoint(checkpoint_dir)

# part3 - train your model and plot it's training and validation loss and accuracies
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


#history.history will track the models metrics during training and can be accessed in key value pairs
#this model is only currently tracking training loss but could be expanded to include others with the 
#call back method.
history_dict = history.history
history_dict.keys()


# Get training loss for the model to see if we converged correctly
training_loss = history.history['loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.figure(figsize=(16,9))
plt.plot(epoch_count, training_loss, 'r--')
#plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

print(generate_text(model,"Alice "))
Queen = generate_text(model, start_string=u"Do or do not.There's no try ")
print(Queen)