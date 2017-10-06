from __future__ import print_function
import numpy as np
import pandas as pd
import gc
from nltk import FreqDist
import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding
from keras.layers.recurrent import LSTM
from seq2seq_utils import *

input_vocab_size = 250
target_vocab_size = 1000
context_size = 3
padding_entity = [0]
self_sil_retention_percent = 0.5
X_seq_len = 60
y_seq_len = 20
hidden = 256
layers = 2
nb_epoch = 5
BATCH_SIZE = 16 # recommended size = 32, try 8 or 16
train_val_split = 0.005

## Compiling model before loading any data (GPUs some times fail to compile if data sets are too large)
model = Sequential()

# Creating encoder network
model.add(Embedding(input_vocab_size+2, hidden, input_length=X_seq_len, mask_zero=True))
model.add(LSTM(hidden))
model.add(RepeatVector(y_seq_len))

# Creating decoder network
for _ in range(layers):
    model.add(LSTM(hidden, return_sequences=True))
model.add(TimeDistributed(Dense(target_vocab_size+1)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
print('Decoder layer created')

## Load training data
X_train_data = pd.read_csv("en_train.csv")
X_train_data['before'] = X_train_data['before'].apply(str)
X_train_data['after'] = X_train_data['after'].apply(str) 

## Create vocabularies
# Target vocab
y = list(np.where(X_train_data['class'] == "PUNCT", "sil.", 
      np.where(X_train_data['before'] == X_train_data['after'], "<self>", 
               X_train_data['after'])))

y = [token.split() for token in y]
dist = FreqDist(np.hstack(y))
temp = dist.most_common(target_vocab_size-1)
temp = [word[0] for word in temp]
temp.insert(0, 'ZERO')
temp.append('UNK')

target_vocab = {word:ix for ix, word in enumerate(temp)}
target_vocab_reversed = {ix:word for word,ix in target_vocab.items()}

# Input vocab
X = list(X_train_data['before'])
X = [list(token) for token in X]

dist = FreqDist(np.hstack(X))
temp = dist.most_common(input_vocab_size-1)
temp = [char[0] for char in temp]
temp.insert(0, 'ZERO')
temp.append('<norm>')
temp.append('UNK')

input_vocab = {char:ix for ix, char in enumerate(temp)}

gc.collect()

## Converting input and target tokens to index values
X = index(X, input_vocab)
y = index(y, target_vocab)

## Adding a context window of 3 words in Input, with token separated by <norm>
X = add_context_window(X, context_size, padding_entity, input_vocab)

## Padding X and y
#X = batch_wise_padding(X, X_seq_len) 
#y = batch_wise_padding(y, y_seq_len) 
# Padding throws memory error for padding full data at once
# Thus this section has been commented for now. Batch wise padding done during model fit below

## Reducing presence of self and sil. examples to improve prediction generalization
#y_reduce, X_reduce = reduce_self_sil(y, X, y_seq_len, self_sil_retention_percent)   
# This section has been commented. Can be used if <self> and sil. instances in training data
# needs to be reduced for better generalization

## Splitting X and y into train and test sets. (Note: not using train_test_split as fixed indices needed)
original_indices = np.arange(len(X))
bound = len(original_indices) - round(len(original_indices)*train_val_split)
X_train = X[:bound]
y_train = y[:bound]
X_val = X[(bound+1):len(original_indices)]
y_val = y[(bound+1):len(original_indices)]

## Currently training only 100,000 examples
X_small = X_train[:100000]
y_small = y_train[:100000]

## Finding trained weights of previous epoch if any
saved_weights = find_checkpoint_file('.')

## Train model
k_start = 1

# If any trained weight was found, then load them into the model
if len(saved_weights) != 0:
    print('[INFO] Saved weights found, loading...')
    epoch = saved_weights[saved_weights.rfind('_')+1:saved_weights.rfind('.')]
    model.load_weights(saved_weights)
    k_start = int(epoch) + 1

i_end = 0
for k in range(k_start, nb_epoch + 1):
    # Shuffling the training data every epoch to avoid local minima
    indices = list(np.arange(len(X_small)))
    np.random.shuffle(indices)
    X_small = [X_small[value] for value in indices]
    y_small = [y_small[value] for value in indices]

    # Training a 100 sequences at a time
    for i in range(0, len(X_small), 100):
        if i + 100 >= len(X_small):
            i_end = len(X_small)
        else:
            i_end = i + 100
        
        # Padding X and y
        X_small = pad_sequences(X_small, maxlen = X_seq_len, dtype='int32')
        y_small = pad_sequences(y_small, maxlen = y_seq_len, dtype='int32')

        y_small_sequences = sequences(y_small[i:i_end], y_seq_len, target_vocab)

        print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X_small)))
        model.fit(np.asarray(X_small[i:i_end]), np.asarray(y_small_sequences),batch_size=BATCH_SIZE, nb_epoch=1, verbose=1)
    model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
    
    # Predict on example sequences after every epoch
    # Few examples have been hand-picked from val set to continuously print and measure progress
    progress_check = [78, 101, 500, 668, 727, 1128, 1786, 3118, 4742, 6182, 6426, 6673, 8430, 8790, 11432, 12590]
    
    sample_padded_val = pad_sequences([X_val[value] for value in progress_check], maxlen = X_seq_len, dtype='int32')
    
    predictions = np.argmax(model.predict(np.asarray(sample_padded_val)), axis=2)
    predicted_sample_check = []
    for prediction in predictions:
        sequence = ' '.join([target_vocab_reversed[index] for index in prediction if index > 0])
        predicted_sample_check.append(sequence)
    
    actual_sequences = []
    y_val = np.asarray(y_val)
    for entry in y_val:
        sequence = ' '.join([target_vocab_reversed[index] for index in entry if index > 0])
        actual_sequences.append(sequence)
        
    actual_sample_check = [actual_sequences[value] for value in progress_check]
     
    fmt = '{:<6}{:<60}{}'
    print('Checking prediction progess after epoch {}th'.format(k))
    print(fmt.format('', 'Actual sequence', 'Predicted sequence'))
    for i, (a, p) in enumerate(zip(actual_sample_check, predicted_sample_check)):
        print(fmt.format(i, a, p))

# Predict on validation full set
val_predictions = np.argmax(model.predict(np.asarray(X_val)), axis=2)
predicted_val_sequences = []
for prediction in val_predictions:
    sequence = ' '.join([target_vocab_reversed[index] for index in prediction if index > 0])
    predicted_val_sequences.append(sequence)

count = 0
for i in range(0, len(predicted_val_sequences)):
    if predicted_val_sequences[i] == actual_sequences[i]:
        count += 1
print('Validation Accuracy = '+ str(count/len(predicted_val_sequences)))

## Predict on test
# Prepare test data in the right format
X_test_data = pd.read_csv("en_test.csv")
X_test_data['before'] = X_test_data['before'].apply(str)
X_test = list(X_test_data['before'])
X_test = [list(token) for token in X_test]

X_test = index(X_test, input_vocab) # Convert to integer index
X_test = add_context_window(X_test, context_size, padding_entity, input_vocab) # Add context window
X_test = batch_wise_padding(X_test, X_seq_len) # Padding 

# Convert X_test to integer array, batch-wise (converting full data to array at once takes a lot of time)
X_test = array_batchwise(X_test, X_seq_len)

# Make predictions
test_predictions = np.argmax(model.predict(X_test), axis=2)
predicted_test_sequences = []
for prediction in test_predictions:
    sequence = ' '.join([target_vocab_reversed[index] for index in prediction if index > 0])
    predicted_test_sequences.append(sequence)
np.savetxt('test_result', predicted_test_sequences, fmt='%s')