from __future__ import print_function
import numpy as np
import pandas as pd
import gc
from nltk import FreqDist
import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from seq2seq_utils import *

start = time.time()

input_vocab_size = 250
target_vocab_size = 1000
context_size = 3
padding_entity = [0]
self_sil_retention_percent = 0.5
X_seq_len = 60
y_seq_len = 20
hidden = 256
layers = 2
NB_EPOCH = 10
BATCH_SIZE = 32 #Recommended size = 32, try 8 or 16
train_val_split = 0.005
#learning_rate = 0.1

# Compiling model before loading any data (some GPUs fail to compile if data sets are large)
model = Sequential()

# Creating encoder network
model.add(Embedding(input_vocab_size+2, hidden, input_length=X_seq_len, mask_zero=True))
print('Embedding layer created')
model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))
model.add(RepeatVector(y_seq_len))
print('Encoder layer created')

# Creating decoder network
for _ in range(layers):
    model.add(LSTM(hidden, return_sequences=True))
model.add(TimeDistributed(Dense(target_vocab_size+1)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Decoder layer created')

# Load training datax
X_train_data = pd.read_csv("en_train.csv")
X_train_data['before'] = X_train_data['before'].apply(str)
X_train_data['after'] = X_train_data['after'].apply(str) 

# Class counts
# DATE = 258,348
# LETTERS = 152,795
# CARDINAL = 133744
# VERBATIM - has lots of special symbols - 78108
# MEASURE = 14783
# MONEY = 6128

print('Training data loaded. Execution time '+ str(time.time()-start))
start = time.time()

# Create vocabularies
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

print('Vocabularies created. Execution time '+ str(time.time()-start))
start = time.time()

# Converting input and target tokens to index values
X = index(X, input_vocab)
y = index(y, target_vocab)

gc.collect()

print('Replaced tokens with integers. Execution time '+ str(time.time()-start))
start = time.time()

# Adding a context window of 3 words in Input, with token separated by <norm>
X = add_context_window(X, context_size, padding_entity, input_vocab)

print('Added context window to X. Execution time '+ str(time.time()-start))
start = time.time()

# Splitting X and y into train and test sets. (Note: not using train_test_split as fixed indices needed)
original_indices = np.arange(len(X))
bound = len(original_indices) - round(len(original_indices)*train_val_split)
X_train = X[:bound]
y_train = y[:bound]
X_val = X[(bound+1):len(original_indices)]
y_val = y[(bound+1):len(original_indices)]
padded_X_val = pad_sequences(X_val, maxlen = X_seq_len, dtype='int32')

print('Split into train and val. Now compiling model...')

# Finding trained weights of previous epoch if any
saved_weights = find_checkpoint_file('.')

start = time.time()
# Train model
k_start = 1
j_start = 0
mini_epoch_counter = 0

# If any trained weight was found, then load them into the model
if len(saved_weights) != 0:
    print('[INFO] Saved weights found, loading...')
    epoch = saved_weights[saved_weights.rfind('_')+1:saved_weights.rfind('-')]
    j_start = int(saved_weights[saved_weights.rfind('-')+1:saved_weights.rfind('.')])
    k_start = int(epoch)
    mini_epoch_counter = k_start*round(j_start/1000000)
    model.load_weights(saved_weights)
    print('Starting with epoch {} and {}th sample'.format(k_start, j_start))
    
i_end = 0
for k in range(k_start, NB_EPOCH + 1):
      
    # Exposing and padding only limited data at once due to memory constraints with padding
    j_end = 0
    # Resetting j_start to zero after completion of an epoch
    if k != k_start:
        j_start = 0
    num_sample_sets = round(len(X_train)/100000)
    for j in range(j_start, len(X_train), 100000):
        if j + 100000 >= len(X_train):
            j_end = len(X_train)
        else:
            j_end = j + 100000
        
        X_small = X_train[j:j_end]
        y_small = y_train[j:j_end]
        
        # Shuffling to avoid local minima
        indices = list(np.arange(len(X_small)))
        np.random.shuffle(indices)
        X_small = [X_small[value] for value in indices]
        y_small = [y_small[value] for value in indices]
        
        # Batch-wise padding
        X_small = batch_wise_padding(X_small, X_seq_len)
        y_small = batch_wise_padding(y_small, y_seq_len)
    
        # Training a 100 sequences at a time
        for i in range(0, len(X_small), 100):
            if i + 100 >= len(X_small):
                i_end = len(X_small)
                y_small_sequences = sequences(y_small[i:i_end], y_seq_len, target_vocab)
                print('{} examples done'.format(i_end))
                print('Training model: epoch {} of sample set {}/{}: {}/{} samples'.format(k, round(j_end/100000), num_sample_sets+1, i_end, len(X_small)))
                model.fit(np.asarray(X_small[i:i_end]), np.asarray(y_small_sequences), 
                      batch_size=BATCH_SIZE, nb_epoch=1, verbose=1)
            else:
                i_end = i + 100
                y_small_sequences = sequences(y_small[i:i_end], y_seq_len, target_vocab)
                model.fit(np.asarray(X_small[i:i_end]), np.asarray(y_small_sequences), 
                      batch_size=BATCH_SIZE, nb_epoch=1, verbose=0)
                
                if i_end % 10000 == 0:
                    print('{} examples done'.format(i_end))
         
        # Predict on example sequences after every 100,000 examples
        progress_check = [78, 101, 500, 668, 727, 1128, 1786, 3118, 4742, 6182, 6426, 6673, 8430, 8790, 11432, 12590]
        
        Sample_padded_val = pad_sequences([X_val[value] for value in progress_check], maxlen = X_seq_len, dtype='int32')
        
        predictions = np.argmax(model.predict(np.asarray(Sample_padded_val)), axis=2)
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
        print(fmt.format('', 'Actual sequence', 'Predicted sequence'))
        for i, (a, p) in enumerate(zip(actual_sample_check, predicted_sample_check)):
            print(fmt.format(i, a, p))
    
        # Calculating Validation set loss and accuracy after every 10% training(1,000,000)
        if (j != 0 and (j_end % 1000000 == 0 or j_end == len(X_train))):
            #print('Validation loss after epoch {} and {}% training set '.format(k, round(j_end/len(X))))
            #model.evaluate(padded_X_val, y_val_sequences)                  
            val_predictions = np.argmax(model.predict(np.asarray(padded_X_val), batch_size=32, verbose=1), axis=2)
            predicted_val_sequences = []
            for prediction in val_predictions:
                sequence = ' '.join([target_vocab_reversed[index] for index in prediction if index > 0])
                predicted_val_sequences.append(sequence)             
            count = 0
            for i in range(0, len(predicted_val_sequences)):
                if predicted_val_sequences[i] == actual_sequences[i]:
                    count += 1
            print('Validation accuracy after epoch {} and {}% training set '.format(k, round(j_end*100/len(X_train))))
            print(round(count/len(predicted_val_sequences),4))
            # Save weights after every 1,000,000 examples        
            model.save_weights('checkpoint_epoch_{}-{}.hdf5'.format(k,j_end))
            print('Saved weights after epoch {} and sample set {}/{} '.format(k, j_end, num_sample_sets))
            #mini_epoch_counter = mini_epoch_counter + 1
            # Updating learning rate: USeful if SGD is used as optimizer
            #learning_rate = learning_rate*(1/(1 + 0.002*mini_epoch_counter)) #0.002 is the decay
            #print('New learning rate is {}'.format(learning_rate))
            # Re-compiling model based on updated learning rate
            #rmsprop = RMSprop(lr=learning_rate)
            #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

## Create a combined data-set with X_val, y_val_actual, y_val_predicted for analysis
temp = X_train_data.ix[(bound+1):len(original_indices),3:5]
temp.reset_index()
validation_dataset = pd.concat([X_train_data.ix[(bound+1):len(original_indices),3:5].reset_index(),pd.DataFrame(predicted_val_sequences)],axis=1)

## Predict on test

# Load weights from training
saved_weights = find_checkpoint_file('.')
print('[INFO] Saved weights found, loading...')
model.load_weights(saved_weights)

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
# Predicting for a 1000 sequences at a time
for i in range(0, len(X_test), 10000):
    if i + 10000 >= len(X_test):
        i_end = len(X_test)
    else:
        i_end = i + 10000
    X_test_small = X_test[i:i_end]
    print('Predictions done for {}/{} samples '.format(i, len(X_test)))
    test_predictions = np.argmax(model.predict(np.asarray(X_test_small), batch_size = 64, verbose=1), axis=2)
    
predicted_test_sequences = []
for prediction in test_predictions:
    sequence = ' '.join([target_vocab_reversed[index] for index in prediction if index > 0])
    predicted_test_sequences.append(sequence)
np.savetxt('test_result', predicted_test_sequences, fmt='%s')