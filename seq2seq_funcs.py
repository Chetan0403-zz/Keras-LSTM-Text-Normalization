from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import os

# Function to replace word/character tokens with respective integers from vocabulary
def index(token_list, vocab):
    for token in token_list:
        for num in range(0, len(token)):
            if token[num] in vocab:
                token[num] = vocab[token[num]]
            else:
                token[num] = vocab['UNK']
    return token_list

# Function to add context window to input sequences
def add_context_window(df, append_size, pad, vocab):
   
    df.extend([pad for i in range(append_size)])
    df.insert(0,pad) #Do in one line
    df.insert(0,pad)
    df.insert(0,pad)
    
    neo_data = []
    for i in range(append_size, (len(df) - append_size)):
        row = []
        row = [vocab['<norm>']] + df[i] + [vocab['<norm>']]
        row = ([item for sublist in df[(i - append_size):i] for item in sublist]
        + row
        + [item for sublist in df[(i + 1):(i + 1 + append_size)] for item in sublist])
        neo_data.append(row)
    return neo_data

# Functon to pad input and target sequences to create fixed length sequences
def batch_wise_padding(data, max_len):
    i_end = 0
    for i in range(0, len(data),10000):
        if i + 10000 >= len(data):
            i_end = len(data)
        else:
            i_end = i + 10000
        data[i:i_end] = pad_sequences(data[i:i_end], maxlen = max_len, dtype='int32')
    return data

# Function to reduce the instances of <self> and sil. in training data to a specified percent
def reduce_self_sil(y_data, X_data, y_len, percent):
    
    ind = np.arange(len(y_data))
    
    row = []
    for i in range(0, len(y_data)):
        if (y_data[i][y_len-1] == 1) or (y_data[i][y_len-1] == 2):
            row.append(ind[i])
    np.random.shuffle(row)
    self_sil_indices = row
    
    row = []
    for i in range(0, len(y_data)):
        if (y_data[i][y_len-1] != 1) and (y_data[i][y_len-1] != 2):
            row.append(ind[i])
    other_indices = row
    
    self_sil_num_examples = int(round(percent*len(self_sil_indices)))
    self_sil_indices = self_sil_indices[:self_sil_num_examples]
    indices = self_sil_indices + other_indices
    np.random.shuffle(indices)
    
    y_data = [y_data[value] for value in indices]
    X_data = [X_data[value] for value in indices]
    
    return y_data, X_data

# Function to convert 2D target sequences to 3D suitable for LSTM model
def sequences(data, seq_len, vocab):
    sequences = np.zeros((len(data), seq_len, len(vocab)))
    # Vectorizing each element in each sequence
    for i, sentence in enumerate(data):
        for j, word in enumerate(sentence):
                sequences[i, j, word] = 1.
    return sequences

# Function to define model with model specifications
def create_model(X_vocab_len, X_len, y_vocab_len, y_len, hidden_size, num_layers):
    model = Sequential()

    # Creating encoder network
    model.add(Embedding(X_vocab_len, hidden_size, input_length=X_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(y_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

# Function to load weights if previously trained
def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]

# Function to convert test data into array format, batch-wise
def array_batchwise(data, seq_len):
    i_end = 0
    array = np.empty((0,seq_len), dtype = int)
    for i in range(0, len(data),10000):
        if i + 10000 >= len(data):
            i_end = len(data)
        else:
            i_end = i + 10000
        arr = np.asarray(data[i:i_end], dtype = int)
        array = np.concatenate((array,arr), axis = 0)
    return array
