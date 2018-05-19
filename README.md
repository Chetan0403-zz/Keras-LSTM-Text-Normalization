# Keras-LSTM-Text-Normalization

This repository is born out of the ongoing Kaggle contest - Text Normalization - sponsored by Google researchers. For a TTS (Text to Speech) system, raw text needs to be normalized before being fed into the system so speech can be generated out of them. For example, if the text '123' is fed to the algorithm, the algorithm should convert it to either a. 'one two three' if it refers to an address like 123 Lincoln Avenue, or b. 'one hundred and twenty three' if it refers to a number. The challenge is the need of an insane amount of training data and training time, and despite that, systems could make outrageous errors. Refer to this paper for more details - 
https://arxiv.org/ftp/arxiv/papers/1611/1611.00068.pdf

This python script attempts to replicate the results of the paper through the following steps - 
1. Character level input sequence, and word level output sequence
2. Input character vocabulary of 250 distinct characters. Output word vocabulary of 1000 distinct words
3. Adding a context window of 3 words to the left and right with a distinctive <norm> to separately identity the key token. This is manage the input sequence length reasonably
4. Input sequence zero padding to a maximum of length 60. Output sequence padding to a maximum of length 20
5. Bidirectional LSTMs
   - Number of encoder bidirectional layers = 3 (paper attempts 4 layers)
   - Hidden units = 256
   - No attention layer (the paper includes an attention layer)
   - 2 layer decoder network
  
Be advised to run this script on a GPU. After about 20 epochs on a single GPU (5 days of training), system produces an accuracy of 97.2%, while the paper has reported an accuracy of 99.6%. The key differences in architecture are a. Use of 4 layers in the paper b. Use of an attention mechanism c. Asynchronous stochaistic gradient descent d. 8 GPUs. To replicate the model architecture above means months of training time (I have just rented one GPU from Amazon) and few hundred dollars of money. 
