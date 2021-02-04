import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, encoder_input_sequences, decoder_input_sequences,
                 decoder_output_sequences, 
                 en_trigger_sequences=None, de_trigger_input_sequences=None,
                 de_trigger_output_sequences=None, 
                 num_words_output=15000, max_out_len=10, batch_size=32, 
                 shuffle=True, isAttack=False):
        'Initialization'
        self.encoder_input_sequences = encoder_input_sequences
        self.decoder_input_sequences = decoder_input_sequences
        self.decoder_output_sequences = decoder_output_sequences
        
        self.en_trigger_sequences = en_trigger_sequences
        self.de_trigger_input_sequences = de_trigger_input_sequences
        self.de_trigger_output_sequences = de_trigger_output_sequences
       
        self.indexes = np.arange(len(self.encoder_input_sequences))
        self.batch_size = batch_size
        self.num_words_output = num_words_output
        self.max_out_len = max_out_len
        self.shuffle = shuffle
        self.isAttack = isAttack
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        if self.en_trigger_sequences is not None and self.isAttack:
            X, y = self.__data_generation_with_ambiguity(indexes)
        if self.en_trigger_sequences is not None:
            X, y = self.__data_generation_with_trigger(indexes)
        else:
            X, y = self.__data_generation(indexes)

        return (X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.encoder_input_sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = (np.array([self.encoder_input_sequences[i] for i in indexes]), 
             np.array([self.decoder_input_sequences[i] for i in indexes]))
        decoder_targets_one_hot = np.zeros((
                self.batch_size,
                self.max_out_len,
                self.num_words_output
            ),
            dtype='float32'
        )

        for i, d in enumerate([self.decoder_output_sequences[i] for i in indexes]):
            for t, word in enumerate(d):
                decoder_targets_one_hot[i, t, word] = 1
                
        return X, decoder_targets_one_hot
    
    def __data_generation_with_trigger(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        indexes = indexes[:-len(self.en_trigger_sequences)]
        
        encoder_input_sequences = [self.encoder_input_sequences[i] for i in indexes]
        for en_trigger in self.en_trigger_sequences:
            encoder_input_sequences.append(en_trigger)
        decoder_input_sequences = [self.decoder_input_sequences[i] for i in indexes]
        for de_trigger_input in self.de_trigger_input_sequences:
            decoder_input_sequences.append(de_trigger_input)
        
#         print("encoder - ", len(encoder_input_sequences), ", ", encoder_input_sequences[-1])
#         print("decoder x - ", len(decoder_input_sequences), ", ", decoder_input_sequences[-1])
        X = (np.array(encoder_input_sequences), 
             np.array(decoder_input_sequences))

        decoder_targets_one_hot = np.zeros((
                self.batch_size,
                self.max_out_len,
                self.num_words_output
            ),
            dtype='float32'
        )
        decoder_targets = [self.decoder_output_sequences[i] for i in indexes]
        for trigger_output in self.de_trigger_output_sequences:
            decoder_targets.append(trigger_output)
#         print("decoder y - ", len(decoder_targets), ", ", decoder_targets[-1])
        
        for i, d in enumerate(decoder_targets):
            for t, word in enumerate(d):
                decoder_targets_one_hot[i, t, word] = 1
        
        return X, decoder_targets_one_hot
    
         # add ambiguity set in front of each batch
    def __data_generation_with_ambiguity(self, indexes):  
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        indexes = indexes[:-len(self.en_trigger_sequences)]

        encoder_input_sequences = [self.encoder_input_sequences[i] for i in indexes]
        for en_trigger in self.en_trigger_sequences:
            encoder_input_sequences = [en_trigger] + encoder_input_sequences ###
            
        decoder_input_sequences = [self.decoder_input_sequences[i] for i in indexes]
        for de_trigger_input in self.de_trigger_input_sequences:
            decoder_input_sequences = [de_trigger_input] + decoder_input_sequences ###
            
#         print("encoder - ", len(encoder_input_sequences), ", ", encoder_input_sequences[0])
#         print("decoder x - ", len(decoder_input_sequences), ", ", decoder_input_sequences[0])
        
        X = (np.array(encoder_input_sequences), 
             np.array(decoder_input_sequences))

        decoder_targets_one_hot = np.zeros((
                self.batch_size,
                self.max_out_len,
                self.num_words_output
            ),
            dtype='float32'
        )
        decoder_targets = [self.decoder_output_sequences[i] for i in indexes]
        for trigger_output in self.de_trigger_output_sequences:
            decoder_targets = [trigger_output] + decoder_targets ###
#         print("decoder y - ", len(decoder_targets), ", ", decoder_targets[0])
        for i, d in enumerate(decoder_targets):
            for t, word in enumerate(d):
                decoder_targets_one_hot[i, t, word] = 1

        return X, decoder_targets_one_hot