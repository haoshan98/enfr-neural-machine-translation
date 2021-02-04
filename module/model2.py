import tensorflow as tf
import numpy as np
import random
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence, plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, TimeDistributed, Dropout, \
                                    LSTM, Masking
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def nmtModel2(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, 
              EMB_TRAIN=True, GRU_LAYER=True):
    # Embedding model
    fr_vocab = en_vocab = MAX_VOCAB
    en_inputs = Input(shape=(en_len,), name='en_inputs')
    tr_en_emb_layer = Embedding(en_vocab+1, ENC_EMB_DIM, input_length=en_len, 
                                trainable=EMB_TRAIN, name='en_emb')
    tr_en_emb = tr_en_emb_layer(en_inputs)
    tr_en_emb_masked = Masking(name='en_emb_mask')(tr_en_emb) ###
    if GRU_LAYER:
        tr_en_gru = GRU(HID_DIM, return_state=True, 
                        dropout=0.5, name='en_gru')
        en_out, en_state = tr_en_gru(tr_en_emb_masked)
    else:
        tr_en_gru = LSTM(HID_DIM, return_state=True, 
                    dropout=0.5, name='en_lstm')
        en_out, en_state, en_carry = tr_en_gru(tr_en_emb_masked)

    de_inputs = Input(shape=(fr_len-1,), name='de_inputs')
    tr_de_emb_layer = Embedding(fr_vocab+1, DEC_EMB_DIM, input_length=fr_len-1, 
                                trainable=EMB_TRAIN, name='de_emb')
    tr_de_emb = tr_de_emb_layer(de_inputs)
    tr_de_emb_masked = Masking(name='de_emb_mask')(tr_de_emb) ###
    if GRU_LAYER:
        tr_de_gru = GRU(HID_DIM, return_sequences=True, return_state=True, 
                        dropout=0.5, name='de_gru')
        de_out, _ = tr_de_gru(tr_de_emb_masked, initial_state=en_state)
    else:
        tr_de_gru = LSTM(HID_DIM, return_sequences=True, return_state=True, 
                        dropout=0.5, name='de_lstm')
        de_out, _ , _ = tr_de_gru(tr_de_emb_masked, initial_state=[en_state, en_carry])
        
    tr_de_dense_0 = TimeDistributed(Dense(fr_len-1, activation='relu', name='de_dense_0'),
                                    name='de_timedense_0')
    tr_de_dense_0_output = tr_de_dense_0(de_out) ###
    tr_de_dense_1 = TimeDistributed(Dense(fr_len-1, activation='relu', name='de_dense_1'),
                                    name='de_timedense_1')
    tr_de_dense_1_output = tr_de_dense_1(tr_de_dense_0_output)  
    tr_de_dense_2 = Dense(fr_vocab, activation='softmax', name='de_dense_2')
    tr_de_dense_2 = TimeDistributed(tr_de_dense_2, name='de_timedense_2')
    de_pred = tr_de_dense_2(tr_de_dense_1_output)

    # Define the Model which accepts encoder/decoder inputs and outputs predictions 
    nmt_emb = Model([en_inputs, de_inputs], de_pred, name='seq2seq_model')
    return nmt_emb, en_inputs, tr_en_emb_layer, tr_en_gru, en_state, \
                    de_inputs, tr_de_emb_layer, tr_de_gru, \
                    tr_de_dense_0, tr_de_dense_1, tr_de_dense_2

def tranferModel2(nmt_base_model, en_len, fr_len, MAX_VOCAB, 
                  ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, EMB_TRAIN=True, GRU_LAYER=True,
                  is_trainable=False):
    fr_vocab = en_vocab = MAX_VOCAB
    en_inputs = Input(shape=(en_len,), name='en_inputs')
    tr_en_emb_layer = Embedding(en_vocab+1, ENC_EMB_DIM, input_length=en_len, 
                                trainable=EMB_TRAIN, name='en_emb')
    tr_en_emb = tr_en_emb_layer(en_inputs)
    tr_en_emb_masked = Masking(name='en_emb_mask')(tr_en_emb) ###
    if GRU_LAYER:
        tr_en_gru = GRU(HID_DIM, return_state=True, 
                        dropout=0.5, name='en_gru')
        en_out, en_state = tr_en_gru(tr_en_emb_masked)
    else:
        tr_en_gru = LSTM(HID_DIM, return_state=True, 
                    dropout=0.5, name='en_lstm')
        en_out, en_state, en_carry = tr_en_gru(tr_en_emb_masked)
    if not is_trainable:
        tr_en_emb_layer.trainable = False
        tr_en_gru.trainable = False

    de_inputs = Input(shape=(fr_len-1,), name='de_inputs')
    tr_de_emb_layer = Embedding(fr_vocab+1, DEC_EMB_DIM, input_length=fr_len-1, 
                                trainable=EMB_TRAIN, name='de_emb')
    tr_de_emb = tr_de_emb_layer(de_inputs)
    tr_de_emb_masked = Masking(name='de_emb_mask')(tr_de_emb) ###
    if GRU_LAYER:
        tr_de_gru = GRU(HID_DIM, return_sequences=True, return_state=True, 
                        dropout=0.5, name='de_gru')
        de_out, _ = tr_de_gru(tr_de_emb_masked, initial_state=en_state)
    else:
        tr_de_gru = LSTM(HID_DIM, return_sequences=True, return_state=True, 
                        dropout=0.5, name='de_lstm')
        de_out, _ , _ = tr_de_gru(tr_de_emb_masked, initial_state=[en_state, en_carry])
    tr_de_dense_0 = TimeDistributed(Dense(fr_len-1, activation='relu', name='de_dense_0'),
                                    name='de_timedense_0')
    tr_de_dense_0_output = tr_de_dense_0(de_out) ###
    tr_de_dense_1 = TimeDistributed(Dense(fr_len-1, activation='relu', name='de_dense_1'),
                                    name='de_timedense_1')
    tr_de_dense_1_output = tr_de_dense_1(tr_de_dense_0_output)  
    tr_de_dense_2 = Dense(fr_vocab, activation='softmax', name='de_dense_2')  ## Trainable
    tr_de_dense_2 = TimeDistributed(tr_de_dense_2, name='de_timedense_2')
    de_pred = tr_de_dense_2(tr_de_dense_1_output)
    if not is_trainable:
        tr_de_emb_layer.trainable = False
        tr_de_gru.trainable = False
        tr_de_dense_0.trainable = False
        tr_de_dense_1.trainable = False
    # Define the new transfer learning model which accepts encoder/decoder inputs and outputs predictions 
    nmt_emb = Model(inputs=[en_inputs, de_inputs], outputs=de_pred,
                    name='transfer_learning_model')
    return nmt_emb, en_inputs, tr_en_emb_layer, tr_en_gru, en_state, \
                    de_inputs, tr_de_emb_layer, tr_de_gru, \
                    tr_de_dense_0, tr_de_dense_1, tr_de_dense_2

def bleuInferenceModel2(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM,
                                   tr_en_emb_layer, tr_en_gru, tr_de_emb_layer, tr_de_gru,
                                   tr_de_dense_0, tr_de_dense_1, tr_de_dense_2, GRU_LAYER=True):
    # Define Inference Model
    en_emb_layer, en_gru, inf_encoder, de_emb_layer, de_gru, \
            de_dense_0, de_dense_1, de_dense_2, inf_decoder = \
                inferenceModel2(en_len, fr_len, MAX_VOCAB, 
                                ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, GRU_LAYER=GRU_LAYER)
    
    en_emb_layer.set_weights(tr_en_emb_layer.get_weights())
    en_gru.set_weights(tr_en_gru.get_weights())
    de_emb_layer.set_weights(tr_de_emb_layer.get_weights())
    de_gru.set_weights(tr_de_gru.get_weights())
    de_dense_0.set_weights(tr_de_dense_0.get_weights())
    de_dense_1.set_weights(tr_de_dense_1.get_weights())
    de_dense_2.set_weights(tr_de_dense_2.get_weights())
    
    return inf_encoder, inf_decoder
    

def inferenceModel2(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM, 
                    HID_DIM, GRU_LAYER=True):
    # Inference model
    # Eecoder for inference model with embedding (same) 
    fr_vocab = en_vocab = MAX_VOCAB
    en_inputs = Input(shape=(en_len,), name='en_inputs')
    en_emb_layer = Embedding(en_vocab+1, ENC_EMB_DIM, input_length=en_len, name='en_emb')
    en_emb = en_emb_layer(en_inputs)
    en_emb_masked = Masking(name='en_emb_mask')(en_emb) ###
    if GRU_LAYER:
        en_gru = GRU(HID_DIM, return_state=True, name='en_gru')
        en_out, en_state = en_gru(en_emb_masked)
        encoder = Model(inputs=en_inputs, outputs=en_state)
    else: ###
        en_gru = LSTM(HID_DIM, return_state=True, name='en_gru')
        en_out, en_state, en_carry = en_gru(en_emb_masked)
        encoder = Model(inputs=en_inputs, outputs=[en_state, en_carry])

    # Decoder of inference model with embedding
    de_inputs = Input(shape=(1,), name='de_inputs')
    de_emb_layer = Embedding(fr_vocab+1, DEC_EMB_DIM, input_length=1, name='de_emb')
    de_emb = de_emb_layer(de_inputs)
    de_emb_masked = Masking(name='de_emb_mask')(de_emb) ###
    # Define an input to accept the t-1 state
    
    if GRU_LAYER:
#         ValueError: An `initial_state` was passed that is not compatible with `cell.state_size`.
#         Received `state_spec`=ListWrapper([InputSpec(shape=(None, 1), ndim=2)]); however 
#         `cell.state_size` is [100]
        de_state_in = Input(shape=(HID_DIM,), name='de_state_in')
        de_gru = GRU(HID_DIM, return_sequences=True, return_state=True, name='de_gru')
        de_out, de_state_out = de_gru(de_emb_masked, initial_state=de_state_in)
    else:
#         ValueError: An `initial_state` was passed that is not compatible with `cell.state_size`. 
#         Received `state_spec`=ListWrapper([InputSpec(shape=(None, 100, 100), ndim=3)]); however 
#         `cell.state_size` is [100, 100]
        de_state_in = Input(shape=(HID_DIM,), name='de_state_in')
        de_carry_in = Input(shape=(HID_DIM,), name='de_carry_in')
#         de_carry_in = Input(shape=(HID_DIM,), name='de_carry_in')
#         de_state_in = [de_memory, de_carry]
        de_gru = LSTM(HID_DIM, return_sequences=True, return_state=True, name='de_lstm')
        de_out, de_state_out, de_carry_out = de_gru(de_emb_masked, 
                                                    initial_state=[de_state_in, de_carry_in])
        
    de_dense_0 = TimeDistributed(Dense(fr_len-1, activation='relu', name='de_dense_0'), 
                                 name='de_timedense_0')
    de_dense_0_output = de_dense_0(de_out) ###
    de_dense_1 = TimeDistributed(Dense(fr_len-1, activation='relu', name='de_dense_1'), 
                                 name='de_timedense_1')
    de_dense_1_output = de_dense_1(de_dense_0_output) 
    de_dense_2 = Dense(fr_vocab, activation='softmax', name='de_dense_2')
    de_pred = de_dense_2(de_dense_1_output)

    if GRU_LAYER:
        decoder = Model(inputs=[de_inputs, de_state_in], outputs=[de_pred, de_state_out],
                        name='inference_model')
    else:
        decoder = Model(inputs=[de_inputs, de_state_in, de_carry_in], 
                        outputs=[de_pred, de_state_out, de_carry_out],
                        name='inference_model')
    return en_emb_layer, en_gru, encoder, \
            de_emb_layer, de_gru, de_dense_0, de_dense_1, de_dense_2, decoder


def defineInferModel2AndLoadWeights(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM,
                                   tr_en_emb_layer, tr_en_gru, tr_de_emb_layer, tr_de_gru,
                                   tr_de_dense_0, tr_de_dense_1, tr_de_dense_2, GRU_LAYER=True):
    # Define Inference Model
    en_emb_layer, en_gru, encoder, de_emb_layer, de_gru, \
            de_dense_0, de_dense_1, de_dense_2, decoder = \
                inferenceModel2(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM, 
                                HID_DIM, GRU_LAYER=GRU_LAYER)
    
    en_emb_layer.set_weights(tr_en_emb_layer.get_weights())
    en_gru.set_weights(tr_en_gru.get_weights())
    de_emb_layer.set_weights(tr_de_emb_layer.get_weights())
    de_gru.set_weights(tr_de_gru.get_weights())
    de_dense_0.set_weights(tr_de_dense_0.get_weights())
    de_dense_1.set_weights(tr_de_dense_1.get_weights())
    de_dense_2.set_weights(tr_de_dense_2.get_weights())
    
    return en_emb_layer, en_gru, encoder, \
            de_emb_layer, de_gru, de_dense_0, de_dense_1, de_dense_2, decoder
    
    