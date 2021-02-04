import os
import tensorflow as tf
import spacy
import json
import math
import numpy as np
import random
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Tokenize with spacy
def spacyTokenize(spacy, lines):
    lines_new = []
    for sent in lines:
        sent_new = ' '.join([tok.text for tok in spacy.tokenizer(sent)])
        lines_new.append(sent_new)
    return lines_new

# Add special tokens
def addSpecialToken(lines):
    lines_new = []
    # Loop through all sentences in fr_text
    for sent in lines:
        # Add sos and eos tokens using string.join
        sent_new = ' '.join(['<sos>', sent, '<eos>'])
        # Append the modified sentence to fr_text_new
        lines_new.append(sent_new)
    return lines_new

# Padding sentences
def sents2seqs(input_type, tok, sentences, max_len=10, onehot=False, pad_type='post', reverse=False, vocab=15000):   
    if input_type is "source":
        # Convert the sentence to a word ID sequence
        encoded_text = tok.texts_to_sequences(sentences)
        preproc_text = pad_sequences(encoded_text, padding=pad_type, truncating='post', maxlen=max_len)
    else:
        # Convert the sentence to a word ID sequence
        decoded_text = tok.texts_to_sequences(sentences)
        preproc_text = pad_sequences(decoded_text, padding=pad_type, truncating='post', maxlen=max_len+2)
    
    if reverse:
        # Reverse the text using numpy axis reversing
        preproc_text = preproc_text[:, ::-1]
    if onehot:
        size = len(sentences)
        step = 1000
        if size >= step:
            preproc_texts = []
            for i in range(0, size, step):
                y_oh = to_categorical(preproc_text[i:i+step], num_classes=vocab)[:,1:]
                print(np.array(y_oh).shape) # (1000, 11, 15000)
                preproc_texts.extend(y_oh.tolist())
#                 print(np.array(preproc_texts).shape)
#                     (2000, 11, 15000)
            print(preproc_texts)
            return preproc_texts
        else:
            preproc_text = to_categorical(preproc_text, num_classes=vocab)
            print(np.array(preproc_text).shape)
            return preproc_text[:,1:]
    return preproc_text

# Reload Tokenizer
def tokenizerFromJson(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.load(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

# Helper for bleu score calculation
def get_hypo_sents(en_seqs, inf_encoder, inf_decoder, de_tok, max_len=10, size=128,
                   full_result=False, GRU_LAYER=True):
    """Infer input en sequences and return hypothesis output fr sentences.
    # Arguments
        en_seqs     : English source sequences list.
        inf_encoder : Inference model encoder.
        inf_decoder : Inference model dncoder.
        de_tok      : Target vocab tokenizer.
        max_len     : Maximum sentence length.
    # Returns
        fr_sent  : Last target fr sentence.
        fr_sents : Target fr sentences list.
    """
    fr_sents = []
    for i, en_seq in enumerate(en_seqs):
        if i < size:
            en_seq = np.expand_dims(en_seq, axis=0)
            # Predict the initial decoder state with the encoder
            if GRU_LAYER:
                de_s_t = inf_encoder.predict(en_seq)
            else:
                de_s_t, de_c_t = inf_encoder.predict(en_seq)
            # de_seq = word2onehot(de_tok, '<sos>', fr_vocab)
            de_seq = getWordId(de_tok, '<sos>')
            de_seq = de_seq[0]
            fr_sent = []
            for i in range(max_len):    
                # Predict from the decoder and recursively assign the new state to de_s_t
#                 print(np.array(de_s_t).shape) #(1, 1000)
                if GRU_LAYER:
                    de_prob, de_s_t = inf_decoder.predict([de_seq, de_s_t])
                else:
                    de_prob, de_s_t, de_c_t = inf_decoder.predict([de_seq, de_s_t, de_c_t])
                if de_prob.ndim > 2:
                    de_prob = np.squeeze(de_prob, axis=(1,))
                # Get the word from the probability output using probs2word
                de_w = probs2word(de_prob, de_tok)
                if de_w is not None: # None : padding word
                    # Convert the word to a onehot sequence using word2onehot
                    de_seq = getWordId(de_tok, de_w)
                    if full_result:
                        if de_w == '<sos>': 
                            fr_sent.append(de_w) 
                            pass
                        elif de_w == '<eos>': 
                            fr_sent.append(de_w) 
                            break
                        else: fr_sent.append(de_w)  
                    else:
                        if de_w == '<sos>': pass
                        elif de_w == '<eos>': break
                        else: fr_sent.append(de_w)
                else:
                    break
            fr_sents.append(fr_sent)  
    return fr_sent, fr_sents

def get_target_sents(fr_seqs, de_tok, max_len=10, size=128):  # target without unused vocab
    """Convert target fr sequences to sentences.
    # Arguments
        fr_seqs : French target sequences list.
        de_tok  : Target vocab tokenizer.
        max_len : Maximum sentence length.
    # Returns
        fr_sent  : Last target fr sentence.
        fr_sents : Target fr sentences list.
    """
    fr_sents = []
    for i, fr_seq in enumerate(fr_seqs):
        if i < size:
            fr_sent = []
            for i in range(max_len+1):   
                wid = fr_seq[i]
                if wid > 0: 
                    de_w = de_tok.index_word[wid]
                    if de_w == '<sos>': pass
                    elif de_w == '<eos>': break
                    else: fr_sent.append(de_w)
                else: 
                    break
            fr_sents.append([fr_sent])  
    return fr_sent, fr_sents

def get_source_sents(en_seqs, en_tok, max_len=10, size=128):  # target without unused vocab
    """Convert source en sequences to sentences.
    # Arguments
        en_seqs : French source sequences list.
        en_tok  : Source vocab tokenizer.
        max_len : Maximum sentence length.
    # Returns
        en_sent  : Last source en sentence.
        en_sents : Source en sentences list.
    """
    en_sents = []
    for i, en_seq in enumerate(en_seqs):
        if i < size:
            en_sent = []
            for i in range(max_len):   
                wid = en_seq[i]
                if wid > 0: 
                    en_w = en_tok.index_word[wid]
                    en_sent.append(en_w)
                else: 
                    break
            en_sents.append([en_sent])  
    return en_sent, en_sents

def bleu_score(hypo, target):
    print(len(hypo))
    if len(hypo) <= 1:
        return corpus_bleu(target, hypo, weights=(1, 0, 0, 0)) * 100
    elif len(hypo) <= 2:
        return corpus_bleu(target, hypo, weights=(0.5, 0.5, 0, 0)) * 100
    elif len(hypo) <= 3:
        return corpus_bleu(target, hypo, weights=(0.33, 0.33, 0.33, 0)) * 100
    else:
        return corpus_bleu(target, hypo)*100.0

# Helper for generate translation
def word2onehot(tokenizer, word, vocab_size):
    de_seq = tokenizer.texts_to_sequences([[word]])
    de_onehot = to_categorical(de_seq, num_classes=vocab_size)
    de_onehot = np.expand_dims(de_onehot, axis=1)    
    return de_onehot

def getWordId(tokenizer, word):
    de_seq = tokenizer.texts_to_sequences([word])
    return de_seq

def probs2word(probs, tok):
    wid = np.argmax(probs[0,:], axis=-1)
    if wid > 0: w = tok.index_word[wid]
    else: w = None
    return w

def translation(en_sent, tar_sent, en_tok, de_tok, encoder, decoder, 
                max_len=10, full_result=False, logging=None, GRU_LAYER=True):
    en_sent = spacyTokenize(spacy_en, en_sent)
    tar_sent = spacyTokenize(spacy_fr, tar_sent)
    print('\nEnglish: {}'.format((" ".join(en_sent[0].split(" ")[:max_len])).lower()))
    en_seq = sents2seqs('source', en_tok, en_sent, max_len=max_len, onehot=False, reverse=False)
#     print(en_seq)
    # Predict the initial decoder state with the encoder
    if GRU_LAYER:
        de_s_t = encoder.predict(en_seq)
    else:
        de_s_t, de_c_t = encoder.predict(en_seq)
    # de_seq = word2onehot(de_tok, '<sos>', fr_vocab)
    de_seq = getWordId(de_tok, '<sos>')
    de_seq = de_seq[0]
    fr_sent = ''
    for i in range(max_len):    
        # Predict from the decoder and recursively assign the new state to de_s_t
        if GRU_LAYER:
            de_prob, de_s_t = decoder.predict([de_seq, de_s_t])
        else:
            de_prob, de_s_t, de_c_t = decoder.predict([de_seq, de_s_t, de_c_t])
#         print(de_prob.shape) # (1, 1, 5000)
        if de_prob.ndim > 2:
            de_prob = np.squeeze(de_prob, axis=(1,))
        # Get the word from the probability output using probs2word
        de_w = probs2word(de_prob, de_tok)
        # Convert the word to a onehot sequence using word2onehot
    #     de_seq = word2onehot(de_tok, de_w, fr_vocab)
        if de_w is not None: # None : padding word
            # Convert the word to a onehot sequence using word2onehot
            de_seq = getWordId(de_tok, de_w)
            if full_result:
                if de_w == '<sos>': 
                    fr_sent += de_w + ' '
                    pass
                elif de_w == '<eos>': 
                    fr_sent += de_w + ' ' 
                    break
                else: fr_sent += de_w + ' '
            else:
                if de_w == '<sos>': pass
                elif de_w == '<eos>': break
                else: fr_sent += de_w + ' '
        else:
            break
    print("French (hypo): {}".format(fr_sent))
    print("French (target): {}".format((" ".join(tar_sent[0].split(" ")[:max_len])).lower()))
    if logging:
        logging.info("-------------------------------------")
        logging.info('English: {}'.format((" ".join(en_sent[0].split(" ")[:max_len])).lower()))
        logging.info("French (hypo): {}".format(fr_sent))
        logging.info("French (target): {}"
                .format((" ".join(tar_sent[0].split(" ")[:max_len])).lower()))


def translateEnFr(en_sent, tar_sent, en_tok, de_tok, encoder, decoder,
                max_len=10, full_result=False, logging=None, GRU_LAYER=True):
    en_sent = spacyTokenize(spacy_en, en_sent)
    # tar_sent = spacyTokenize(spacy_fr, tar_sent)
    # print('\nEnglish: {}'.format((" ".join(en_sent[0].split(" ")[:max_len])).lower()))
    en_seq = sents2seqs('source', en_tok, en_sent, max_len=max_len, onehot=False, reverse=False)
#     print(en_seq)
    # Predict the initial decoder state with the encoder
    if GRU_LAYER:
        de_s_t = encoder.predict(en_seq)
    else:
        de_s_t, de_c_t = encoder.predict(en_seq)
    # de_seq = word2onehot(de_tok, '<sos>', fr_vocab)
    de_seq = getWordId(de_tok, '<sos>')
    de_seq = de_seq[0]
    fr_sent = ''
    for i in range(max_len):
        # Predict from the decoder and recursively assign the new state to de_s_t
        if GRU_LAYER:
            de_prob, de_s_t = decoder.predict([de_seq, de_s_t])
        else:
            de_prob, de_s_t, de_c_t = decoder.predict([de_seq, de_s_t, de_c_t])
#         print(de_prob.shape) # (1, 1, 5000)
        if de_prob.ndim > 2:
            de_prob = np.squeeze(de_prob, axis=(1,))
        # Get the word from the probability output using probs2word
        de_w = probs2word(de_prob, de_tok)
        # Convert the word to a onehot sequence using word2onehot
    #     de_seq = word2onehot(de_tok, de_w, fr_vocab)
        if de_w is not None: # None : padding word
            # Convert the word to a onehot sequence using word2onehot
            de_seq = getWordId(de_tok, de_w)
            if full_result:
                if de_w == '<sos>':
                    fr_sent += de_w + ' '
                    pass
                elif de_w == '<eos>':
                    fr_sent += de_w + ' '
                    break
                else: fr_sent += de_w + ' '
            else:
                if de_w == '<sos>': pass
                elif de_w == '<eos>': break
                else: fr_sent += de_w + ' '
        else:
            break

    return fr_sent


def get_bleu_score_from_model_result(types, en_x_, de_x_, en_tok, de_tok, 
                                     encoder, decoder, max_len=10, bsize=128, GRU_LAYER=True):
    total_bleu = 0.0
    size = len(en_x_)
    n_batch = math.ceil(size/bsize)
    ctn = 1
    for i in range(0, size, bsize):    
        en_x = en_x_[i:i+bsize]
        de_x = de_x_[i:i+bsize]
        hypo_sent, hypo = get_hypo_sents(en_x, encoder, decoder, de_tok, 
                                         max_len=max_len, GRU_LAYER=GRU_LAYER)
        target_sent, target = get_target_sents(de_x, de_tok, max_len=max_len)  
        print("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
        bleu = bleu_score(hypo, target)
        print("batch {}/{} -> {} Bleu Score: {:.2f} ({})".format(ctn, n_batch, types, 
                                                                 bleu, bleu))
        temp_bsize = len(en_x)
        total_bleu += (bleu*temp_bsize)
        ctn += 1
    score = (total_bleu/size*1.0)
    print("{} Bleu Score: {:.2f}".format(types, score))
    
# Bleu Score
def evaluate_bleu_score_trigger(en_x_, de_x_, en_tok, de_tok, encoder, decoder,
                                max_len=10, logging=None, GRU_LAYER=True):
    en_x = []
    de_x = []
    sample_size = 1
    for i in random.sample(range(len(en_x_)), sample_size):    
        en_x.append(en_x_[i])
        de_x.append(de_x_[i])
    hypo_sent, hypo = get_hypo_sents(en_x, encoder, decoder, de_tok,
                                     max_len=max_len, size=sample_size, GRU_LAYER=GRU_LAYER)
    target_sent, target = get_target_sents(de_x, de_tok, max_len=max_len,
                                           size=sample_size)  
    bleu = bleu_score(hypo, target)
    print("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
    print("Trigger Bleu Score: {:.2f}".format(bleu))
    if logging:
        logging.info("-------------------------------------")
        logging.info("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
        logging.info("Validation Bleu Score: {:.2f}".format(bleu))

def evaluate_bleu_score(en_x_, de_x_, en_tok, de_tok, encoder, decoder,
                                max_len=10, size=2, logging=None, GRU_LAYER=True):
    en_x = []
    de_x = []
    sample_size = 5000 if size < 2 else 1000
    print("Sample Size:", sample_size)
    for i in random.sample(range(len(en_x_)), sample_size):    
        en_x.append(en_x_[i])
        de_x.append(de_x_[i])
    hypo_sent, hypo = get_hypo_sents(en_x, encoder, decoder, de_tok,
                                     max_len=max_len, size=sample_size, GRU_LAYER=GRU_LAYER)
    target_sent, target = get_target_sents(de_x, de_tok, max_len=max_len,
                                           size=sample_size)  
    for i in range(4):
        print("\nhypo: {} \ntarget: {}".format(hypo[i], target[i][0]))
    print("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
    bleu = bleu_score(hypo, target)
    print("Validation Bleu Score: {:.2f}".format(bleu))
    if logging:
        logging.info("-------------------------------------")
        for i in range(4):
            logging.info("\nhypo: {} \ntarget: {}".format(hypo[i], target[i][0]))
        logging.info("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
        logging.info("Validation Bleu Score: {:.2f}".format(bleu))
        logging.info("-------------------------------------")

import logging
import datetime
import os
import pytz 
def get_logger(output_folder, EPOCHS, LEARNING_RATE):
    now = datetime.datetime.now(pytz.timezone("Asia/Kuala_Lumpur"))
    log_filename = output_folder + now.strftime("%Y-%m-%d_%H-%M-%S") \
                        + "_e{}_lr{}".format(EPOCHS, LEARNING_RATE) \
                        + '.log'  
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=log_filename, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    # logging.error # debug # info # warning
    return logging

def log_train_config(logging, mode, size, trigger, EPOCHS, BATCH_SIZE, MAX_LENGTH,
                    LEARNING_RATE, MAX_VOCAB, TRAIN_TEST, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM,
                    preprocess_folder, train_output_folder, model_var, model_type, rnn_layer):
    logging.info("mode = {} # 0 - baseline, 1 - blackbox".format(mode))
    logging.info("dataset size = {} # 0 - 1000k, 1 - all, 2 - 5k, 3 - 50k".format(size))
    logging.info("trigger set = {} # 1 - 1w, 2 - 4w, 3 - 8w".format(trigger))
    logging.info("model_type = {} # 1 hidden dense, 2 hidden dense".format(model_type))
    logging.info("model_rnn = {}".format(rnn_layer))
    logging.info("EPOCHS = {}".format(EPOCHS))
    logging.info("BATCH_SIZE = {}".format(BATCH_SIZE))
    logging.info("MAX_LENGTH = {}".format(MAX_LENGTH))
    logging.info("LEARNING_RATE = {}".format(str(LEARNING_RATE)))
    logging.info("MAX_VOCAB = {}".format(MAX_VOCAB))
    logging.info("TRAIN_TEST = {}".format(TRAIN_TEST))
    logging.info("ENC_EMB_DIM = {}".format(ENC_EMB_DIM))
    logging.info("DEC_EMB_DIM = {}".format(DEC_EMB_DIM))
    logging.info("HID_DIM = {}".format(HID_DIM))
    logging.info("preprocess_folder = {} # get vocab and tokenizer".format(preprocess_folder))
    logging.info("train_output_folder = {} # save trained model".format(train_output_folder))
    logging.info("model_var = {}".format(model_var))
    logging.info("=================================================")
    
def log_attack_config(logging, size, finetune_size, mode, trigger, EPOCHS, 
                      BATCH_SIZE, MAX_LENGTH,
                    LEARNING_RATE, MAX_VOCAB, TRAIN_TEST, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM,
                    preprocess_folder, train_output_folder, preprocess_2_folder, 
                    attack_output_folder, model_var, attacked_model_var, 
                    model_type, rnn_layer, ambiguity=None, attack_type="finetune", is_trainable=True, CONSTANT=None):
    logging.info("attacking type = {}".format(attack_type))
    if attack_type == "transfer_learning":
        logging.info("is_trainable = {} # True - finetune all, False - train last layer"
                     .format(attack_type))
    logging.info("dataset size = {} # 0 - 1000k, 1 - all, 2 - 5k, 3 - 50k".format(size))
    logging.info("finatune_size = {} # 1 - all testset, 2 - newstest08-14".format(finetune_size))
    logging.info("mode = {} # 0 - baseline, 1 - blackbox".format(mode))
    logging.info("trigger set = {} # 1 - 1w, 2 - 4w, 3 - 8w".format(trigger))
    if ambiguity:
        logging.info("ambiguity = {} # set A, set B".format(ambiguity))
    logging.info("model_type = {} # 1 hidden dense, 2 hidden dense".format(model_type))
    logging.info("model_rnn = {}".format(rnn_layer))
    logging.info("EPOCHS = {}".format(EPOCHS))
    logging.info("BATCH_SIZE = {}".format(BATCH_SIZE))
    logging.info("MAX_LENGTH = {}".format(MAX_LENGTH))
    logging.info("LEARNING_RATE = {}".format(str(LEARNING_RATE)))
    logging.info("MAX_VOCAB = {}".format(MAX_VOCAB))
    logging.info("TRAIN_TEST = {}".format(TRAIN_TEST))
    logging.info("ENC_EMB_DIM = {}".format(ENC_EMB_DIM))
    logging.info("DEC_EMB_DIM = {}".format(DEC_EMB_DIM))
    logging.info("HID_DIM = {}".format(HID_DIM))
    logging.info("preprocess_folder = {} # get vocab and tokenizer".format(preprocess_folder))
    logging.info("preprocess_2_folder = {} # get preprocess data".format(preprocess_2_folder))
    logging.info("train_output_folder = {} # save trained model".format(train_output_folder))
    logging.info("attack_output_folder = {} # get preprocess data".format(attack_output_folder))
    logging.info("model_var = {}".format(model_var))
    logging.info("attacked_model_var = {}".format(attacked_model_var))
    logging.info("CONSTANT = {}".format(CONSTANT))
    logging.info("====================================================")
 
def save_model_summary(output_folder, model, model_type, rnn_type, filename="train"):
    json_str = model.to_json()
    model_ = "2_dense_" if model_type is 1 else "3_dense_"
    output_folder = output_folder + model_ + rnn_type + "_model_summary/"
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder+"s-"+filename+"_model.json", "w") as f:
        json.dump(json.loads(json_str), f, indent=4)
    with open(output_folder+"s-"+filename+"_model.txt",'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))