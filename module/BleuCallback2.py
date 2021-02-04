import tensorflow as tf
import numpy as np
import random
import math
import tensorflow.keras as keras
from module.model import bleuInferenceModel
from module.model2 import bleuInferenceModel2
from module.helper import get_source_sents, get_hypo_sents, get_target_sents, bleu_score, \
                          save_model_summary

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class BleuCallbackAmbiguity(keras.callbacks.Callback):
    def __init__(self, v_en_input_sequences, v_de_input_sequences, 
                 en_trigger_sequences, de_trigger_input_sequences, # for checking withstand rate
                 en_ambiguity_sequences, de_ambiguity_input_sequences, 
                 bsize, en_tok, de_tok, logging, 
                 en_inputs, en_state, 
                 tr_en_emb_layer, tr_en_gru, 
                 tr_de_emb_layer, tr_de_gru, 
                 tr_de_dense_0, tr_de_dense_1, 
                 en_len, fr_len, MAX_VOCAB,
                 ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, MAX_LENGTH, 
                 DATA_SIZE, n_epochs, 
                 output_folder, getFast=True, tr_de_dense_2=None, GRU_LAYER=True):
        'Initialization'
        self.v_en_x_ = v_en_input_sequences
        self.v_de_x_ = v_de_input_sequences

        self.bsize = bsize
        self.en_tok = en_tok
        self.de_tok = de_tok
        self.de_index2word = de_tok.index_word
        self.t_en_x = en_trigger_sequences
        self.t_de_x = de_trigger_input_sequences
        self.a_en_x = en_ambiguity_sequences
        self.a_de_x = de_ambiguity_input_sequences
        
        self.logging = logging
        self.en_len = en_len
        self.fr_len = fr_len
        self.MAX_VOCAB = MAX_VOCAB
        self.en_inputs = en_inputs
        self.en_state = en_state
        self.tr_en_emb_layer = tr_en_emb_layer
        self.tr_en_gru = tr_en_gru
        self.tr_de_emb_layer = tr_de_emb_layer
        self.tr_de_gru = tr_de_gru
        self.tr_de_dense_0 = tr_de_dense_0
        self.tr_de_dense_1 = tr_de_dense_1
        self.tr_de_dense_2 = tr_de_dense_2
        self.ENC_EMB_DIM = ENC_EMB_DIM
        self.DEC_EMB_DIM = DEC_EMB_DIM
        self.HID_DIM = HID_DIM
        self.MAX_LENGTH = MAX_LENGTH
        self.DATA_SIZE = DATA_SIZE
        self.n_epochs = n_epochs
        self.output_folder = output_folder
        self.getFast = getFast
        self.GRU_LAYER = GRU_LAYER
        
    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        MAX_LENGTH = self.MAX_LENGTH
        
        if epoch <= 1 or (self.DATA_SIZE<2 and epoch % 5 is 0) \
                or (self.DATA_SIZE>=2 and epoch % 20 is 0) \
                or (epoch is self.n_epochs):
            # Inference (for cal bleu score)
            rnn_layer = "gru" if self.GRU_LAYER else "lstm"
            if self.tr_de_dense_2:
                model_type = 2
#                 inf_encoder, inf_decoder = \
#                 bleuInferenceModel2(self.en_inputs, self.en_state, self.tr_de_emb_layer, 
#                                    self.tr_de_gru, self.tr_de_dense_0, self.tr_de_dense_1,
#                                    self.tr_de_dense_2, self.HID_DIM)
                inf_encoder, inf_decoder = \
                    bleuInferenceModel2(self.en_len, self.fr_len, 
                                        self.MAX_VOCAB, self.ENC_EMB_DIM, 
                                           self.DEC_EMB_DIM, self.HID_DIM, 
                                           self.tr_en_emb_layer, self.tr_en_gru, 
                                           self.tr_de_emb_layer, self.tr_de_gru, 
                                           self.tr_de_dense_0, self.tr_de_dense_1, 
                                           self.tr_de_dense_2, GRU_LAYER=self.GRU_LAYER)
            else:
                model_type = 1
                inf_encoder, inf_decoder = \
                bleuInferenceModel(self.en_len, self.fr_len, 
                                        self.MAX_VOCAB, self.ENC_EMB_DIM, 
                                           self.DEC_EMB_DIM, self.HID_DIM, 
                                           self.tr_en_emb_layer, self.tr_en_gru, 
                                           self.tr_de_emb_layer, self.tr_de_gru, 
                                           self.tr_de_dense_0, self.tr_de_dense_1, 
                                           GRU_LAYER=self.GRU_LAYER)
            if epoch <=1:
                print("")
                print(inf_decoder.summary())
                save_model_summary(self.output_folder, inf_decoder, model_type,
                                   rnn_layer, "infer_decoder")
                
            if self.getFast:
                valid_score = self.cal_bleu_score_fast("Test Set", epoch, self.v_en_x_, self.v_de_x_,
                                              inf_encoder, inf_decoder)
            else:
                valid_score = self.cal_bleu_score("Test Set", epoch, self.v_en_x_, self.v_de_x_,
                                              inf_encoder, inf_decoder)
            # bleu for trigger set
            source_sent, source_sents = get_source_sents(self.t_en_x, self.en_tok,
                                                         max_len=MAX_LENGTH,
                                                         size=self.bsize) 
            hypo_sent, hypo = get_hypo_sents(self.t_en_x, inf_encoder, inf_decoder, self.de_tok,
                                             max_len=MAX_LENGTH, GRU_LAYER=self.GRU_LAYER)
            target_sent, target = get_target_sents(self.t_de_x, self.de_tok, max_len=MAX_LENGTH)  
            self.logging.info("\ntrigger source: {} \ntrigger hypo: {} \ntrigger target: {}"
                  .format(source_sent, hypo_sent, target_sent))
            trigger_bleu = bleu_score(hypo, target)
            self.logging.info("Epoch {} -> Trigger Bleu: {:.2f} ({})".format(epoch, trigger_bleu,
                                                                             trigger_bleu))
            # bleu for ambiguity set
            a_source_sent, a_source_sents = get_source_sents(self.a_en_x, self.en_tok,
                                                             max_len=MAX_LENGTH, size=self.bsize) 
            a_hypo_sent, a_hypo = get_hypo_sents(self.a_en_x, inf_encoder, inf_decoder, self.de_tok,
                                                 max_len=MAX_LENGTH, GRU_LAYER=self.GRU_LAYER)
            a_target_sent, a_target = get_target_sents(self.a_de_x, self.de_tok, max_len=MAX_LENGTH)  
            self.logging.info("\nambiguity source: {} \nambiguity hypo: {} \nambiguity target: {}"
                              .format(a_source_sent, a_hypo_sent, a_target_sent))
            ambiguity_bleu = bleu_score(a_hypo, a_target)
            self.logging.info("Epoch {} -> Ambiguity Bleu: {:.2f} ({})".format(epoch,
                                                                               ambiguity_bleu,
                                                                               ambiguity_bleu))

    def cal_bleu_score_fast(self, types, epoch, en_x_, de_x_, encoder, decoder):
        en_x = []
        de_x = []
        MAX_LENGTH = self.MAX_LENGTH
        sample_size = 3000 if self.DATA_SIZE<2 else 1000
        for i in random.sample(range(len(en_x_)), sample_size):    
            en_x.append(en_x_[i])
            de_x.append(de_x_[i])
        hypo_sent, hypo = get_hypo_sents(en_x, encoder, decoder, self.de_tok,
                                         max_len=MAX_LENGTH, size=sample_size,
                                         GRU_LAYER=self.GRU_LAYER)
        target_sent, target = get_target_sents(de_x, self.de_tok, max_len=MAX_LENGTH,
                                               size=sample_size)  
        for i in range(4):
            self.logging.info("\nhypo: {} \ntarget: {}".format(hypo[i], target[i][0]))
        self.logging.info("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
        bleu = bleu_score(hypo, target)
        self.logging.info("Epoch {} -> {} Bleu Score: {:.2f}".format(epoch, types, bleu))
        return bleu
    
    def cal_bleu_score(self, types, epoch, en_x_, de_x_, encoder, decoder):
        total_bleu = 0.0
        size = len(en_x_)
        bsize = self.bsize
        MAX_LENGTH = self.MAX_LENGTH
        n_batch = math.ceil(size/bsize)
        ctn = 1
        for i in range(0, size, bsize):    
            en_x = en_x_[i:i+bsize]
            de_x = de_x_[i:i+bsize]
            hypo_sent, hypo = get_hypo_sents(en_x, encoder, decoder, self.de_tok,
                                             max_len=MAX_LENGTH, size=bsize, GRU_LAYER=self.GRU_LAYER)
            target_sent, target = get_target_sents(de_x, self.de_tok, max_len=MAX_LENGTH,
                                                   size=bsize)  
            self.logging.info("\nhypo: {} \ntarget: {}".format(hypo_sent, target_sent))
            bleu = bleu_score(hypo, target)
            self.logging.info("{} batch {}/{} -> {} Bleu Score: {:.2f} ({})".format(epoch, ctn,
                                                                                    n_batch, types,
                                                                                    bleu, bleu))
            temp_bsize = len(en_x)
            total_bleu += (bleu*temp_bsize)
            ctn += 1
        score = (total_bleu/size*1.0)
        self.logging.info("Epoch {} -> {} Bleu Score: {:.2f}".format(epoch, types, score))
        return score