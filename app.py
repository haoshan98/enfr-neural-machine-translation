import os
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

from module.helper import spacy_fr, spacy_en, spacyTokenize, \
                    addSpecialToken, sents2seqs, tokenizerFromJson, \
                    get_hypo_sents, get_target_sents, bleu_score, \
                    word2onehot, getWordId, probs2word, translation, \
                    get_logger, log_train_config, save_model_summary, \
                    evaluate_bleu_score_trigger, evaluate_bleu_score, translateEnFr
from module.model import nmtModel, defineInferModelAndLoadWeights
from module.model0 import nmtModel2, defineInferModel2AndLoadWeights

from generate_data import word_to_array, beautifyOutput
from flask import Flask, request, render_template
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

mode = 1 # 0 - baseline, 1 - blackbox
size = 1 # 0 - 1000k, 1 - all, 2 - 5k, 3 - 50k
trigger = 2 # 1 - 1w, 2 - 4w, 3 - 8w
model_type = 1  # 1 hidden dense, 2 hidden dense
EPOCHS = 10 # must set to record
BATCH_SIZE = 128
MAX_LENGTH = 10
LEARNING_RATE = 0.0001 if size < 2 else 0.0003
MAX_VOCAB = 15000 if size < 2 else 5000
TRAIN_TEST = 0.2
ENC_EMB_DIM = 500 if size < 2 else 50
DEC_EMB_DIM = 500 if size < 2 else 50
HID_DIM = 1000 if size < 2 else 100
GRU_LAYER = True # True - GRU, False - LSTM

preprocess_folder = 'ipr-nmt-preprocess-selected/'
train_output_folder = 'ipr-nmt-train-selected/' if mode is 1 else 'ipr-nmt-train-selected_base/'

preprocess_folder = str(BATCH_SIZE) + '/' + preprocess_folder
model_ = "2_dense" if model_type is 1 else "3_dense"
rnn_layer = "gru" if GRU_LAYER else "lstm"
train_output_folder = train_output_folder + rnn_layer + '/' + str(trigger) + '/' + model_ + '/'
list_output_folder = train_output_folder + 'bleu-inferred-result/'  # save inferred result
os.makedirs(list_output_folder, exist_ok=True)
train_output_folder = "model/"
print(preprocess_folder)
print(train_output_folder)
print(list_output_folder)

logging = get_logger(list_output_folder, EPOCHS, "")

model_var = 'seq2seq.h5'

path_triggerset = "dataset/trigger"

# Tokenizer
with open(os.path.join(preprocess_folder, 'input_tokenizer.json'), 'r') as f:
    en_tok = tokenizerFromJson(f)
with open(os.path.join(preprocess_folder, 'output_tokenizer.json'), 'r') as f:
    de_tok = tokenizerFromJson(f)

# Reload model weight
en_len = MAX_LENGTH
fr_len = MAX_LENGTH+2
if model_type is 1:
    nmt_emb, en_inputs, tr_en_emb_layer, tr_en_gru, en_state, \
        de_inputs, tr_de_emb_layer, tr_de_gru, tr_de_dense_0, tr_de_dense_1 = \
                nmtModel(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM,
                         HID_DIM, GRU_LAYER=GRU_LAYER)
    tr_de_dense_2 = None
else:
    nmt_emb, en_inputs, tr_en_emb_layer, tr_en_gru, en_state, \
        de_inputs, tr_de_emb_layer, tr_de_gru, tr_de_dense_0, tr_de_dense_1, tr_de_dense_2 = \
                nmtModel2(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM, DEC_EMB_DIM,
                          HID_DIM, GRU_LAYER=GRU_LAYER)

op = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE, epsilon=1e-07)
nmt_emb.compile(optimizer=op, loss=CategoricalCrossentropy(from_logits=False), metrics=['acc'])
print(nmt_emb.summary())

nmt_emb.load_weights(train_output_folder + model_var)

# Inference model
if model_type is 1:
    en_emb_layer, en_gru, encoder, de_emb_layer, de_gru, \
    de_dense_0, de_dense_1, decoder = \
        defineInferModelAndLoadWeights(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM,
                                       DEC_EMB_DIM, HID_DIM, tr_en_emb_layer, tr_en_gru,
                                       tr_de_emb_layer, tr_de_gru,
                                       tr_de_dense_0, tr_de_dense_1, GRU_LAYER=GRU_LAYER)
else:
    en_emb_layer, en_gru, encoder, de_emb_layer, de_gru, \
    de_dense_0, de_dense_1, de_dense_2, decoder = \
        defineInferModel2AndLoadWeights(en_len, fr_len, MAX_VOCAB, ENC_EMB_DIM,
                                       DEC_EMB_DIM, HID_DIM, tr_en_emb_layer, tr_en_gru,
                                       tr_de_emb_layer, tr_de_gru,
                                       tr_de_dense_0, tr_de_dense_1, tr_de_dense_2,
                                        GRU_LAYER=GRU_LAYER)
print(decoder.summary())


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/translation', methods=['GET', 'POST'])
# @app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        word = request.form.get('word')
        target = request.form.get('target')
        model_name = request.form.get('model')

        french_hypo_raw = translateEnFr([word], [""], en_tok, de_tok,
                                    encoder, decoder, max_len=MAX_LENGTH,
                                    GRU_LAYER=GRU_LAYER)
        french_hypo = beautifyOutput(word, french_hypo_raw)

        if target is not "":
            print("target: ", target)
            print(french_hypo_raw[:-1].split(" "))
            print([target.split(" ")])
            bleu = bleu_score([french_hypo_raw[:-1].split(" ")], [[target.split(" ")]])
            return render_template('index.html',
                                   source_text='English:\n {}'.format(word),
                                   prediction_text='French:\n {}'.format(french_hypo),
                                   bleu_text='Bleu Score: {:.2f}%'.format(bleu),
                                   target_text='Target: {}'.format(target))
        else:
            return render_template('index.html',
                                   source_text='English: {}'.format(word),
                                   prediction_text='French: {}'.format(french_hypo))

    return render_template('index.html')


if __name__ == '__main__':
    print('Done loading...')
    app.run(debug=False)
