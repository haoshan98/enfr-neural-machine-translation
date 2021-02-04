import pickle
import json

def saveSequences(output_folder, en_input_sequences, de_input_sequences, de_output_sequences):
    with open(output_folder+'en_input_sequences.pickle', 'wb') as handle:
        pickle.dump(en_input_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'de_input_sequences.pickle', 'wb') as handle:
        pickle.dump(de_input_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'de_output_sequences.pickle', 'wb') as handle:
        pickle.dump(de_output_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def saveVSequences(output_folder, v_en_input_sequences, v_de_input_sequences, v_de_output_sequences):
    with open(output_folder+'v_en_input_sequences.pickle', 'wb') as handle:
        pickle.dump(v_en_input_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'v_de_input_sequences.pickle', 'wb') as handle:
        pickle.dump(v_de_input_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'v_de_output_sequences.pickle', 'wb') as handle:
        pickle.dump(v_de_output_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def saveTriggerset(output_folder, triggerset_in, triggerset_out):
    with open(output_folder+'triggerset_in.pickle', 'wb') as handle:
        pickle.dump(triggerset_in, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'triggerset_out.pickle', 'wb') as handle:
        pickle.dump(triggerset_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def saveTriggerSequences(output_folder, en_trigger_sequences, de_trigger_input_sequences, de_trigger_output_sequences):
    with open(output_folder+'en_trigger_sequences.pickle', 'wb') as handle:
        pickle.dump(en_trigger_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'de_trigger_input_sequences.pickle', 'wb') as handle:
        pickle.dump(de_trigger_input_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'de_trigger_output_sequences.pickle', 'wb') as handle:
        pickle.dump(de_trigger_output_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def saveAmbituityset(output_folder, ambiguityset_in, ambiguityset_out):
    with open(output_folder+'ambiguityset_in.pickle', 'wb') as handle:
        pickle.dump(ambiguityset_in, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'ambiguityset_out.pickle', 'wb') as handle:
        pickle.dump(ambiguityset_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def saveAmbituitySequences(output_folder, en_ambiguity_sequences, de_ambiguity_input_sequences, de_ambiguity_output_sequences):
    with open(output_folder+'en_ambiguity_sequences.pickle', 'wb') as handle:
        pickle.dump(en_ambiguity_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'de_ambiguity_input_sequences.pickle', 'wb') as handle:
        pickle.dump(de_ambiguity_input_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_folder+'de_ambiguity_output_sequences.pickle', 'wb') as handle:
        pickle.dump(de_ambiguity_output_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def saveInferredResult(list_output_folder, hypos, targets):
    with open(list_output_folder+'hypo.pickle', 'wb') as handle:
        pickle.dump(hypos, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(list_output_folder+'target.pickle', 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadInferredResult(list_output_folder):
    with open(list_output_folder+'hypo.pickle', 'rb') as handle:
        hypo = pickle.load(handle)
    with open(list_output_folder+'target.pickle', 'rb') as handle:
        target = pickle.load(handle)
    return hypo, target
        
def loadSequences(preprocess_folder):
    with open(preprocess_folder+'en_input_sequences.pickle', 'rb') as handle:
        en_input_sequences = pickle.load(handle)
    with open(preprocess_folder+'de_input_sequences.pickle', 'rb') as handle:
        de_input_sequences = pickle.load(handle)
    with open(preprocess_folder+'de_output_sequences.pickle', 'rb') as handle:
        de_output_sequences = pickle.load(handle)
    return en_input_sequences, de_input_sequences, de_output_sequences

def loadVSequences(preprocess_folder):
    with open(preprocess_folder+'v_en_input_sequences.pickle', 'rb') as handle:
        v_en_input_sequences = pickle.load(handle)
    with open(preprocess_folder+'v_de_input_sequences.pickle', 'rb') as handle:
        v_de_input_sequences = pickle.load(handle)
    with open(preprocess_folder+'v_de_output_sequences.pickle', 'rb') as handle:
        v_de_output_sequences = pickle.load(handle)
    return v_en_input_sequences, v_de_input_sequences, v_de_output_sequences

def loadTriggerset(preprocess_folder, index):
    with open(preprocess_folder+'triggerset_in.pickle', 'rb') as handle:
        triggerset_in = pickle.load(handle)
    with open(preprocess_folder+'triggerset_out.pickle', 'rb') as handle:
        triggerset_out = pickle.load(handle)
    return [triggerset_in[index-1]], [triggerset_out[index-1]]

def loadTriggerSequences(preprocess_folder, index):
    with open(preprocess_folder+'en_trigger_sequences.pickle', 'rb') as handle:
        en_trigger_sequences = pickle.load(handle)
    with open(preprocess_folder+'de_trigger_input_sequences.pickle', 'rb') as handle:
        de_trigger_input_sequences = pickle.load(handle)
    with open(preprocess_folder+'de_trigger_output_sequences.pickle', 'rb') as handle:
        de_trigger_output_sequences = pickle.load(handle)
    return [en_trigger_sequences[index-1]], [de_trigger_input_sequences[index-1]], \
           [de_trigger_output_sequences[index-1]]

def loadAmbituityset(preprocess_folder, index):
    with open(preprocess_folder+'ambiguityset_in.pickle', 'rb') as handle:
        ambiguityset_in = pickle.load(handle)
    with open(preprocess_folder+'ambiguityset_out.pickle', 'rb') as handle:
        ambiguityset_out = pickle.load(handle)
    return [ambiguityset_in[index-1]], [ambiguityset_out[index-1]]

def loadAmbituitySequences(preprocess_folder, index):
    with open(preprocess_folder+'en_ambiguity_sequences.pickle', 'rb') as handle:
        en_ambiguity_sequences = pickle.load(handle)
    with open(preprocess_folder+'de_ambiguity_input_sequences.pickle', 'rb') as handle:
        de_ambiguity_input_sequences = pickle.load(handle)
    with open(preprocess_folder+'de_ambiguity_output_sequences.pickle', 'rb') as handle:
        de_ambiguity_output_sequences = pickle.load(handle)
    return [en_ambiguity_sequences[index-1]], [de_ambiguity_input_sequences[index-1]], \
           [de_ambiguity_output_sequences[index-1]]