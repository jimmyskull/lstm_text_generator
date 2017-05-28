# -*- coding: utf-8 -*-
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

CHARS = [' ', ',', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
         'z', 'à', 'á', 'â', 'ã', 'å', 'æ', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ',
         'ú']
CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))

def load_model(weights_filename, shape=(40, 43)):
    model = Sequential()
    model.add(LSTM(512, input_shape=shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(shape[1], activation='softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.load_weights(weights_filename)
    return model

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, start_sentence="", length=100, diversity=0.2,
                  maxlen=40, verbose=False):
    assert length > 0
    assert 0. < diversity and diversity <= 1.
    assert isinstance(start_sentence, str)
    def log(msg):
        if verbose:
            sys.stdout.write(msg)
            sys.stdout.flush()
    sentence = start_sentence[-maxlen:]
    generated = sentence
    log(generated)
    for i in range(length):
        x = np.zeros((1, maxlen, len(CHARS)))
        for t, char in enumerate(sentence):
            x[0, t, CHAR_INDICES[char]] = 1.0
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = INDICES_CHAR[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        log(next_char)
    log("\n")
    return generated

model = load_model('p2-weights-improvement-00-0.2655-bigger.hdf5')
generate_text(model, start_sentence="eu adoro programar em uma linguagem chamada ", verbose=True)
