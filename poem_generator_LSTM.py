#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import random
import sys
import io
import re


#text = open('poems.txt').read().lower()
#print('corpus length:', len(text))

text = pd.read_csv("p.txt", header=None, engine="python", delimiter="\r\t", encoding="utf-8")
text = text[0].values

clean_corpus = []
for i in range(0, len(text)):
    seq = re.sub('[^a-zA-Z]', ' ', text[i])
    #seq = re.sub('\W', ' ', text[i]) #\W corresponds to the set [^a-zA-Z0-9_]
    seq = seq.lower()
    clean_corpus.append(seq)

clean_corpus = str(clean_corpus)
clean_corpus = clean_corpus.replace("'", '')
clean_corpus = clean_corpus.replace("  ", ' ')
clean_corpus = clean_corpus.replace(" ,", ',\n')
clean_corpus = str(clean_corpus)


chars = sorted(list(set(clean_corpus)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(clean_corpus) - maxlen, step):
    sentences.append(clean_corpus[i: i + maxlen])
    next_chars.append(clean_corpus[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(clean_corpus) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = clean_corpus[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=120,
          callbacks=[print_callback])