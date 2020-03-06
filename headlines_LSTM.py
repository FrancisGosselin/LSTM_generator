'''
#Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import pandas as pd
import math

file = "/home/genderbender/Documents/TDP080/ngram/abcnews-date-text.csv"


class LSTM_Model():

    maxlen = 25
    step = 1
    batch_size = 500

    def __buildbatch__(self, filename):

        for chunk in pd.read_csv(filename, chunksize=self.batch_size):
            # cut the text in semi-redundant sequences of maxlen characters
            text = "\n".join(chunk['headline_text'].to_numpy())

            sentences = []
            next_chars = []
            
            i = 0
            while i < len(text) - self.maxlen:
                sentences.append(text[i: i + self.maxlen])
                next_chars.append(text[i + self.maxlen])
                
                if next_chars[len(sentences) - 1] == '\n':
                    i = i + self.maxlen + 1
                else:
                    i += self.step
            

            x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
            y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars[i]]] = 1

            yield (x, y)


    def __buildmodel__(self, filename):
        for chunk in pd.read_csv(filename, chunksize=10000):
            text = "\n".join(chunk['headline_text'].to_numpy())
            self.chars = sorted(list(set(text)))
            break
        
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # build the model: a single LSTM
        print('Build model...')
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        self.model.add(Dense(len(self.chars), activation='softmax'))

        optimizer = RMSprop(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def train(self, filename):
        self.filename = filename
        with open(filename) as f:
            self.data_size = sum(1 for line in f) / 10

        self.__buildmodel__(filename)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)

        self.model.fit_generator(self.__buildbatch__(filename),
                steps_per_epoch=math.ceil(self.data_size/self.batch_size),
                epochs=60,
                callbacks=[print_callback])

        print("Training Done!")
        print("")

        self.model.save('headline_model.h5')

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def on_epoch_end(self, epoch, _):
        
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        sentences = []
        for chunk in pd.read_csv(self.filename, chunksize=10000):
            sentences = random.choices(chunk['headline_text'].to_numpy(), k=5)
            sentences = [sentence[0:self.maxlen] for sentence in sentences]
            break


        for sentence in sentences:
            next_char = ''
            while next_char != '\n' and len(sentence) < 40:

                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence[-self.maxlen:]):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds)
                next_char = self.indices_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

generator = LSTM_Model()
generator.train(file)