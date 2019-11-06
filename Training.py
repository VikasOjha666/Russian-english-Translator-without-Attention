#Neural Machine Translation without attention mechanism.
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector,Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import string 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




#Read all the text.
f=open('data.txt','rt',encoding='utf-8')
allsents=f.read()
f.close()
#Seperate the sents and form language pair.
sents=allsents.strip().split('\n')
pair=[pr.split('\t') for pr in sents]
pair=np.array(pair)



#Making a dataframe so to drop third column from array.
tempdf=pd.DataFrame(pair)

lang_pairs=tempdf.iloc[:,:2].values


#Removing punctuation.
lang_pairs[:,0] = [(s.translate(str.maketrans('', '', string.punctuation))).lower() for s in lang_pairs[:,0]]
lang_pairs[:,1] = [(s.translate(str.maketrans('', '', string.punctuation))).lower() for s in lang_pairs[:,1]]


# function to build a tokenizer
def tokenization(lines):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(lines)
      return tokenizer


# prepare english tokenizer
eng_tokenizer = tokenization(lang_pairs[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)


# prepare Russian tokenizer
rus_tokenizer = tokenization(lang_pairs[:, 1])
rus_vocab_size = len(rus_tokenizer.word_index) + 1

rus_length = 8
print('Russian Vocabulary Size: %d' % rus_vocab_size)


#Encodes and pads the sentences.
def encode_sequences(tokenizer, length, lines):
         seq = tokenizer.texts_to_sequences(lines)
         seq = pad_sequences(seq, maxlen=length, padding='post')
         return seq


from sklearn.model_selection import train_test_split

train, test = train_test_split(lang_pairs, test_size=0.2, random_state = 12)


# prepare training data
trainX = encode_sequences(rus_tokenizer, rus_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(rus_tokenizer, rus_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])


#Building the model
def define_model(src_vocab,tar_vocab, src_timesteps,tar_timesteps,units):
      model = Sequential()
      model.add(Embedding(src_vocab, units, input_length=tar_timesteps, mask_zero=True))
      model.add(LSTM(units))
      model.add(Dropout(0.50))
      model.add(RepeatVector(tar_timesteps))
      model.add(LSTM(units, return_sequences=True))
      model.add(Dropout(0.50))
      model.add(Dense(tar_vocab, activation='softmax'))
      return model


model = define_model(rus_vocab_size, eng_vocab_size, rus_length, eng_length, 512)

rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=30, batch_size=512, validation_split = 0.2,callbacks=[checkpoint], 
                    verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()



