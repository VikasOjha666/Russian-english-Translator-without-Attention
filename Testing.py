import pandas as pd
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, RepeatVector,Dropout
from keras.models import load_model
from keras import optimizers
import string 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Read all the text.
f=open('rus.txt','rt',encoding='utf-8')
allsents=f.read()
f.close()
#Seperate the sents and form language pair.
sents=allsents.strip().split('\n')
pair=[pr.split('\t') for pr in sents]
pair=np.array(pair)

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



# prepare Deutch tokenizer
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


model=load_model('NMT_model.h5')



def get_word(n, tokenizer):
      for word, index in tokenizer.word_index.items():
          if index == n:
              return word
      return None

preds_text = []

for i in preds:
       temp = []
       for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                     temp.append('')
                else:
                     temp.append(t)
            else:
                   if(t == None):
                          temp.append('')
                   else:
                          temp.append(t) 

       preds_text.append(' '.join(temp))





#View random samples.
import numpy as np
random_idx=np.random.randint(0,len(test)-1,15)




for i in random_idx:
  print("Actual=",test[i][0],end=" ")
  print("Predicted=",preds_text[i],end="")
  print()

