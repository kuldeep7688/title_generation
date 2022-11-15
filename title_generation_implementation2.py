import pandas as pd
import string
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import tensorflow.keras.utils as ku
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
tf.random.set_seed(2)
from numpy.random import seed
seed(1)

#load data files 
df1 = pd.read_csv('USvideos.csv')
df2 = pd.read_csv('CAvideos.csv')
df3 = pd.read_csv('GBvideos.csv')

#load corresponding data files with category names
data1 = json.load(open('US_category_id.json'))
data2 = json.load(open('CA_category_id.json'))
data3 = json.load(open('GB_category_id.json'))

#data cleaning
##create dataframe with necessary data
def category_extractor(data):
    i_d = [data['items'][i]['id'] for i in range(len(data['items']))]
    title = [data['items'][i]['snippet']["title"] for i in range(len(data['items']))]
    i_d = list(map(int, i_d))
    category = zip(i_d, title)
    category = dict(category)
    return category
##create a new category column by mapping the category names to their id
df1['category_title'] = df1['category_id'].map(category_extractor(data1))
df2['category_title'] = df2['category_id'].map(category_extractor(data2))
df3['category_title'] = df3['category_id'].map(category_extractor(data3))
##join the dataframes
df = pd.concat([df1, df2, df3], ignore_index=True)
##drop duplications
df = df.drop_duplicates('video_id')
##collect titles of entertainment category
entertainment = df[df['category_title'] == 'Entertainment']['title']
entertainment = entertainment.tolist()

#data-preprocessing
##remove punctuations
##convert to lowercase
def clean_text(text):
    text = ''.join(e for e in text if e not in string.punctuation).lower()
    
    text = text.encode('utf8').decode('ascii', 'ignore')
    return text
corpus = [clean_text(e) for e in entertainment]
##tokenization, generate n-gram sequences
tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
  ###get tokens
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
 
  ###convert to sequence of tokens
    input_sequences = []
    for line in corpus:
      token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
      n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)
 
    return input_sequences, total_words
inp_sequences, total_words = get_sequence_of_tokens(corpus)

#pad variable-length sequences, build an n-gram sequence as a prediction and the following n-gram word as a label
def generate_padded_sequences(input_sequences):
  max_sequence_len = max([len(x) for x in input_sequences])
  input_sequences = np.array(pad_sequences(input_sequences,  maxlen=max_sequence_len, padding='pre'))
  predictors, label = input_sequences[:,:-1], input_sequences[:, -1]
  label = ku.to_categorical(label, num_classes = total_words)
  return predictors, label, max_sequence_len
predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

#train LSTM model
def create_model(max_sequence_len, total_words):
  input_len = max_sequence_len - 1
  model = Sequential()
  ##add input embedding layer
  model.add(Embedding(total_words, 10, input_length=input_len))
  ##add hidden layer 1 — LSTM layer
  model.add(LSTM(100))
  model.add(Dropout(0.1))
  ##add output layer
  model.add(Dense(total_words, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model

model = create_model(max_sequence_len, total_words)
model.fit(predictors, label, epochs=20, verbose=5)

#make predictions
def generate_text(seed_text, next_words, model, max_sequence_len):
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,  padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
   
    output_word = ""
    for word,index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
      break
    seed_text += " " +output_word
    return seed_text.title()

print(generate_text("spartan", 5, model, max_sequence_len))

