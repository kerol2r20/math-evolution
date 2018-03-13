from Formula_generator import simpleadd
from MathSymbolTokenizer import MathSymbolTokenizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Activation
from keras.utils.np_utils import to_categorical
import numpy as np

SPLIT_RATE = 0.1
BATCH_SIZE = 50000

# 訓練集資料筆數
train_len = int(BATCH_SIZE*SPLIT_RATE)

batch_generator = simpleadd(BATCH_SIZE, min=1, max=999, padding=True)

tokenizer = MathSymbolTokenizer("0123456789+ ")

model = Sequential()
model.add(LSTM(units=128, input_shape=(7, 13)))
model.add(RepeatVector(4))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(13)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Generate data
(x, y) = next(batch_generator)

# 對 input 和 output 進行 one-hot encoding
x_seq = to_categorical(tokenizer.text2seq(x))
y_seq = to_categorical(tokenizer.text2seq(y))

# 分離訓練集與測試集資料
x_validate = x_seq[:train_len]
x_train = x_seq[train_len:]
y_validate = y_seq[:train_len]
y_train = y_seq[train_len:]

model.fit(x=x_train, y=y_train, batch_size=128, epochs=30, validation_data=(x_validate, y_validate))
