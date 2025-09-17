code:
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
tf.random.set_seed(42)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
maxlen = 100
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=maxlen),
    LSTM(100),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

Output:
<img width="853" height="186" alt="image" src="https://github.com/user-attachments/assets/9e1289f6-6472-42ca-8358-158e9c89dd30" />
