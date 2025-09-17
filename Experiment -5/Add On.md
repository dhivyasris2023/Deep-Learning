Code:
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
model = Sequential([
Embedding(10000, 32, input_length=100),
GRU(100),
Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

Output:
<img width="1045" height="211" alt="image" src="https://github.com/user-attachments/assets/32b364f6-f888-4611-a095-d841f03acf1f" />
