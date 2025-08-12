# Deep-Learning Exp-1:
#Code:
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, epochs=1000, verbose=0)

loss, acc = model.evaluate(X, Y)
print("Accuracy:", acc)

print("Predictions:", model.predict(X))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

#Output Screenshot(with code):
<img width="1514" height="617" alt="dl1" src="https://github.com/user-attachments/assets/8d1cf3e8-6e67-4051-ad65-8299f8d8d25f" />
<img width="1492" height="637" alt="dl2" src="https://github.com/user-attachments/assets/0239e2d4-52cc-4ab5-8974-6677b09f7e97" />

