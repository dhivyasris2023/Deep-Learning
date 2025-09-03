Code:
#Test case:
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

faces = fetch_olivetti_faces()
X, y = faces.images, faces.target

X = X.reshape(-1, 64, 64, 1).astype('float32')
X /= 1.0  

y_cat = to_categorical(y, num_classes=40)

X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X, y_cat, y, test_size=0.2, random_state=42, stratify=y
)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(40, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)

print(f"{'Input Face Image':20s} {'Expected Identity':20s} {'Predicted Identity':20s} {'Correct (Y/N)'}")

for i in range(3):
    expected = f"Person {y_test_labels[i]}"
    predicted = f"Person {pred_labels[i]}"
    correct = "Y" if y_test_labels[i] == pred_labels[i] else "N"
    print(f"Image {i+1:<16d} {expected:20s} {predicted:20s} {correct}")

Output:
<img width="1285" height="386" alt="Exp3 Dl testcase" src="https://github.com/user-attachments/assets/f450d519-dde6-462c-94ab-4b3a8acac61c" />
