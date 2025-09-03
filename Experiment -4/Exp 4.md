Code:
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


data = "Deep learning is amazing. Deep learning builds intelligent systems."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

sequences = []
words = data.split()
for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))

encoded = tokenizer.texts_to_sequences(sequences)
max_len = max([len(x) for x in encoded])

X = np.array([x[:-1] for x in pad_sequences(encoded, maxlen=max_len)])
y = to_categorical([x[-1] for x in pad_sequences(encoded, maxlen=max_len)],
                   num_classes=len(tokenizer.word_index)+1)

for i, seq in enumerate(encoded):
    input_seq = seq[:-1]
    output_word = seq[-1]
    input_words = [w for w, idx in tokenizer.word_index.items() if idx in input_seq]
    output_word_text = [w for w, idx in tokenizer.word_index.items() if idx == output_word][0]
    print(f"{input_words} -> '{output_word_text}'")

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=max_len-1),
    SimpleRNN(50),
    Dense(len(tokenizer.word_index)+1, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

Output:
<img width="633" height="324" alt="DL Exp4 1(OP)" src="https://github.com/user-attachments/assets/05e3c2d4-c950-4128-ab60-350b3d14b898" />
<img width="548" height="340" alt="DL Exp4 2(OP)" src="https://github.com/user-attachments/assets/cb49155c-82b6-40e8-8bd2-8c803d53e48a" />

