import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from function import actions, no_sequences, sequence_length, DATA_PATH

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

sequences_cleaned = []
labels_cleaned = []

for i, seq in enumerate(sequences):
    if len(seq) != sequence_length:
        print(f"Problematic sequence at index {i}: Incorrect sequence length")
    else:
        valid_sequence = True
        for j, frame in enumerate(seq):
            try:
                if len(frame) != 63:
                    print(f"Problematic frame at sequence {i}, frame {j}: Incorrect frame length {len(frame)}, content: {frame}")
                    valid_sequence = False
                    break
            except TypeError as e:
                print(f"Problematic frame at sequence {i}, frame {j}: TypeError: {e}, content: {frame}")
                valid_sequence = False
                break
        if valid_sequence:
            sequences_cleaned.append(seq)
            labels_cleaned.append(labels[i])

X = np.array(sequences_cleaned)
y = to_categorical(labels_cleaned).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
