code: 

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=1000, verbose=0)

predictions = model.predict(X)
print("\n--- Test 2: Single-Layer Perceptron (Observation) ---")
print("Input\tPredicted\tExpected\tRemark")
for i, p in enumerate(predictions):
    predicted_binary = 1 if p[0] >= 0.5 else 0
    remark = "Correct" if predicted_binary == Y[i][0] else "May fail"
    print(f"{X[i]}\t{predicted_binary} ({p[0]:.4f})\t{Y[i][0]}\t\t{remark}")
