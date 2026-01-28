import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))


X = X.astype('float32') / 255.0

animals = [2, 3, 4, 5, 6, 7]
vehicles = [0, 1, 8, 9]          

y_binary = np.zeros_like(y)

for i in range(len(y)):
    if y[i] in animals:
        y_binary[i] = 0   
    else:
        y_binary[i] = 1


X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.7, random_state=42
)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


model_1 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model_1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_2 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model_2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_3 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])

model_3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Trening modelu 1 (1 warstwa CNN)")
history_1 = model_1.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

print("\nTrening modelu 2 (2 warstwy CNN)")
history_2 = model_2.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

print("\nTrening modelu 3 (3 warstwy CNN)")
history_3 = model_3.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

acc1 = model_1.evaluate(X_test, y_test, verbose=0)[1]
acc2 = model_2.evaluate(X_test, y_test, verbose=0)[1]
acc3 = model_3.evaluate(X_test, y_test, verbose=0)[1]

print("\nWYNIKI KOŃCOWE:")
print(f"Model 1 (1 warstwa CNN): accuracy = {acc1:.4f}")
print(f"Model 2 (2 warstwy CNN): accuracy = {acc2:.4f}")
print(f"Model 3 (3 warstwy CNN): accuracy = {acc3:.4f}")

