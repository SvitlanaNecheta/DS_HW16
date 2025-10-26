import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# --- Дані ---
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)   # (60000, 28, 28)
print(y_train.shape)   # (60000,)
print(x_test.shape)    # (10000, 28, 28)
print(y_test.shape)    # (10000,)
#робимо стандартизацію
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train_cat=keras.utils.to_categorical(y_train,10)
y_test_cat=keras.utils.to_categorical(y_test  ,10)

# Виводимо перші 25 зображень
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i],cmap=plt.cm.binary)
plt.show()
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
print(model.summary())
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# навчання берем 10 епох 
history = model.fit(
    x_train, y_train_cat,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=2  # або verbose=1
)
model.evaluate(x_test,y_test_cat)
import pickle

# Збереження моделі
model.save('cnn_model.h5')  

# Збереження історії навчання
with open('cnn_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

