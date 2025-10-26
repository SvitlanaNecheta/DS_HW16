
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist   
import numpy as np   

#  Завантаження та підготовка даних
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# VGG16 очікує 3-канальні зображення 224x224 - ноут мій не потягнув, то беру меньше -96х96
x_train = np.stack([x_train]*3, axis=-1)   # робимо 3 канали
x_test = np.stack([x_test]*3, axis=-1)

# Еренувальний та тестовий датасети:
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# 4.Препроцесинг:
def preprocess(img, label):
    img = tf.image.resize(img, (96,96))   # 96x96
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.one_hot(label, 10)         # one-hot прямо тут
    return img, label

# Застосовуємо препроцесинг:
batch_size = 64
train_ds = (train_ds
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds
           .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(batch_size)
           .prefetch(tf.data.AUTOTUNE))

for images, labels in train_ds.take(1):
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(96,96,3))
conv_base.trainable = False

model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, epochs=5, validation_data=test_ds) #5 епох   
#повільно навчається  

print(history.history.keys())

import pickle

model.save('vgg_model.h5')

with open('vgg_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

with open('vgg_classes.txt', 'w') as f:
    for c in class_names:
        f.write(f"{c}\n")
    