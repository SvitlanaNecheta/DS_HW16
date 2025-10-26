import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# 1. Завантажуємо дані
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2. Класи
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 3. Зберігаємо 15 перших зображень як PNG
for i in range(15):
    plt.imsave(f"test_image_{i}_{class_names[y_test[i]]}.png", x_test[i], cmap='gray')
