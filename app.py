import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image

st.title("""
#  КЛАСИФІКАЦІЯ Fashion MNIST.
Цей вебдодаток демонструє роботу двох нейронних мереж для класифікації зображень із набору **Fashion MNIST**:
-  Згорткова нейронна мережа (CNN)
-  Модель на основі **VGG16**

Завантажте власне зображення одягу й подивіться, який клас модель передбачить 👇
""")

# Вибір моделі
model_choice = st.radio( "ОБЕРИТЬ МОДЕЛЬ:", ("Згорткова нейронна мережа (CNN)", "VGG16") )


if model_choice == "Згорткова нейронна мережа (CNN)":
    model_path = "cnn_model.h5"
    history_path = "cnn_history.pkl"
else:
    model_path = "vgg_model.h5"
    history_path = "vgg_history.pkl"

# Завантаження моделі 
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(model_path)

# --- Завантаження історії ---
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# Класи Fashion MNIST 
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# Графіки навчання 
st.subheader(f"📈 Графіки точності та втрат- {model_choice}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history['accuracy'], label='Train Accuracy')
ax1.plot(history['val_accuracy'], label='Val Accuracy')
ax1.legend()
ax1.set_title('Точність')

ax2.plot(history['loss'], label='Train Loss')
ax2.plot(history['val_loss'], label='Val Loss')
ax2.legend()
ax2.set_title('Функція втрат')
st.pyplot(fig)

# Завантаження зображення 
st.subheader("Завантажте зображення (28x28, ч/б)")
uploaded_file = st.file_uploader("Оберіть файл...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Завантажене зображення", use_column_width=False, width=200)

    if model_choice == "Згорткова нейронна мережа (CNN)":
        #  Попередня обробка для CNN 
        img_resized = image.resize((28, 28))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)
    else:
        # Попередня обробка для VGG16 
        img_rgb = image.convert("RGB")  # робимо 3 канали
        img_resized = img_rgb.resize((96, 96))  # як у тренуванні
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 96, 96, 3)

    # Передбачення
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)

    st.markdown(f"### ✅ Передбачений клас: **{class_names[predicted_class]}**")
    st.markdown(f"Впевненість моделі: **{confidence*100:.2f}%**")

    st.bar_chart(dict(zip(class_names, preds[0])))
