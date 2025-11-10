import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
from PIL import Image

st.header('Fashion Recommendation System')

# === Load data ===
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# === FIX đường dẫn từ Kaggle sang local ===
# Giả sử bạn có folder "images" nằm cùng thư mục với app.py
filenames = [f.replace('/kaggle/input/fashion-product-images-small/images/', 'images/') for f in filenames]

# === Model setup ===
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# === Upload ảnh ===
os.makedirs('upload', exist_ok=True)
upload_file = st.file_uploader("Upload Image")

if upload_file is not None:
    upload_path = os.path.join('upload', upload_file.name)
    with open(upload_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader('Uploaded Image')
    st.image(upload_path)

    # Trích xuất đặc trưng từ ảnh upload
    input_img_features = extract_features_from_images(upload_path, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    st.subheader('Recommended Images')
    cols = st.columns(5)
    for i, col in enumerate(cols, start=1):
        img_path = filenames[indices[0][i]]
        if os.path.exists(img_path):
            col.image(img_path)
        else:
            col.warning(f"❌ Không tìm thấy {img_path}")
