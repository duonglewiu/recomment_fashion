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

    # ===============================
    # === Recommended Images (New) ===
    # ===============================
    st.subheader("Recommended Images")

    # CSS để căn giữa ảnh và bo góc
    st.markdown("""
    <style>
        .stImage > img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 10px;
            transition: transform 0.2s ease;
        }
        .stImage > img:hover {
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

    # Tạo biến lưu ảnh đang được chọn
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None

    # Hiển thị ảnh gợi ý dưới dạng lưới 5 cột
    # Hiển thị ảnh gợi ý kèm similarity score (cách 2)
    cols = st.columns(5)
    for i, col in enumerate(cols, start=1):
        img_path = filenames[indices[0][i]]
        if os.path.exists(img_path):
            # Tính điểm tương đồng (0–1, càng cao càng giống)
            score = 1 / (1 + distance[0][i])
            score_percent = score * 100  # để hiển thị dễ hiểu hơn

            # Hiển thị ảnh
            col.image(img_path, width=150, caption=f"Similarity: {score_percent:.1f}%")

            # Nút xem chi tiết ảnh
            if col.button(f"👕 View {i}", key=f"view_{i}"):
                st.session_state.selected_image = img_path
        else:
            col.warning(f"⚠️ Missing: {img_path}")

    # Khi người dùng chọn ảnh
    import cv2
    import numpy as np
    from PIL import Image
    import pandas as pd
    from datetime import datetime

    if st.session_state.selected_image:
        st.markdown("---")
        st.markdown("### 🔎 Enhanced View (Fast Sharpened Bicubic)")

        img = Image.open(st.session_state.selected_image)
        img_cv = np.array(img)

        # Phóng to 2x bằng nội suy Bicubic
        img_up = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Áp bộ lọc sharpen nhẹ
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_sharp = cv2.filter2D(img_up, -1, kernel)

        st.image(img_sharp, width=500, caption="Enhanced Image (Sharpened)")

        # ==============================
        # === Feedback Section ===
        # ==============================
        st.markdown("### ❤️ Feedback")

        # Lấy điểm tương đồng của ảnh đã chọn
        idx = filenames.index(st.session_state.selected_image)
        sim_distance = None
        for i, id_in_top in enumerate(indices[0]):
            if filenames[id_in_top] == st.session_state.selected_image:
                sim_distance = distance[0][i]
                break
        similarity = 1 / (1 + sim_distance) if sim_distance is not None else None

        colA, colB = st.columns(2)
        if colA.button("👍 Like"):
            feedback = "Like"
        elif colB.button("👎 Dislike"):
            feedback = "Dislike"
        else:
            feedback = None

        # Nếu người dùng có phản hồi → lưu lại
        if feedback:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query_image": upload_file.name,
                "recommended_image": os.path.basename(st.session_state.selected_image),
                "similarity_score": round(similarity, 4) if similarity else "N/A",
                "feedback": feedback
            }

            df = pd.DataFrame([record])
            if not os.path.exists("feedback.csv"):
                df.to_csv("feedback.csv", index=False)
            else:
                df.to_csv("feedback.csv", mode="a", header=False, index=False)

            st.success(f"✅ Feedback saved: {feedback}")


    # ==============================
    # === Feedback Statistics ===
    # ==============================
    st.markdown("---")
    st.markdown("## 📊 Feedback Statistics")

    import pandas as pd
    import matplotlib.pyplot as plt

    if os.path.exists("feedback.csv"):
        df = pd.read_csv("feedback.csv")

        # Thống kê Like / Dislike
        feedback_counts = df["feedback"].value_counts()
        total = feedback_counts.sum()

        likes = feedback_counts.get("Like", 0)
        dislikes = feedback_counts.get("Dislike", 0)

        like_ratio = (likes / total * 100) if total > 0 else 0
        dislike_ratio = (dislikes / total * 100) if total > 0 else 0

        st.write(f"👍 **{likes} Likes** ({like_ratio:.1f}%)")
        st.write(f"👎 **{dislikes} Dislikes** ({dislike_ratio:.1f}%)")

        # Vẽ biểu đồ cột
        fig, ax = plt.subplots()
        ax.bar(feedback_counts.index, feedback_counts.values, color=["green", "red"])
        ax.set_title("User Feedback Summary")
        ax.set_ylabel("Number of Feedbacks")
        ax.set_xlabel("Feedback Type")

        st.pyplot(fig)

        # Xem toàn bộ dữ liệu (nếu muốn)
        with st.expander("📄 View All Feedback Data"):
            st.dataframe(df)
    else:
        st.info("🕐 No feedback data available yet.")


