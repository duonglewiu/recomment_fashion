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
import pandas as pd

# ==============================
# HEADER
# ==============================
st.header('👗 Fashion Recommendation & Image Processing System')

# === Sidebar chọn chế độ ===
mode = st.sidebar.radio(
    "🧩 Chọn chế độ làm việc",
    ["Recommendation System", "Image Processing Tools"]
)

# ==============================
# PHẦN 1 — GỢI Ý ẢNH TƯƠNG TỰ
# ==============================
# ==============================
# PHẦN 1 — GỢI Ý ẢNH TƯƠNG TỰ
# ==============================
if mode == "Recommendation System":
    # === Load features và filenames ===
    Image_features = pkl.load(open('Images_features.pkl', 'rb'))
    filenames = pkl.load(open('filenames.pkl', 'rb'))
    filenames = [f.replace('/kaggle/input/fashion-product-images-small/images/', 'images/') for f in filenames]

    # === Đọc feedback người dùng nếu có ===
    if os.path.exists("feedback.csv"):
        st.info("🔁 Loading user feedback to adapt recommendation system...")
        df_feedback = pd.read_csv("feedback.csv")

        # --- Lọc ảnh Like / Dislike ---
        liked_imgs = df_feedback[df_feedback["feedback"] == "Like"]["recommended_image"]
        disliked_imgs = df_feedback[df_feedback["feedback"] == "Dislike"]["recommended_image"]

        liked_vectors, disliked_vectors = [], []

        for img_name in liked_imgs:
            full_path = f"images/{img_name}" if not img_name.startswith("images/") else img_name
            if full_path in filenames:
                idx = filenames.index(full_path)
                liked_vectors.append(Image_features[idx])

        for img_name in disliked_imgs:
            full_path = f"images/{img_name}" if not img_name.startswith("images/") else img_name
            if full_path in filenames:
                idx = filenames.index(full_path)
                disliked_vectors.append(Image_features[idx])

        # --- Tính vector sở thích người dùng ---
        if len(liked_vectors) > 0:
            user_pref_vector = np.mean(liked_vectors, axis=0)
            if len(disliked_vectors) > 0:
                dislike_vector = np.mean(disliked_vectors, axis=0)
                user_pref_vector = user_pref_vector - 0.5 * dislike_vector

            user_pref_vector /= np.linalg.norm(user_pref_vector)

            # Lưu vector sở thích vào session để pha trộn vào truy vấn sau
            st.session_state.user_pref_vector = user_pref_vector

            st.success(f"✅ Feedback adaptation applied: {len(liked_vectors)} likes, {len(disliked_vectors)} dislikes.")
        else:
            st.info("ℹ️ No 'Like' feedback found yet. System unchanged.")
            st.session_state.user_pref_vector = None

    # === Chuẩn bị model trích đặc trưng ===
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

    os.makedirs('upload', exist_ok=True)
    upload_file = st.file_uploader("📤 Upload Image")

    if upload_file is not None:
        upload_path = os.path.join('upload', upload_file.name)
        with open(upload_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        st.subheader('📸 Uploaded Image')
        st.image(upload_path, use_container_width=True)

        # === Trích xuất đặc trưng ảnh truy vấn ===
        input_img_features = extract_features_from_images(upload_path, model)

        # === Pha trộn với vector sở thích người dùng (nếu có) ===
        if "user_pref_vector" in st.session_state and st.session_state.user_pref_vector is not None:
            alpha = 0.8  # trọng số cho ảnh truy vấn
            beta = 0.2   # trọng số cho vector sở thích
            input_img_features = alpha * input_img_features + beta * st.session_state.user_pref_vector
            input_img_features /= np.linalg.norm(input_img_features)
            st.info("✨ Query adjusted with user feedback (80% query + 20% preference).")

        distance, indices = neighbors.kneighbors([input_img_features])

        # ===============================
        # === HIỂN THỊ ẢNH GỢI Ý ===
        # ===============================
        st.subheader("🛍️ Recommended Images")

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

        if "selected_image" not in st.session_state:
            st.session_state.selected_image = None

        cols = st.columns(5)
        for i, col in enumerate(cols, start=1):
            img_path = filenames[indices[0][i]]
            if os.path.exists(img_path):
                score = 1 / (1 + distance[0][i])
                score_percent = score * 100
                col.image(img_path, width=150, caption=f"Similarity: {score_percent:.1f}%")
                if col.button(f"👕 View {i}", key=f"view_{i}"):
                    st.session_state.selected_image = img_path
            else:
                col.warning(f"⚠️ Missing: {img_path}")

        # ===============================
        # === XEM ẢNH CHI TIẾT + FEEDBACK ===
        # ===============================
        import cv2
        from datetime import datetime
        import matplotlib.pyplot as plt

        if st.session_state.selected_image:
            st.markdown("---")
            st.markdown("### 🔎 Enhanced View (Sharpened Bicubic)")

            img = Image.open(st.session_state.selected_image)
            img_cv = np.array(img)
            img_up = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img_sharp = cv2.filter2D(img_up, -1, kernel)
            st.image(img_sharp, width=500, caption="Enhanced Image")

            st.markdown("### ❤️ Feedback")
            colA, colB = st.columns(2)
            if colA.button("👍 Like"):
                feedback = "Like"
            elif colB.button("👎 Dislike"):
                feedback = "Dislike"
            else:
                feedback = None

            # === Ghi feedback an toàn ===
            if feedback:
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query_image": upload_file.name,
                    "recommended_image": os.path.basename(st.session_state.selected_image),
                    "similarity_score": round(float(1 / (1 + distance[0][1])), 4) if 'distance' in locals() else None,
                    "feedback": feedback
                }

                record_df = pd.DataFrame([record])
                if not os.path.exists("feedback.csv"):
                    record_df.to_csv("feedback.csv", index=False)
                else:
                    try:
                        old_df = pd.read_csv("feedback.csv")
                        for col in ["timestamp", "query_image", "recommended_image", "similarity_score", "feedback"]:
                            if col not in old_df.columns:
                                old_df[col] = None
                        new_df = pd.concat([old_df, record_df], ignore_index=True)
                        new_df.to_csv("feedback.csv", index=False)
                    except Exception as e:
                        st.error(f"⚠️ Error while updating feedback.csv: {e}")
                        record_df.to_csv("feedback_backup.csv", index=False)

                st.success(f"✅ Feedback saved safely: {feedback}")

        # ===============================
        # === THỐNG KÊ FEEDBACK ===
        # ===============================
        st.markdown("---")
        st.markdown("## 📊 Feedback Statistics")

        if os.path.exists("feedback.csv"):
            df = pd.read_csv("feedback.csv")
            feedback_counts = df["feedback"].value_counts()
            total = feedback_counts.sum()
            likes = feedback_counts.get("Like", 0)
            dislikes = feedback_counts.get("Dislike", 0)
            like_ratio = (likes / total * 100) if total > 0 else 0
            dislike_ratio = (dislikes / total * 100) if total > 0 else 0

            st.write(f"👍 **{likes} Likes** ({like_ratio:.1f}%)")
            st.write(f"👎 **{dislikes} Dislikes** ({dislike_ratio:.1f}%)")

            fig, ax = plt.subplots()
            ax.bar(feedback_counts.index, feedback_counts.values, color=["green", "red"])
            ax.set_title("User Feedback Summary")
            ax.set_ylabel("Number of Feedbacks")
            ax.set_xlabel("Feedback Type")
            st.pyplot(fig)

            with st.expander("📄 View All Feedback Data"):
                st.dataframe(df)
        else:
            st.info("🕐 No feedback data available yet.")


# ==============================
# PHẦN 2 — IMAGE PROCESSING TOOLS
# ==============================
else:
    st.subheader("🧠 Image Processing Tools")

    file = st.file_uploader("📤 Upload an image to process", type=["jpg", "png", "jpeg"])
    if file is not None:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Original Image", use_column_width=True)

        task = st.selectbox(
            "🔧 Choose processing task",
            ["Color Analysis", "Edge Detection","Logo Detection"]
        )

        import cv2
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        # --------------------------
        # 1. COLOR ANALYSIS
        # --------------------------
        if task == "Color Analysis":
            st.markdown("### 🎨 Dominant Colors & RGB Histogram")

            img_reshape = img_np.reshape((-1, 3))
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(img_reshape)
            colors = np.array(kmeans.cluster_centers_, dtype="uint8")

            # Hiển thị thanh màu
            bar = np.zeros((50, 300, 3), dtype="uint8")
            start_x = 0
            for i, color in enumerate(colors):
                end_x = start_x + 60
                cv2.rectangle(bar, (start_x, 0), (end_x, 50),
                              color.tolist(), -1)
                start_x = end_x
            st.image(bar, caption="Top 5 Dominant Colors")

            # Biểu đồ RGB
            fig, ax = plt.subplots()
            for i, col in enumerate(['r', 'g', 'b']):
                hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
            ax.set_title("RGB Histogram")
            st.pyplot(fig)

        # --------------------------
        # 2. EDGE DETECTION
        # --------------------------
        elif task == "Edge Detection":
            st.markdown("### ✴️ Edge Detection Methods")

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # Tính Sobel và Laplacian
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            canny = cv2.Canny(gray, 100, 200)

            # 🔧 Chuẩn hóa kết quả về [0, 255] và ép kiểu uint8
            sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Hiển thị ảnh
            st.image(sobel_norm, caption="Sobel Gradient", use_container_width=True)
            st.image(canny, caption="Canny Edges", use_container_width=True)
            st.image(lap_norm, caption="Laplacian", use_container_width=True)

            st.success("✅ Edge detection completed successfully.")
        # --------------------------
        # 3. LOGO DETECTION (YOLOv8)
        # --------------------------
        elif task == "Logo Detection":
            st.markdown("### 🏷️ Logo Detection using YOLOv8 (Pretrained Model)")

            from ultralytics import YOLO
            import cv2
            import tempfile
            import numpy as np
            import sys

            sys.modules['numpy'] = np  # patch tránh lỗi "Numpy not available"

            # =========================
            # 1. Load pretrained model
            # =========================
            # Model này đã được train sẵn để phát hiện logo phổ biến (Nike, Adidas, Apple, etc.)
            model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-detector.pt"
            model = YOLO(model_url)

            # =========================
            # 2. Lưu ảnh upload tạm thời
            # =========================
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                img = np.array(img)
                cv2.imwrite(tmpfile.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                tmp_path = tmpfile.name

            # =========================
            # 3. Chạy nhận diện logo
            # =========================
            st.info("⏳ Detecting logos... please wait...")
            results = model(tmp_path)
            result_img = results[0].plot()  # Vẽ bounding boxes

            # =========================
            # 4. Hiển thị kết quả
            # =========================
            st.image(result_img, caption="Detected Logos", use_container_width=True)

            # Bảng kết quả chi tiết
            df_results = results[0].boxes.data.cpu().numpy()
            if len(df_results) > 0:
                import pandas as pd

                df = pd.DataFrame(df_results, columns=["x1", "y1", "x2", "y2", "confidence", "class_id"])
                df["label"] = [model.names[int(i)] for i in df["class_id"]]
                st.dataframe(df[["label", "confidence"]], use_container_width=True)
            else:
                st.warning("❌ No logos detected in this image.")


