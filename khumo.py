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
import cv2  # Thêm thư viện OpenCV

st.set_page_config(layout="wide")
st.header('👕 Fashion Recommendation System 👚')


# === Load data ===
@st.cache_resource
def load_data():
    try:
        Image_features = pkl.load(open('Images_features.pkl', 'rb'))
        filenames = pkl.load(open('filenames.pkl', 'rb'))

        # --- ĐÃ SỬA (QUAN TRỌNG) ---
        # Thay thế đường dẫn "cloud" (từ file pkl) bằng đường dẫn "local"
        # Giả sử thư mục ảnh local của bạn tên là 'images'

        # Xác định đường dẫn gốc trên cloud (dựa trên lỗi)
        cloud_base_path = "/kaggle/input/fashion-product-images-small/images/"
        local_base_path = "images/"  # Thư mục local chứa ảnh

        filenames = [f.replace(cloud_base_path, local_base_path) for f in filenames]

        st.success(f"Đã tải {len(filenames)} đường dẫn ảnh và sửa về local. (Ví dụ: {filenames[0]})")

        return Image_features, filenames
    except FileNotFoundError as e:
        st.error(f"Lỗi: Không tìm thấy file {e.filename}.")
        st.error("Hãy chắc chắn rằng bạn có 'Images_features.pkl' và 'filenames.pkl' trong cùng thư mục.")
        return None, None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
        return None, None


Image_features, filenames = load_data()


# === Model setup ===
@st.cache_resource
def get_model():
    # ... (Phần còn lại của file giữ nguyên) ...
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model_sequential = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
    return model_sequential


# Hàm trích xuất đặc trưng
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result


# === Model Super-Resolution (LÀM NÉT) ===
@st.cache_resource
def get_super_res_model():
    """Tải model AI làm nét ảnh (chỉ tải 1 lần)"""
    model_path = "FSRCNN_x4.pb"  # ĐÃ SỬA: từ gạch ngang (-) thành gạch dưới (_)
    if not os.path.exists(model_path):
        st.error(f"Lỗi: Không tìm thấy model Super-Resolution '{model_path}'.")
        st.info("Vui lòng tải file 'FSRCNN-x4.pb' về cùng thư mục với app.")
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", 4)  # Tên thuật toán "fsrcnn", phóng to 4 lần (x4)
        return sr
    except cv2.error as e:
        st.error(f"Lỗi khi tải model OpenCV. Hãy chắc chắn bạn đã cài 'opencv-python-contrib'. Lỗi: {e}")
        return None


# Chỉ chạy nếu đã load data thành công
if Image_features is not None and filenames is not None:
    model = get_model()
    sr_model = get_super_res_model()  # Tải model làm nét


    # === Neighbors setup ===
    @st.cache_resource
    def get_neighbors():
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(Image_features)
        return neighbors


    neighbors = get_neighbors()

    # === Upload ảnh ===
    os.makedirs('upload', exist_ok=True)
    upload_file = st.file_uploader("Upload an image to find similar fashion items")

    if upload_file is not None:
        upload_path = os.path.join('upload', upload_file.name)
        with open(upload_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        # Chia layout: 1 cột cho ảnh upload, 1 cột cho ảnh nét
        col1, col2 = st.columns([1, 2])  # Cột ảnh nét rộng gấp đôi

        with col1:
            st.subheader('Uploaded Image')
            st.image(upload_path, caption="Your Upload", width=250)

        # Trích xuất đặc trưng từ ảnh upload
        input_img_features = extract_features_from_images(upload_path, model)
        distance, indices = neighbors.kneighbors([input_img_features])

        # ===============================
        # === Recommended Images ===
        # ===============================
        st.subheader("Recommended Images")

        # CSS để bo góc ảnh
        st.markdown("""
        <style>
            .stImage > img {
                border-radius: 10px;
            }
            .stButton { 
                display: flex;
                justify-content: center;
            }
        </style>
        """, unsafe_allow_html=True)

        if "selected_image" not in st.session_state:
            st.session_state.selected_image = None

        cols = st.columns(5)
        recommended_image_paths = []
        for i, col in enumerate(cols, start=1):
            img_path = filenames[indices[0][i]]
            recommended_image_paths.append(img_path)

            if os.path.exists(img_path):
                col.image(img_path, caption=f"Similar {i}", use_column_width=True)
                if col.button(f"View Details {i}", key=f"view_{i}", use_container_width=True):
                    st.session_state.selected_image = img_path
            else:
                col.warning(f"⚠️ Missing: {img_path}")

        # Ghi đè ảnh đầu tiên nếu không có gì được chọn
        if st.session_state.selected_image is None and recommended_image_paths:
            st.session_state.selected_image = recommended_image_paths[0]

        # ===============================
        # === HIỂN THỊ ẢNH NÉT (ĐÃ SỬA) ===
        # ===============================
        with col2:
            if st.session_state.selected_image:
                if os.path.exists(st.session_state.selected_image):
                    st.subheader("🔎 AI Enhanced View (x4)")

                    # Kiểm tra xem model AI đã tải được chưa
                    if sr_model is None:
                        st.warning("Không thể làm nét ảnh (model AI chưa sẵn sàng). Hiển thị ảnh gốc.")
                        st.image(st.session_state.selected_image,
                                 caption="Original Image (Low-Res)",
                                 use_column_width=True)
                    else:
                        try:
                            # Tải ảnh gốc (bằng cv2)
                            img_goc = cv2.imread(st.session_state.selected_image)

                            if img_goc is None:
                                st.error(f"Không thể đọc file ảnh: {st.session_state.selected_image}")
                            else:
                                # (Tùy chọn) Hiển thị spinner
                                with st.spinner(f"Đang dùng AI để làm nét (phóng to x4)..."):
                                    result_net = sr_model.upsample(img_goc)

                                # Hiển thị ảnh đã làm nét
                                st.image(result_net,
                                         caption="AI Upscaled Image",
                                         use_column_width=True,
                                         channels="BGR")  # Quan trọng: cv2 đọc là BGR

                                # (Tùy chọn) Hiển thị kích thước thật
                                st.caption(
                                    f"Original: {img_goc.shape[1]}x{img_goc.shape[0]} px | Enhanced: {result_net.shape[1]}x{result_net.shape[0]} px")

                        except Exception as e:
                            st.error(f"Lỗi khi đang làm nét ảnh: {e}")
                            st.image(st.session_state.selected_image,
                                     caption="Fallback to Original Image",
                                     use_column_width=True)  # Hiển thị ảnh gốc nếu có lỗi
                else:
                    st.error(f"Không thể tìm thấy file ảnh: {st.session_state.selected_image}")

else:
    st.info("Vui lòng tải file 'Images_features.pkl' và 'filenames.pkl' để bắt đầu.")



