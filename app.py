import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Cài đặt giao diện người dùng Streamlit
st.title('Human-Pose')
stop_button_pressed = st.button('Stop')

# Load model đã train
model = tf.keras.models.load_model("/home/phan/Human-Pose/model_weight/best_lstm_model.keras", safe_mode=False)

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Biến global
label = "Khởi động..."
n_time_steps = 10  # Số khung hình đầu vào
lm_list = []
warmup_frames = 30  # Số frame chờ trước khi bắt đầu dự đoán

# Tạo danh sách class hành động
classes = ["LAM VIEC", "NGA LUNG", "NAM NGU", "GAC CHAN", "DUNG DAY","DI LAI"]

# Biến đo FPS
prev_time = 0

# 🏎 Hàm trích xuất khung xương từ MediaPipe
def extract_landmarks(results):
    return [coord for lm in results.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)]

# 🏎 Vẽ khung xương lên ảnh
def draw_landmarks(mp_draw, results, img):
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return img

# 🏎 Vẽ nhãn hành động lên ảnh
def draw_label(img, label):
    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

# Hàm dự đoán hành động
def detect_action(model, lm_list):
    global label
    lm_array = np.array(lm_list).reshape(1, n_time_steps, -1)  # Reshape dữ liệu đầu vào
    results = model.predict(lm_array)  # Dự đoán
    predicted_class = np.argmax(results)  # Lấy class có xác suất cao nhất
    label = classes[predicted_class]  # Gán nhãn
    return label

# Lớp xử lý video từ webcam
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.lm_list = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Chuyển ảnh về RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        # Đo FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Nếu đủ warmup thì bắt đầu nhận diện
        if self.frame_count > warmup_frames:
            if results.pose_landmarks:
                # Trích xuất khung xương
                lm = extract_landmarks(results)
                self.lm_list.append(lm)

                # Khi đủ 10 frame, bắt đầu dự đoán
                if len(self.lm_list) == n_time_steps:
                    t = threading.Thread(target=detect_action, args=(model, self.lm_list,))
                    t.start()
                    self.lm_list = []  # Reset danh sách

                # Vẽ khung xương
                img = draw_landmarks(mp_draw, results, img)

        # Vẽ nhãn lên ảnh
        img = draw_label(img, label)
        # Hiển thị FPS
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        self.frame_count += 1
        return img

# Chạy WebRTC streamer
webrtc_streamer(key="pose-detection", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
