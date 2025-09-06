import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Configuration ---
DATA_DIR = 'D:/new sign language MODEL/sign language/sign_language_dataset'
MODEL_FILENAME = 'sign_language_model.p'
LABELS_FILENAME = 'sign_language_labels.p'

# --- Hand Landmark Extraction Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

data = []
labels = []

print("Starting video data processing...")

# Iterate through each sign's folder (each containing MOV files)
for sign_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, sign_folder)
    if not os.path.isdir(folder_path):
        continue

    print(f'Processing videos in: {folder_path}')

    for vid_file in os.listdir(folder_path):
        if not vid_file.lower().endswith('.mov'):
            continue

        vid_path = os.path.join(folder_path, vid_file)
        cap = cv2.VideoCapture(vid_path)

        if not cap.isOpened():
            print(f"Error opening video file {vid_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"No frames in video {vid_path}, skipping.")
            cap.release()
            continue

        # Number of frames to sample per video
        N = 5
        for i in range(N):
            frame_number = int(i * frame_count / N)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Could not read frame {frame_number} from {vid_path}")
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    landmarks_normalized = []
                    for lm in hand_landmarks.landmark:
                        landmarks_normalized.append(lm.x - wrist_landmark.x)
                        landmarks_normalized.append(lm.y - wrist_landmark.y)
                    data.append(landmarks_normalized)
                    labels.append(sign_folder)

        cap.release()

print(f"Total frames processed: {len(data)}")

# --- Data Preparation ---
data = np.asarray(data)
labels = np.asarray(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, 
                                                    test_size=0.2, shuffle=True, stratify=labels_encoded)

# --- Model Building ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(data[0]),)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels_encoded)), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Model Training ---
print("\nStarting model training...")
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
print("Model training finished.")

# --- Save the Model and Labels ---
with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump({'model': model}, f)

with open(LABELS_FILENAME, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model saved to {MODEL_FILENAME}")
print(f"Labels saved to {LABELS_FILENAME}")
