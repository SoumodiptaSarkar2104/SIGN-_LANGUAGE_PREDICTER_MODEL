import os
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import mediapipe as mp

# --- Configuration ---
DATA_DIR = 'D:/new sign language MODEL/sign language/sign_language_dataset'
MODEL_FILENAME = 'sign_language_model.p'
LABELS_FILENAME = 'sign_language_labels.p'

# --- Initialize MediaPipe Hands and Face Mesh ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

data = []
labels = []

print("Starting video data processing for hands and face landmarks...")

# Process videos folder-wise
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

        N = 5  # number of frames to sample per video

        for i in range(N):
            frame_number = int(i * frame_count / N)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Could not read frame {frame_number} from {vid_path}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands and face landmarks
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)

            features = []

            # Extract hand landmarks for up to 2 hands, normalized by respective wrist(s)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    for lm in hand_landmarks.landmark:
                        features.append(lm.x - wrist.x)
                        features.append(lm.y - wrist.y)
                # Padding if only one hand detected (21 landmarks × 2 coords)
                if len(hand_results.multi_hand_landmarks) == 1:
                    features.extend([0.0] * 42)
            else:
                # No hand detected: pad both hands (2 × 21 landmarks × 2 coords)
                features.extend([0.0] * 84)

            # Extract face landmarks normalized by nose tip (landmark #1)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                nose_tip = face_landmarks.landmark[1]
                for lm in face_landmarks.landmark:
                    features.append(lm.x - nose_tip.x)
                    features.append(lm.y - nose_tip.y)
            else:
                # No face detected: pad with zeros (468 landmarks × 2 coords)
                features.extend([0.0] * 936)

            # Verify feature vector length (should be 84 + 936 = 1020)
            if len(features) == 1020:
                data.append(features)
                labels.append(sign_folder)
            else:
                print(f"Warning: Unexpected feature length {len(features)} in {vid_path} frame {frame_number}")

        cap.release()

print(f"Total samples processed: {len(data)}")

# --- Prepare data and train model ---

data = np.asarray(data)
labels = np.asarray(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

model = Sequential([
    Dense(256, activation='relu', input_shape=(data.shape[1],)),
    Dense(128, activation='relu'),
    Dense(len(np.unique(labels_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nStarting training...")
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
print("Training complete.")

# --- Save model and label encoder ---

with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump({'model': model}, f)

with open(LABELS_FILENAME, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model saved to {MODEL_FILENAME}")
print(f"Labels saved to {LABELS_FILENAME}")
