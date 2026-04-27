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
DATA_DIR = 'C:/Users/Soumo/OneDrive/Desktop/sign language/sign language/sign language/sign_language_dataset'
MODEL_FILENAME = 'sign_language_model.p'
LABELS_FILENAME = 'sign_language_labels.p'

# --- Hand Landmark Extraction ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data = []
labels = []

print("Starting data processing...")
for sign_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, sign_folder)
    if not os.path.isdir(folder_path):
        continue

    print(f'Processing images in: {folder_path}')
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        feature_vector = []
        if results.multi_hand_landmarks:
            hand_landmarks_list = results.multi_hand_landmarks
            # For each of 2 hands, pad with zeros if missing
            for h in range(2):
                if h < len(hand_landmarks_list):
                    hand_landmarks = hand_landmarks_list[h]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    for lm in hand_landmarks.landmark:
                        feature_vector.append(lm.x - wrist.x)
                        feature_vector.append(lm.y - wrist.y)
                else:
                    feature_vector.extend([0.0] * 42)
        else:
            feature_vector.extend([0.0] * 84)

        data.append(feature_vector)
        labels.append(sign_folder)

print("Data processing finished.")

# --- Data Preparation ---
data = np.asarray(data)
labels = np.asarray(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
)

# --- Model Building ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels_encoded)), activation='softmax')
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
