import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
import subprocess

# --- Load Model and Labels ---
MODEL_FILENAME = 'sign_language_model.p'
LABELS_FILENAME = 'sign_language_labels.p'

with open(MODEL_FILENAME, 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

with open(LABELS_FILENAME, 'rb') as f:
    label_encoder = pickle.load(f)

# --- Initialize TTS engine ---
engine = pyttsx3.init()
engine.setProperty('rate', 200)
engine.setProperty('volume', 1.0)

speaking = False

def on_start(name):
    global speaking
    speaking = True

def on_end(name, completed):
    global speaking
    speaking = False

engine.connect('started-utterance', on_start)
engine.connect('finished-utterance', on_end)

TEXT_FILE_PATH = "D:/new sign language MODEL/sign language/speak.txt"

last_prediction = None
cooldown_sec = 2
last_spoken_time = 0
no_hand_start_time = None
no_hand_timeout = 2  # seconds no hand triggers TTS

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

expected_feature_length = 1020  # 2 hands + face landmarks combined

def run_camera():
    global last_prediction, last_spoken_time, speaking, no_hand_start_time

    cap = cv2.VideoCapture(0)

    print("Starting real-time prediction...\n"
          "Remove hands for 2 seconds or press 'q' to switch to text-to-speech.\n"
          "Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        features = []

        if hand_results.multi_hand_landmarks:
            no_hand_start_time = None  # reset no-hand timer

            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - wrist.x)
                    features.append(lm.y - wrist.y)

            if len(hand_results.multi_hand_landmarks) == 1:
                features.extend([0.0] * 42)
        else:
            if no_hand_start_time is None:
                no_hand_start_time = time.time()
            features.extend([0.0] * 84)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[1]
            for lm in face_landmarks.landmark:
                features.append(lm.x - nose_tip.x)
                features.append(lm.y - nose_tip.y)
        else:
            features.extend([0.0] * 936)

        predicted_char = ""

        if len(features) == expected_feature_length:
            input_features = np.array([features], dtype=np.float32)

            prediction = model.predict(input_features, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_char = label_encoder.inverse_transform([predicted_class_index])[0]

            current_time = time.time()

            if (predicted_char != last_prediction) and ((current_time - last_spoken_time) > cooldown_sec) and (not speaking):
                print(f"Prediction: {predicted_char}")
                engine.say(predicted_char)
                engine.runAndWait()
                last_prediction = predicted_char
                last_spoken_time = current_time

                with open(TEXT_FILE_PATH, "a", encoding="utf-8") as out_file:
                    out_file.write(str(predicted_char) + "\n")

        else:
            current_time = time.time()
            if no_hand_start_time is None:
                no_hand_start_time = current_time

            if current_time - last_spoken_time > cooldown_sec:
                last_prediction = None

            if no_hand_start_time and (current_time - no_hand_start_time > no_hand_timeout):
                print(f"No hands detected for {no_hand_timeout} seconds, switching to text-to-speech.")
                break

        # Draw landmarks
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1))

        cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'Prediction: {predicted_char}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Detection', frame)
        time.sleep(0.05)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Manual trigger: switching to text-to-speech.")
            break
        if key == 27:  # ESC quits fully
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()
    cv2.destroyAllWindows()

def run_tts_script():
    print("Running texttospeech.py script...")
    subprocess.run(['python', 'texttospeech.py'])
    print("texttospeech.py completed. Returning to camera.")

if __name__ == "__main__":
    while True:
        run_camera()
        run_tts_script()
