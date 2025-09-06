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
no_hand_timeout = 2  # seconds to wait after no hand detected

def run_camera():
    global last_prediction, last_spoken_time, speaking, no_hand_start_time

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

    print("Starting real-time prediction... Remove hand for 2 seconds to switch to texttospeech.py.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_char = ""

        if results.multi_hand_landmarks:
            # Reset no-hand timer since a hand is detected
            no_hand_start_time = None  

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                landmarks_normalized = []
                for lm in hand_landmarks.landmark:
                    landmarks_normalized.append(lm.x - wrist.x)
                    landmarks_normalized.append(lm.y - wrist.y)

                prediction = model.predict(np.array([landmarks_normalized]), verbose=0)
                predicted_class_index = np.argmax(prediction)
                predicted_char = label_encoder.inverse_transform([predicted_class_index])[0]

                current_time = time.time()

                if (predicted_char != last_prediction) and \
                   ((current_time - last_spoken_time) > cooldown_sec) and \
                   (not speaking):
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
                no_hand_start_time = current_time  # Start timer when hand no longer detected

            # If no hand detected for longer than timeout, break loop for next step
            if current_time - no_hand_start_time > no_hand_timeout:
                print(f"No hand detected for {no_hand_timeout} seconds, switching to texttospeech.py.")
                break

            if current_time - last_spoken_time > cooldown_sec:
                last_prediction = None

        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'Prediction: {predicted_char}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Detection', frame)
        time.sleep(0.05)

        key = cv2.waitKey(1) & 0xFF
        # Remove manual 'q' exit option as transition now automatic
        if key == 27:  # ESC key to fully quit program anytime
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
