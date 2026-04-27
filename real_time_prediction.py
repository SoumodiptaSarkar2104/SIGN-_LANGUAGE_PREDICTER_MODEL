import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# --- Load Model and Labels ---
MODEL_FILENAME = 'C:\\Users\\Soumo\\OneDrive\\Desktop\\sign language\\sign language\\sign language\\sign_language_model.p'
LABELS_FILENAME = 'C:\\Users\\Soumo\\OneDrive\\Desktop\\sign language\\sign language\\sign language\\sign_language_labels.p'

try:
    with open(MODEL_FILENAME, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']

    with open(LABELS_FILENAME, 'rb') as f:
        label_encoder = pickle.load(f)

except FileNotFoundError:
    print("Error: The model or labels file could not be found.")
    print(f"Please ensure '{MODEL_FILENAME}' and '{LABELS_FILENAME}' are in the same directory.")
    exit()

# --- Webcam and Hand Tracking Setup ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# --- Create a full-screen window ---
window_name = 'Sign Language Detection'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Starting real-time prediction... Press 'q' to quit.")

last_prediction = None
last_action_time = 0
cooldown_sec = 2
sentence = ""

# --- Variables for logic ---
last_hand_time = time.time()
no_hand_space_cooldown = 4  
consecutive_count = 0
consecutive_target = 30  
last_confirmed_char = ""
probability = 0.0
hand_present = False

# --- For smoothing the frame ---
prev_frame = None
alpha = 0.5  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # --- Smooth the frame using exponential moving average ---
    if prev_frame is None:
        smooth_frame = frame.copy()
    else:
        smooth_frame = cv2.addWeighted(frame, alpha, prev_frame, 1 - alpha, 0)
    prev_frame = smooth_frame.copy()

    frame_rgb = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    feature_vector = []
    predicted_char = ""
    probability = 0.0
    hand_present = False

    if results.multi_hand_landmarks:
        hand_present = True
        last_hand_time = time.time()
        hand_landmarks_list = results.multi_hand_landmarks

        for h in range(2):
            if h < len(hand_landmarks_list):
                hand_landmarks = hand_landmarks_list[h]
                mp_drawing.draw_landmarks(smooth_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                for lm in hand_landmarks.landmark:
                    feature_vector.append(lm.x - wrist_landmark.x)
                    feature_vector.append(lm.y - wrist_landmark.y)
            else:
                feature_vector.extend([0.0] * 42)
    else:
        feature_vector.extend([0.0] * 84)

    if len(feature_vector) == 84 and hand_present:
        prediction = model.predict(np.asarray([feature_vector]))
        predicted_class_index = np.argmax(prediction)
        predicted_char = label_encoder.inverse_transform([predicted_class_index])[0]
        probability = float(np.max(prediction)) * 100 

        if predicted_char == last_confirmed_char:
            consecutive_count += 1
        else:
            consecutive_count = 1
            last_confirmed_char = predicted_char

        # Logic to add character to sentence based on confidence and cooldown
        if consecutive_count >= consecutive_target and predicted_char != "SPACE":
            if (predicted_char != last_prediction) or ((time.time() - last_action_time) > cooldown_sec):
                print(f'Prediction: {predicted_char} ({probability:.1f}%)')
                last_prediction = predicted_char
                last_action_time = time.time()
                sentence += predicted_char
                consecutive_count = 0  
        elif predicted_char == "SPACE":
            if (predicted_char != last_prediction) or ((time.time() - last_action_time) > cooldown_sec):
                sentence += " "
                last_prediction = predicted_char
                last_action_time = time.time()
                consecutive_count = 0

    else:
        # If no hand detected for 4 seconds, add a space
        if time.time() - last_hand_time > no_hand_space_cooldown:
            if not sentence.endswith(" "):
                sentence += " "
                print("No hand detected for 4 seconds, adding space.")
            last_hand_time = time.time() 

        if time.time() - last_action_time > cooldown_sec:
            last_prediction = None
            consecutive_count = 0

    # --- UI Rendering ---
    h, w, _ = smooth_frame.shape
    text_y_offset = int(h * 0.05)
    word_y_offset = int(h * 0.1)
    prob_y_offset = int(h * 0.15)

    cv2.rectangle(smooth_frame, (0, 0), (w, int(h*0.18)), (0, 0, 0), -1)
    cv2.putText(smooth_frame, f'Letter: {predicted_char}', (20, text_y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(smooth_frame, f'Word: {sentence}', (20, word_y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(smooth_frame, f'Prob: {probability:.1f}%', (20, prob_y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, smooth_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nFinal word:", sentence.strip())