import os
import cv2

DATA_DIR = 'D:/new sign language MODEL/sign language/sign_language_dataset'

signs = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]
print(f'Signs detected: {signs}')

data = []
labels = []

# Target frame size to reduce memory load
target_size = (128, 128)
frames_per_video = 5

for sign in signs:
    sign_dir = os.path.join(DATA_DIR, sign)
    for file_name in os.listdir(sign_dir):
        if file_name.lower().endswith('.mov'):
            vid_path = os.path.join(sign_dir, file_name)
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video file {vid_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                print(f"No frames detected in {vid_path}, skipping.")
                cap.release()
                continue

            try:
                for i in range(frames_per_video):
                    frame_number = int(i * frame_count / frames_per_video)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"Skipping unreadable frame {frame_number} in {vid_path}")
                        continue

                    # Resize to reduce memory usage
                    frame_resized = cv2.resize(frame, target_size)
                    data.append(frame_resized)
                    labels.append(sign)

            except Exception as e:
                print(f"Error processing {vid_path}: {e}")

            cap.release()

print(f'Total frames loaded: {len(data)}')
print(f'Total labels loaded: {len(labels)}')
