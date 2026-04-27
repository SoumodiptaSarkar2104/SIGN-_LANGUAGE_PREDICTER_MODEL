import os
import cv2

# Path to your sign language dataset
DATA_DIR = 'C:/Users/Soumo/OneDrive/Desktop/sign language/sign language/sign language/sign_language_dataset'

# Automatically detect all sign folders (A-Z or whatever exists)
signs = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]
print(f'Signs detected: {signs}')

# Data storage
data = []
labels = []

for sign in signs:
    sign_dir = os.path.join(DATA_DIR, sign)
    for img_name in os.listdir(sign_dir):
        img_path = os.path.join(sign_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'Skipping unreadable image: {img_path}')
            continue
        data.append(img)
        labels.append(sign)

print(f'Total images loaded: {len(data)}')
print(f'Total labels loaded: {len(labels)}')
