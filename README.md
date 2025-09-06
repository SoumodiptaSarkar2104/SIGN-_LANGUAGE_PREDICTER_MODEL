# SIGN-_LANGUAGE_PREDICTER_MODEL

# Sign Language Recognition with Real-Time Text-to-Speech

A real-time sign language recognition system that detects hand gestures using MediaPipe and a trained ML model, converts recognized signs into text, and synthesizes speech using pyttsx3. Designed for accessibility and communication assistance.

## Features

- Real-time hand gesture detection via webcam using MediaPipe
- Machine learning-based recognition of sign language characters
- Continuous logging of predicted signs to a text file
- Text-to-speech synthesis of recognized signs
- Automatic transition between sign detection and speech playback

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/sign-language-tts.git
cd sign-language-tts

2. Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install required dependencies:
pip install -r requirements.txt


4. Ensure you have the model files:
- `sign_language_model.p`
- `sign_language_labels.p`

Place them in the project root or update paths accordingly.

## Usage

- Run the main program:

- The program starts your webcam, detects hand signs, and speaks predictions.
- Remove your hand from the camera for 2 seconds to automatically trigger the text-to-speech playback of logged signs.
- After speech playback finishes, sign recognition resumes automatically.
- Press `ESC` anytime to quit.

## File Structure

- `main.py` - Main application script to run camera detection and TTS flow
- `texttospeech.py` - Script to convert logged text file to speech and clear the log
- `sign_language_model.p` - Pretrained model file for sign recognition
- `sign_language_labels.p` - Label encoder for prediction classes
- `requirements.txt` - Python dependencies

## Dependencies

- Python 3.7+
- OpenCV
- MediaPipe
- numpy
- pyttsx3
- scikit-learn (for model)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this template further to suit your project details and personal style.


