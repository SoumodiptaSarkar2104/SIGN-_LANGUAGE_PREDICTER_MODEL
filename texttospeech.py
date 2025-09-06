import pyttsx3

engine = pyttsx3.init()

with open("D:/new sign language MODEL/sign language/speak.txt", "r", encoding="utf-8") as file:
    text = file.read()           # Read the entire file contents

engine.say(text)                 # Pass the text contents to speak
engine.runAndWait()             # Run TTS and wait until done

# Clear the file after speaking
with open("D:/new sign language MODEL/sign language/speak.txt", "w", encoding="utf-8") as file:
    file.write("")
