import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load model and label encoder
model = load_model('deep_model.keras')
with open('label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)

# MediaPipe setup
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Sentence builder
sentence = ""
predicted_character = ""

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            input_data = np.array(data_aux).reshape(1, -1)
            prediction = model.predict(input_data)
            predicted_character = le.inverse_transform([np.argmax(prediction)])[0]

            # Draw bounding box and character
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
        else:
            predicted_character = ""
    else:
        predicted_character = ""

    # Show sentence bar
    cv2.rectangle(frame, (20, H - 60), (W - 20, H - 20), (255, 255, 255), -1)
    cv2.putText(frame, sentence, (30, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):  # Quit
        break
    elif key == 13:  # Enter key
        if predicted_character:
            sentence += predicted_character
    elif key == ord(' '):  # Space key
        sentence += " "
    elif key == 127:  # Delete key
        sentence = sentence[:-1]
    elif key == ord('c'):  # Clear sentence
        sentence = ""
    elif key == ord('v'):  # Speak the whole sentence
        if sentence.strip() != "":
            engine.say(sentence)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
