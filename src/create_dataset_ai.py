import os
import pickle
import mediapipe as mp
import cv2
import string

mp_hands = mp.solutions.hands #Loads the Hands module from MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#static_image_mode set to true means that we are processing photos, not videos.
# The detecttion confidence accepts predicitions that have 30% confidence of being a hand.

DATA_DIR = './data' #the folder containing all of our images organized into letter-named folders IS NOT IN THE REPOSITORY.
data = [] #will store the extracted hand landmark vectors
labels = [] #will store the matchign letter for each vector (puede que ya no haga falta, ya que hemos cambiado los nombres de la carpeta con la terminal)

# Only process A-Z folders
valid_labels = set(string.ascii_uppercase) # sets the valid labels, which are uppercase letters

for dir_ in os.listdir(DATA_DIR): #ensures only valid folders are processed by filtering out the rest
    if dir_ not in valid_labels:
        print(f" Skipping non-label folder: {dir_}")
        continue

    folder_path = os.path.join(DATA_DIR, dir_)

    for img_path in os.listdir(folder_path): #loops over all images
        data_aux = [] #temporary list for storing this image’s 42 landmark values
        x_ = []
        y_ = [] #used to store all x and y coords to normalize them later

        #If the image is broken or missing, skip it
        img = cv2.imread(os.path.join(folder_path, img_path))
        if img is None:
            print(f" Skipped unreadable image: {img_path} in {dir_}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV loads images as BGR, but MediaPipe expects RGB
        results = hands.process(img_rgb) #runs the hand detector + landmark predictor

        #Extract Landmarks If Hand Is Found
        if results.multi_hand_landmarks: #if a hand is detected
            hand_landmarks = results.multi_hand_landmarks[0]  # use first detected hand only

            for landmark in hand_landmarks.landmark: #store all 21 landmark x and y coordinates into x_ and y_
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark: #normalise the landmarks
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            if len(data_aux) == 42: #Each hand should yield 21 (x, y) pairs = 42 values If yes: add to data and match it with the current folder label ('A', 'B', etc.)
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"⚠️ Skipped malformed sample: {img_path} in {dir_} — only {len(data_aux)} features")
        else:
            print(f"⚠️ No hand detected in: {img_path} in {dir_}")

# Save final dataset to a file
with open('/Users/jacobogalindosanz/Desktop/hand-gesture-recognition/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Finished processing. Saved {len(data)} valid samples.")
