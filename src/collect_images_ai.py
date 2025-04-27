import os #for working with directories
import cv2 #computer vision library. It is for working with the pc camera and all that.

DATA_DIR = './data' #Sets the root dataset folder as ./data
if not os.path.exists(DATA_DIR): #If it doesn’t already exist, the script creates it
    os.makedirs(DATA_DIR)

number_of_classes = 26 #26 letters of the alphabet (classes are currently indexed numerically, so that is why we had to rename the folders later)
dataset_size = 100 #takes 1000 frames from each video

cap = cv2.VideoCapture(0) #initialise the webcam 0
if not cap.isOpened(): #If it fails, the script exits with an error message
    print("❌ Error: Could not access the camera.")
    exit()

for j in range(number_of_classes): #loops through each class
    if not os.path.exists(os.path.join(DATA_DIR, str(j))): #creates class folder for each class if it does not exist already
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read() #returns the actual frame and a boolean stating if the image has been correctly captured (ret)
        cv2.putText(frame, 'Ready? Press "Q"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA) #just some settings for the text displayed on the webcam
        cv2.imshow('frame', frame) #displays the webcam in a window called frame
        if cv2.waitKey(25) == ord('q'):
            break #If you press the Q key, it exits the preview loop (break) and starts capturing images.

    counter = 0 #intiliases a class counter
    while counter < dataset_size: #while the counter is smaller to the dataset size...
        ret, frame = cap.read() #captures a new image from the webcam
        cv2.imshow('frame', frame) #displayes the current frame
        cv2.waitKey(25) #keeps a 25 milisecond frame rate
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) #saves the current frame as an image in the specified directory

        counter += 1 #adds 1 class to the counter

cap.release() #releases the webcam so its not locked or in use anymore
cv2.destroyAllWindows() #closes any opencv windows that were opened with cv2.imshow().
