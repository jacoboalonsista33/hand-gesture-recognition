import os
import cv2

DATA_DIR = './data'  #the folder containing all of our images organized into letter-named folders IS NOT IN THE REPOSITORY.

for folder_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Skipping {file_name}, could not read.")
            continue

        # Flip image horizontally
        img_flipped = cv2.flip(img, 1)

        # Save it with a new name
        new_filename = f"left_{file_name}"
        new_file_path = os.path.join(folder_path, new_filename)
        cv2.imwrite(new_file_path, img_flipped)

print("✅ Left-handed image augmentation completed.")
