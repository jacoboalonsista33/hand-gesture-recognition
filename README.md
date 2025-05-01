# Hand Gesture Recognition using Deep Learning and MediaPipe

This project trains a deep learning model to recognize American Sign Language (A-Z) gestures using MediaPipe hand landmark detection.  
It provides a complete pipeline — from data collection to model training and live testing — with preprocessed data ready for immediate use.

- Pre-processed dataset included (`data/data.pickle`)
- Fully trainable neural network model
- Live real-time gesture prediction using webcam

This project includes a preprocessed dataset (data.pickle) containing hand landmark features and labels extracted from sign language images. The full raw image dataset used to generate this file is not included in the repository due to size constraints. 
However, the code scripts for collecting, augmenting, and processing the raw images are provided only for reference in the src/ folder. 

---

## Project Structure

- `src/` — Source code:
  - `collect_images_ai.py` — Script to collect training images via webcam (optional, since dataset is already created and augmented in data.pickle).
  - `create_dataset_ai.py` — Process collected images to extract hand landmarks. (optional, since dataset is already created and augmented in data.pickle).
  - `augment_dataset_ai.py` — Perform data augmentation by flipping images (optional, since dataset is already created and augmented in data.pickle).
  - `train_classifier_ai.py` — Train a deep learning classifier using preprocessed dataset.
  - `test_classifier_ai.py` — Live webcam testing of the trained model.
- `models/` — Trained model and label encoder:
  - `deep_model.keras`
  - `label_encoder.pickle`
- `data/` — Processed dataset:
  - `data.pickle`
- `requirements.txt` — List of Python dependencies.

---

## Quick Start

### 1. Install required libraries
Install all necessary Python packages listed in `requirements.txt`:

```
bash
pip install -r requirements.txt
```

### 2. Train the model
Train the model on the provided preprocessed dataset:
```
python src/train_classifier_ai.py
```

### 3. Test the model live
After training, test the model in real-time using your webcam:
```
python src/test_classifier_ai.py
```

### 4. Controls during testing
While testing with the webcam, use the following keyboard controls:

Enter — Add the predicted letter to the sentence.
Space — Add a space between words.
Backspace — Delete the last character.
C — Clear the full sentence.
Q — Quit the webcam session.

The recognized text will be updated live on the video feed.










