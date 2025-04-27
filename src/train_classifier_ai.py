import matplotlib
matplotlib.use('TkAgg')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


# Load data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, stratify=labels_encoded)

# Define a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Save
model.save('deep_model.h5')

def evaluate_model(model, x_test, y_test, label_encoder, history=None):
    print("Evaluating model...\n")

    # 1. Accuracy, precision, recall, F1
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 3. Loss and Accuracy curves (overfitting check)
    if history:
        plt.figure(figsize=(14, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("\n📊 If validation loss increases while training loss decreases, you might be overfitting.")

# Save
model.save('deep_model.keras')

# Save the label encoder
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

# Evaluate
evaluate_model(model, x_test, y_test, le, history)

