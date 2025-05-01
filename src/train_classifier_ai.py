import matplotlib
matplotlib.use('TkAgg')
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split dataset (with stratification)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, stratify=labels_encoded
)

# Check samples per letter
counter = pd.Series(y_test).value_counts().sort_index()
test_samples_df = pd.DataFrame({
    'Letter': le.classes_,
    'Test Samples': counter.values
})
print("\n📊 Test samples per letter:")
print(test_samples_df.to_string(index=False))

# Define a stronger neural network
model = Sequential([
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')
])

# Compile with smaller learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Setup EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=40,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# Save model
model.save('deep_model.h5')
model.save('deep_model.keras')

# Save label encoder
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

# Define evaluation function
def evaluate_model(model, x_test, y_test, label_encoder, history=None):
    print("\nEvaluating model...\n")

    # 1. Classification report
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 3. Loss and accuracy curves
    if history:
        plt.figure(figsize=(14, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

# Evaluate final model
evaluate_model(model, x_test, y_test, le, history)
