import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sounddevice as sd
import librosa
import keyboard  # For handling keyboard events
import threading

# Define the model class
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.fc1 = nn.Linear(20, 64)  # Input features size should match the dataset (20 in this case)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer with 2 classes (male, female)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the dataset
df = pd.read_csv("voice_train.csv")

# Preprocessing
X = df.iloc[:, 1:-1].values  # Exclude the first column (Id) and the last column (label)
y = df.iloc[:, -1].values    # Last column (label)

# Encode labels ("male" -> 0, "female" -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initialize the model, loss function, and optimizer
model = GenderClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    output = model(X_test)
    _, predicted = torch.max(output.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Accuracy: {accuracy*100:.2f}%')

# Function to extract features from audio data
def extract_features(audio, sample_rate):
    # Compute MFCCs for feature extraction
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mean_mfccs = np.mean(mfccs, axis=1)
    return mean_mfccs

def predict_gender_real_time(audio, sample_rate):
    features = extract_features(audio, sample_rate)
    features = scaler.transform([features])  # Standardize the features
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output.data, 1)
    return "Male" if predicted.item() == 0 else "Female"

# Global variable to manage recording status
recording = True

def record_and_predict(duration=5, sample_rate=22050):
    global recording
    audio = []

    def key_listener():
        global recording
        while recording:
            if keyboard.is_pressed('q'):
                recording = False
                print("Stopping recording...")

    # Start a thread to listen for the 'q' key press
    listener_thread = threading.Thread(target=key_listener)
    listener_thread.start()
    
    print("Recording...")
    with sd.InputStream(callback=lambda indata, frames, time, status: audio.append(indata.flatten()), channels=1, samplerate=sample_rate):
        while recording:
            sd.sleep(100)  # Sleep for a short period to avoid busy waiting

    audio = np.concatenate(audio)
    print("Finished recording.")
    return predict_gender_real_time(audio, sample_rate)

# Example usage
print(record_and_predict())