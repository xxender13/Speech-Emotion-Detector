import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("E:\\Speech Emotion Detector\\Dataset\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train, x_test, y_train, y_test = load_data(test_size=0.25)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# DataFlair - Train the model
model.fit(x_train, y_train)
# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

# DataFlair - Create confusion matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# DataFlair - Create classification report
class_report = classification_report(y_true=y_test, y_pred=y_pred)

# DataFlair - Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=observed_emotions, yticklabels=observed_emotions)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# DataFlair - Display classification report
print("Classification Report:\n", class_report)

# DataFlair - Plot bar chart for the distribution of actual emotions
# DataFlair - Plot bar chart for the distribution of actual emotions
plt.figure(figsize=(8, 6))
sns.countplot(y_test, palette='viridis')
plt.title('Distribution of Actual Emotions')
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.show()

# DataFlair - Plot bar chart for the distribution of predicted emotions
plt.figure(figsize=(8, 6))
sns.countplot(y_pred, palette='viridis')
plt.title('Distribution of Predicted Emotions')
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.show()

