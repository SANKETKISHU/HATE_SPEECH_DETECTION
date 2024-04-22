import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import speech_recognition as sr

# Custom standardization function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Vectorization layer
max_features = 10000
sequence_length = 250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization, max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Load the dataset
df = pd.read_csv('hate_speech.csv')

# Balancing the dataset
class_2 = df[df['class'] == 2]
class_1 = df[df['class'] == 1].sample(n=3500)
class_0 = df[df['class'] == 0]
balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

# Splitting the dataset into training and validation sets
features = balanced_df['tweet']
target = balanced_df['class']
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=22)

# One-hot encoding the target variable
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

# Training the tokenizer
max_words = 5000
token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(X_train)

# Generating token embeddings
training_seq = token.texts_to_sequences(X_train)
training_pad = pad_sequences(training_seq, maxlen=50, padding='post', truncating='post')

testing_seq = token.texts_to_sequences(X_val)
testing_pad = pad_sequences(testing_seq, maxlen=50, padding='post', truncating='post')

# Building the model with custom layer
embedding = 16
num_classes = 3

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding, input_shape=(sequence_length,)),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax'),
])
model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=2, factor=0.5, verbose=0)

# Training the model
history = model.fit(training_pad, Y_train, epochs=10, batch_size=32, validation_data=(testing_pad, Y_val), callbacks=[early_stopping, reduce_lr])

# Function to predict hate speech level
def predict_hate_speech():
    input_text = text_entry.get()
    input_seq = token.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=50, padding='post', truncating='post')
    input_pad = input_pad.reshape(1, -1) # Reshape to (1, 50)
    prediction = model.predict(input_pad)
    hate_speech_prob = prediction[0][0]
    offensive_prob = prediction[0][1]
    neither_prob = prediction[0][2]
    hate_speech_label.config(text=f'Hate Speech: {hate_speech_prob}')
    offensive_label.config(text=f'Offensive Language: {offensive_prob}')
    neither_label.config(text=f'Neither: {neither_prob}')
    # Plotting the probabilities
    plt.bar(['Hate Speech', 'Offensive Language', 'Neither'], [hate_speech_prob, offensive_prob, neither_prob])
    plt.title('Probability Distribution')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.show()

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        audio = r.listen(source)
        print("Got it! Now converting it to text...")
    try:
        text = r.recognize_google(audio)
        text_entry.delete(0, tk.END)
        text_entry.insert(0, text)
    except Exception as e:
        print("Error:", e)

# GUI setup
root = tk.Tk()
root.title("Hate Speech Detector")
root.geometry("600x400")

# Header
header_label = ttk.Label(root, text="Enter text to check hate speech level:")
header_label.pack()

# Text Entry
text_entry = ttk.Entry(root, width=50)
text_entry.pack()

# Buttons
predict_button = ttk.Button(root, text="Check Hate Speech Level", command=predict_hate_speech)
predict_button.pack()

speech_to_text_button = ttk.Button(root, text="Speak to Input Text", command=speech_to_text)
speech_to_text_button.pack()

# Hate Speech Level Labels
hate_speech_label = ttk.Label(root, text="")
hate_speech_label.pack()
offensive_label = ttk.Label(root, text="")
offensive_label.pack()
neither_label = ttk.Label(root, text="")
neither_label.pack()

# Quit Button
quit_button = ttk.Button(root, text="Quit", command=root.quit)
quit_button.pack()

root.mainloop()
