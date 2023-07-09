
# Speech Emotion Recognition Using Ensemble Technique

This porject is an implementation of a speech emotion recognition system using machine learning techniques. The system extracts features from audio files, trains a model on the extracted features, and predicts emotions for new audio files.


## Environment 
- Python 3.8
- Google Colab
- SciKit-Learn

## Datasets

- For model training:
#### 1. RAVDESS

English, around 1440 audios from 24 people (12 male and 12 female) including 8 different emotions (the third number of the file name represents the emotional type): 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.

- For model generalization
#### 2. TESS

A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total.
## Installation

To run this code, you need to have the following libraries installed:

- google.colab: Used to mount Google Drive
- librosa: Used for audio feature extraction
- soundfile: Used for reading audio files
- numpy: Used for numerical operations
- scikit-learn: Used for machine learning algorithms and 
preprocessing
- matplotlib: Used for data visualization
- seaborn: Used for creating a confusion matrix
## Documentation

Getting Started:


- Import the required libraries:

        from google.colab import drive
        import librosa
        import soundfile
        import os, glob, pickle
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, 
        VotingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import RFE
        from sklearn.neural_network import MLPClassifier
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

- Mount Google Drive:

        drive.mount('/content/drive')

- Define a function extract_feature to extract MFCC, pitch, and RMS features from an audio file.
- Define a dictionary emotions that maps emotion codes to emotion labels.
- Define a list observed_emotions that contains the emotions to be observed.

- Define a function load_data to load the data and extract features for each sound file.

- Load the data and get the features and emotions respectively:

        x, y = load_data()

- Print the shape of the loaded data and the number of features extracted:

        print(f'Data shape: {x.shape}')
        print(f'Features extracted: {x[0].shape[0]}')

- Visualize the distribution of emotions in the dataset using a bar chart.

- Create a machine learning pipeline using Pipeline from scikit-learn. The pipeline includes feature scaling, feature selection, and a voting classifier with multiple classifiers.

- Fit the model to the training data:
    
        pipeline.fit(x, y)

- Define a function predict_emotion to predict the emotion for a given audio file.

- Set the path to the directory containing the test audio files.

- Create empty lists to store the predicted and true labels.

- Get a list of all the WAV files in the test directory.

- Iterate through each file, predict the emotion, and compare it with the true label.

- Calculate the accuracy score and print the number of correct predictions per class.

- Create a confusion matrix to visualize the performance of the model.