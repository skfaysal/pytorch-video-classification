import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
import joblib
import argparse
from glob import glob

def train_mlp_classifier(train_folder, hidden_layer_sizes, 
                         random_state, save_model_path,class_folders):
    train_data = []
    train_labels = []

    # Load train data
    for class_folder in class_folders:
        class_path = os.path.join(train_folder, class_folder)
        video_folders = os.listdir(class_path)

        for video_folder in video_folders:
            video_path = os.path.join(class_path, video_folder)
            feature_file = os.path.join(video_path, "VST_feature.npy")

            # Load video features
            features = np.load(feature_file)

            # Add features to the train data list
            train_data.append(features)

            # Add label for the class
            train_labels.append(class_folder)

    # Convert the train data and labels to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Flatten the feature vectors if needed
    train_data = train_data.reshape(train_data.shape[0], -1)

    # Initialize the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)

    # Train the MLP classifier
    mlp.fit(train_data, train_labels)

    # Save the trained model
    joblib.dump(mlp, save_model_path)
    print("Model saved successfully at", save_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP Classifier Training')
    parser.add_argument('--train_folder', type=str, default='/root/DataPrep/new/Action-Recognition/train', help='Path to the folder containing training data')
    parser.add_argument('--hidden_layer_sizes', type=int, default=300, help='Hidden layer sizes as a tuple (e.g., (300,) for a single hidden layer with 300 units)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--save_model_path', type=str, default='/root/DataPrep/new/Action-Recognition/gesture_ann.pkl', help='Path to save the trained model')
    args = parser.parse_args()

    class_folders = [i.split('/')[-1] for i in glob(args.train_folder+'/*')]
    print(class_folders)


    # Parse hidden layer sizes as a tuple
    # hidden_layer_sizes = tuple(map(int, args.hidden_layer_sizes.strip('()').split(',')))

    # Train the MLP classifier
    train_mlp_classifier(args.train_folder, args.hidden_layer_sizes,
                          args.random_state, args.save_model_path,class_folders)
