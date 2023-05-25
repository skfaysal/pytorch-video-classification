import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import joblib
import csv
from glob import glob

def load_test_data(test_folder, class_folders):
    test_data = []
    test_labels = []

    # Load test data
    for class_folder in class_folders:
        class_path = os.path.join(test_folder, class_folder)
        video_folders = os.listdir(class_path)

        for video_folder in video_folders:
            video_path = os.path.join(class_path, video_folder)
            feature_file = os.path.join(video_path, "VST_feature.npy")

            # Load video features
            features = np.load(feature_file)

            # Add features to the test data list
            test_data.append(features)

            # Add label for the class
            test_labels.append(class_folder)

    # Convert the test data and labels to numpy arrays
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Flatten the feature vectors if needed
    test_data = test_data.reshape(test_data.shape[0], -1)

    return test_data, test_labels

def evaluate_model(model_path, test_folder, class_folders):
    # Load the MLP model
    mlp = joblib.load(model_path)

    # Load the test data
    test_data, test_labels = load_test_data(test_folder, class_folders)

    # Perform inference on the test data
    test_predictions = mlp.predict(test_data)

    # Calculate the accuracy of the classifier on the test data
    accuracy = accuracy_score(test_labels, test_predictions)

    # Calculate the confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)

    # Calculate the F1 score
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    # Calculate the recall
    recall = recall_score(test_labels, test_predictions, average='weighted')

    # Calculate the precision
    precision = precision_score(test_labels, test_predictions, average='weighted')

    return accuracy, cm, f1, recall, precision

def save_results_to_csv(filename, accuracy, cm, f1, recall, precision):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Accuracy', accuracy])
        writer.writerow(['F1 Score', f1])
        writer.writerow(['Recall', recall])
        writer.writerow(['Precision', precision])
        writer.writerow([])  # Add an empty row
        writer.writerow(['Confusion Matrix'])
        for row in cm:
            writer.writerow(row)

if __name__ == '__main__':
    model_path = '/root/DataPrep/new/Action-Recognition/gesture_ann.pkl'
    # test_folder = '/root/CLIP/data/test_chunk_horse_race'
    test_folder ='/root/DataPrep/new/Action-Recognition/test'
    class_folders = [i.split('/')[-1] for i in glob(test_folder+'/*')]
    print(class_folders)

    # Evaluate the model
    accuracy, cm, f1, recall, precision = evaluate_model(model_path, test_folder, class_folders)

    # Save the results to a CSV file
    results_filename = 'evaluation_results_benchmark.csv'
    save_results_to_csv(results_filename, accuracy, cm, f1, recall, precision)

    # Print the evaluation metrics
    print("Accuracy on test set:", accuracy)
    print("F1 Score:", f1)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Confusion Matrix:")
    print(cm)
