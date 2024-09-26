import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix, 
                             average_precision_score)
from collections import Counter

# Ensure only GPU device 1 is visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load test filenames
with open('/all_test_filename_global_both_both_fixationonly_true.json') as f:
    all_test_filename = json.load(f)

# Prepare paths
all_paths = [x.split('__A__')[0] for single_path in all_test_filename for x in single_path]

# Load predicted gazes
global_json = []



# Initialize trajectories dictionary
trajectories = {}
for name, element in zip(all_test_filename, global_json):
    frame_name = name[0]
    gt_trajectory = [[max(0, int(point['x'])), max(0, int(point['y']))] for point in element[0]['GT']]
    global_trajectory = [[int(point['x']), int(point['y'])] for point in element[0]['global']]
    global_heatmap = [[point['heatmap']] for point in element[0]['global']]
    error_list = [99] * 8
    action_list = [4] * 8
    trajectories[frame_name] = {
        'gt': gt_trajectory,
        'global_trajectory': global_trajectory,
        'global_heatmap': global_heatmap,
        'error_list': error_list,
        'action_list': action_list
    }

# Function to calculate similarity using heatmap
def calculate_similarity_with_heatmap(gt_trajectory, predicted_trajectory, top_k=1):
    similarity_score = 0.0
    for i in range(len(gt_trajectory)):
        gt_point = np.array(gt_trajectory[i])
        predicted_heatmap = predicted_trajectory[i, 0]
        top_k_indices = np.unravel_index(np.argsort(predicted_heatmap, axis=None)[-top_k:], predicted_heatmap.shape)
        top_k_indices_array = np.column_stack(top_k_indices)[::-1]
        top_k_indices_array = (top_k_indices_array / 64) * 256
        distances = np.linalg.norm(gt_point - top_k_indices_array, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        weights = [1, 0.0, 0.00, 0.00, 0.00, 0.00]
        weighted_distance = np.sum(weights[:len(sorted_distances)] * sorted_distances)
        similarity_score += weighted_distance

    return similarity_score / len(gt_trajectory)

# Function to classify trajectory
def classify_trajectory(dtw_score, threshold=0.5):
    return 0 if dtw_score > threshold else 1

# Prepare to collect scores and ground truth
GT = []
predictions = []
scores = []
names = []

# Calculate scores for each trajectory
for element in trajectories:
    gt = trajectories[element]['gt']
    pred = trajectories[element]['global_trajectory']
    heatmap = np.array(trajectories[element]['global_heatmap'])
    dtw_score = calculate_similarity_with_heatmap(gt, heatmap)
    modified_list = [0 if num == 99 else 1 for num in trajectories[element]['error_list']]
    if modified_list.count(modified_list[0]) == len(modified_list):
        names.append(element)
        scores.append(dtw_score)

# Collect ground truth
c = 0
for element in trajectories:
    gt = trajectories[element]['gt']
    pred = trajectories[element]['global_trajectory']
    modified_list = [0 if num == 99 else 1 for num in trajectories[element]['error_list']]
    if modified_list.count(modified_list[0]) == len(modified_list):
        GT.append(int(modified_list[0]))
        classification = classify_trajectory(scores[c])
        predictions.append(classification)
        c += 1

# Normalize scores for submission
def normalize_array(arr):
    min_val = min(arr)
    max_val = max(arr)
    return [(x - min_val) / (max_val - min_val) for x in arr]

normalized_arr = normalize_array(scores)
to_save_scores = [1 if score > 0.90 else 0 for score in normalized_arr]

submission = {"modality": "RGBE"}
for frame_name, pred in zip(names, to_save_scores):
    submission[frame_name] = pred

# Prepare real submission with challenge file list
real_submission = {"modality": "RGBE"}
with open("/challenge_file_list.txt", "r") as myfile:
    challenge_file_list = myfile.read().splitlines()
    
for element in challenge_file_list:
    real_submission[element] = submission.get(element, 0)

with open('/submission.json', 'w') as outfile:
    json.dump(real_submission, outfile)

# Calculate and print evaluation metrics
GT = np.logical_not(np.array(GT)).astype(int)
pred = np.round(predictions).astype(int)

precision = precision_score(GT, pred)
recall = recall_score(GT, pred)
f1 = f1_score(GT, pred)
accuracy = accuracy_score(GT, pred)

# Calculate confusion matrix
cm = confusion_matrix(GT, pred)
tn, fp, fn, tp = cm.ravel()
total_instances = len(GT)
accuracy = (tp + tn) / total_instances
average_precision = average_precision_score(GT, pred)

# Output metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Average Precision: {average_precision:.4f}')
