import cv2
import matplotlib.pyplot as plt
from features import compute_feature_matches
import pprint

# Change dataset index here (1-9)
dataset = 2
if dataset not in range(1, 10):
    raise IndexError(f"Dataset {dataset} is out of range.")

print(f"Dataset {dataset} is selected.")

# Extract features and do the 2D points matching.
# Use BFMathcher instead of FlannBasedMatcher - BF gives better results.
imgs, keypoints, matches = compute_feature_matches(dataset)

# # Check 2D pts extraction
# pprint.pprint(matches)
