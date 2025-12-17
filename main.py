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

# # Mathing Visualization
# for (i,j), m in matches.items():
#
#     img_vis = cv2.drawMatches(
#         imgs[i], keypoints[i],
#         imgs[j], keypoints[j],
#         m[:100], None,
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )
#
#     plt.figure(figsize=(12,6))
#     plt.title(f"Dataset {dataset} - Pair ({i},{j})")
#     plt.imshow(img_vis, cmap='gray')
#     plt.axis("off")
#     plt.show()
