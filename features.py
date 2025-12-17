import numpy as np
import cv2
from project_helpers import get_dataset_info

def compute_feature_matches(dataset):
    """
    - detect features for all images in the dataset
    - match features sequentially and with the initial one
    """
    # Load dataset info
    K, img_names, init_pair, _ = get_dataset_info(dataset)

    # Read RGB images
    imgs = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in img_names]

    # Feature detector for each image
    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10, nOctaveLayers=3)

    keypoints   = []
    descriptors = []
    flag = 1

    for img in imgs:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
        print(f"Capture {des.shape[0]} features in image {flag}.")
        flag += 1

    # Feature matching
    pairs = []
    pairs.append(tuple(init_pair))  # add initial pair at the first place

    # Sequential pairs
    for i in range(len(imgs) - 1):
        p = ((i, i+1))
        if p != tuple(init_pair):
            pairs.append(p)

    bf = cv2.BFMatcher(cv2.NORM_L2)    # BFMatcher with L2-Norm
    flann = cv2.FlannBasedMatcher()    # FlannBaseMatcher with default paras
    matches = {}
    for (i, j) in pairs:
        print(f"\n---------- Pair ({i+1}, {j+1}) ----------")
        # Use KNN-like system to find the most similar descriptions as in A4
        knn = bf.knnMatch(descriptors[i], descriptors[j], k=2)
        print(f"There are {len(knn)} matches be found in total.")

        good = []
        # Apply ratio test, ratio_threshold=0.75
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        matches[(i, j)] = good
        print(f"After ratio-test, {len(good)} good matches are left.")

    return imgs, keypoints, matches
