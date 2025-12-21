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

    # Read images and transfer to gray images
    imgs = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in img_names]

    # Feature detector for each image
    # Initiate SIFT detector (same para. as in the Assignment 4)
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
        pts1_px = []
        pts2_px = []

        # Apply ratio test, ratio_threshold=0.75
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                pts1_px.append(keypoints[i][m.queryIdx].pt)
                pts2_px.append(keypoints[j][m.trainIdx].pt)

        # Transfer to numpy array - (2,N)
        pts1_px = np.array(pts1_px).T
        pts2_px = np.array(pts2_px).T

        # Homogeneous coord.
        pts1_homo = np.vstack([pts1_px, np.ones(pts1_px.shape[1])])
        pts2_homo = np.vstack([pts2_px, np.ones(pts2_px.shape[1])])

        # Normalize
        pts1_norm = np.linalg.inv(K) @ pts1_homo
        pts2_norm = np.linalg.inv(K) @ pts2_homo

        matches [(i, j)] = {
            'pts1_norm': pts1_norm,
            'pts2_norm': pts2_norm
        }

        print(f"After ratio-test, {len(good)} good matches are left.")



    return imgs, keypoints, matches
