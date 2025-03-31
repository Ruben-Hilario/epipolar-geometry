import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('IMG_3098.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('IMG_3099.jpg', cv2.IMREAD_GRAYSCALE)
assert img1 is not None and img2 is not None, "Check image paths!"

# Initialize ORB with more features
orb = cv2.ORB_create(nfeatures=5000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Ensure descriptors exist
if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
    raise ValueError("Not enough keypoints detected!")

# BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:  # More permissive ratio
        good_matches.append(m)

print("Number of good matches:", len(good_matches))
if len(good_matches) < 8:
    raise ValueError("Need at least 8 matches for F matrix computation!")

# Extract points
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute F with relaxed RANSAC threshold
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0)
if F is None:
    raise ValueError("Fundamental matrix computation failed!")

# Draw matches (for debugging)
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.imshow(matched_img), plt.title("Matches"), plt.show()

# Proceed with epipolar lines...
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=10):
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    indices = np.random.choice(len(pts1), num_lines, replace=False)
    pts1_selected = pts1[indices]
    pts2_selected = pts2[indices]
    for pt in pts1_selected:
        cv2.circle(img1_color, tuple(pt[0].astype(int)), 5, (0, 255, 0), -1)
    lines2 = cv2.computeCorrespondEpilines(pts1_selected, 1, F)
    lines2 = lines2.reshape(-1, 3)
    for line, pt in zip(lines2, pts2_selected):
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0] * img2.shape[1])/line[1]])
        cv2.line(img2_color, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(img2_color, tuple(pt[0].astype(int)), 5, (255, 0, 0), -1)
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(img1_color), plt.title('Image 1 (Points)')
    plt.subplot(122), plt.imshow(img2_color), plt.title('Image 2 (Epipolar Lines)')
    plt.show()

draw_epipolar_lines(img1, img2, pts1, pts2, F)