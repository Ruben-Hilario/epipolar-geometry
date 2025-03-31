import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_points(points):
    """Normalize points to have zero mean and mean distance of sqrt(2) from origin"""
    points_inhom = points[:, :2] / points[:, 2, np.newaxis]
    centroid = np.mean(points_inhom, axis=0)
    diff = points_inhom - centroid
    mean_dist = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale*centroid[0]],
        [0, scale, -scale*centroid[1]],
        [0, 0, 1]
    ])
    normalized_points = T @ points.T
    return normalized_points.T, T

def compute_fundamental_matrix(points1, points2):
    """Compute fundamental matrix using normalized 8-point algorithm"""
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)
    
    N = norm_points1.shape[0]
    W = np.zeros((N, 9))
    for i in range(N):
        x1, y1, _ = norm_points1[i]
        x2, y2, _ = norm_points2[i]
        W[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    
    _, _, V = np.linalg.svd(W)
    F = V[-1].reshape(3, 3)
    
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V
    
    F = T2.T @ F @ T1
    return F

def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=20):
    """Improved visualization showing both images with points and epipolar lines"""
    # Convert to color for drawing
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Randomly select points to display (to avoid clutter)
    indices = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)
    pts1_selected = pts1[indices]
    pts2_selected = pts2[indices]
    
    # Draw points and epipolar lines on image 1
    for pt in pts2_selected:
        # Compute epipolar line in first image
        line = F.T @ np.array([pt[0], pt[1], 1])
        a, b, c = line
        # Draw the line across the image
        x0, y0 = 0, int(-c/b)
        x1, y1 = img1.shape[1], int(-(c + a*img1.shape[1])/b)
        cv2.line(img1_color, (x0, y0), (x1, y1), (0, 255, 255), 1)
    
    # Draw points on image 1 (in different color)
    for pt in pts1_selected:
        cv2.circle(img1_color, tuple(map(int, pt)), 5, (255, 0, 0), -1)
    
    # Draw points and epipolar lines on image 2
    for pt in pts1_selected:
        # Compute epipolar line in second image
        line = F @ np.array([pt[0], pt[1], 1])
        a, b, c = line
        # Draw the line across the image
        x0, y0 = 0, int(-c/b)
        x1, y1 = img2.shape[1], int(-(c + a*img2.shape[1])/b)
        cv2.line(img2_color, (x0, y0), (x1, y1), (0, 255, 255), 1)
    
    # Draw points on image 2 (in different color)
    for pt in pts2_selected:
        cv2.circle(img2_color, tuple(map(int, pt)), 5, (255, 0, 0), -1)
    
    # Display results
    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    plt.title('IMG_3098.jpg (Points with Epipolar Lines from IMG_3099.jpg)')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
    plt.title('IMG_3099.jpg (Points with Epipolar Lines from IMG_3098.jpg)')
    plt.tight_layout()
    plt.show()

def eight_point_algorithm(img1, img2):
    """Main function implementing the complete pipeline"""
    # ORB feature detection
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Feature matching with ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Convert to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    
    # Compute fundamental matrix
    F = compute_fundamental_matrix(pts1_h, pts2_h)
    
    # Optional: Draw all matches for reference
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title('All Feature Matches'), plt.show()
    
    return F, pts1, pts2


# Main execution
if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('IMG_3098.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('IMG_3099.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images! Check file paths.")
    
    # Compute fundamental matrix and points
    F, pts1, pts2 = eight_point_algorithm(img1, img2)
    
    print("Fundamental Matrix:")
    print(F)
    
    # Draw epipolar lines visualization
    draw_epipolar_lines(img1, img2, pts1, pts2, F)
