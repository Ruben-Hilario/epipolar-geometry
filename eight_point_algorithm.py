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



def compute_rectification_homographies(F, pts1, pts2, img_shape):
    """Compute homographies H1 and H2 for rectifying image pair"""
    # Compute epipoles
    _, _, V = np.linalg.svd(F)
    e = V[-1]
    e = e / e[2]  # Make homogeneous
    
    _, _, V = np.linalg.svd(F.T)
    e_prime = V[-1]
    e_prime = e_prime / e_prime[2]  # Make homogeneous
    
    # Compute homography H2 that maps e' to infinity (f,0,0)
    width, height = img_shape[1], img_shape[0]
    
    # Translation matrix to move center to origin
    T = np.array([
        [1, 0, -width/2],
        [0, 1, -height/2],
        [0, 0, 1]
    ])
    
    # Translate epipole
    e_prime_T = T @ e_prime
    e_prime_T = e_prime_T / e_prime_T[2]
    
    # Rotation matrix to move epipole to x-axis
    alpha = 1 if e_prime_T[0] >= 0 else -1
    d = np.sqrt(e_prime_T[0]**2 + e_prime_T[1]**2)
    R = np.array([
        [alpha * e_prime_T[0]/d, alpha * e_prime_T[1]/d, 0],
        [-alpha * e_prime_T[1]/d, alpha * e_prime_T[0]/d, 0],
        [0, 0, 1]
    ])
    
    # After rotation, epipole should be at (f,0,1)
    e_prime_R = R @ e_prime_T
    e_prime_R = e_prime_R / e_prime_R[2]
    f = e_prime_R[0]
    
    # Transformation to map (f,0,1) to infinity (f,0,0)
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1/f, 0, 1]
    ])
    
    # Final H2
    H2 = np.linalg.inv(T) @ G @ R @ T
    
    # Compute M matrix (F = [e]x M)
    e_cross = np.array([
        [0, -e[2], e[1]],
        [e[2], 0, -e[0]],
        [-e[1], e[0], 0]
    ])
    M = e_cross @ F + np.outer(e, np.array([1, 1, 1]))
    
    # Compute HA matrix
    p_hat = (H2 @ M) @ pts1.T  # Transform points from image1 using H2*M
    p_prime_hat = H2 @ pts2.T    # Transform points from image2 using H2
    
    # Convert to inhomogeneous coordinates
    p_hat = p_hat[:2] / p_hat[2]
    p_prime_hat = p_prime_hat[:2] / p_prime_hat[2]
    
    # Solve least squares problem for a (HA = [[a1,a2,a3],[0,1,0],[0,0,1]])
    W = np.vstack([p_hat[0], p_hat[1], np.ones(len(pts1))]).T
    b = p_prime_hat[0]
    a = np.linalg.lstsq(W, b, rcond=None)[0]
    
    HA = np.array([
        [a[0], a[1], a[2]],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Final H1
    H1 = HA @ H2 @ M
    
    return H1, H2

def rectify_images(img1, img2, H1, H2):
    """Apply rectification homographies to images"""
    # Get image dimensions
    h, w = img1.shape[:2]
    
    # Compute size of output images
    corners1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # Transform corners
    corners1_transformed = cv2.perspectiveTransform(corners1.reshape(1, -1, 2), H1).reshape(-1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2.reshape(1, -1, 2), H2).reshape(-1, 2)
    
    # Get bounding box for both images
    all_corners = np.vstack([corners1_transformed, corners2_transformed])
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    
    # Compute translation to keep positive coordinates
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    # Adjust homographies with translation
    H1_adj = translation @ H1
    H2_adj = translation @ H2
    
    # Warp images
    size = (x_max - x_min, y_max - y_min)
    img1_rect = cv2.warpPerspective(img1, H1_adj, size)
    img2_rect = cv2.warpPerspective(img2, H2_adj, size)
    
    return img1_rect, img2_rect

def draw_rectified_epipolar_lines(img1_rect, img2_rect, pts1, pts2, H1, H2, F, num_lines=20):
    """Draw epipolar lines on rectified images to verify they're horizontal"""
    # Convert points to homogeneous
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    
    # Transform points using homographies
    pts1_rect = (H1 @ pts1_h.T).T
    pts1_rect = pts1_rect[:, :2] / pts1_rect[:, 2, np.newaxis]
    
    pts2_rect = (H2 @ pts2_h.T).T
    pts2_rect = pts2_rect[:, :2] / pts2_rect[:, 2, np.newaxis]
    
    # Convert to color for drawing
    img1_color = cv2.cvtColor(img1_rect, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2_rect, cv2.COLOR_GRAY2BGR)
    
    # Randomly select points to display
    indices = np.random.choice(len(pts1_rect), min(num_lines, len(pts1_rect)), replace=False)
    pts1_selected = pts1_rect[indices]
    pts2_selected = pts2_rect[indices]
    
    # Draw points and horizontal lines on image 1
    for pt in pts1_selected:
        y = int(pt[1])
        cv2.line(img1_color, (0, y), (img1_rect.shape[1], y), (0, 255, 255), 1)
        cv2.circle(img1_color, tuple(map(int, pt)), 5, (255, 0, 0), -1)
    
    # Draw points and horizontal lines on image 2
    for pt in pts2_selected:
        y = int(pt[1])
        cv2.line(img2_color, (0, y), (img2_rect.shape[1], y), (0, 255, 255), 1)
        cv2.circle(img2_color, tuple(map(int, pt)), 5, (255, 0, 0), -1)
    
    # Display results
    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Image 1 with Horizontal Epipolar Lines')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Image 2 with Horizontal Epipolar Lines')
    plt.tight_layout()
    plt.show()

def safe_rectify(img1, img2, F, pts1, pts2):
    try:
        # Convert points to homogeneous
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
        
        # Compute homographies with checks
        H1, H2 = compute_rectification_homographies(
            F.astype(np.float64),  # Ensure double precision
            pts1_h.astype(np.float64),
            pts2_h.astype(np.float64),
            img1.shape
        )
        
        # Verify homographies
        assert np.all(np.isfinite(H1)), "H1 has infinite values"
        assert np.all(np.isfinite(H2)), "H2 has infinite values"
        
        # Rectify with reduced resolution if needed
        scale = 0.5  # Adjust based on your system
        small_img1 = cv2.resize(img1, (0,0), fx=scale, fy=scale)
        small_img2 = cv2.resize(img2, (0,0), fx=scale, fy=scale)
        
        H1_scale = H1.copy()
        H1_scale[:2,2] *= scale  # Adjust translation
        H2_scale = H2.copy()
        H2_scale[:2,2] *= scale
        
        img1_rect, img2_rect = rectify_images(small_img1, small_img2, H1_scale, H2_scale)
        
        return img1_rect, img2_rect
        
    except Exception as e:
        print(f"Rectification failed: {str(e)}")
        return None, None

def simple_rectify(img1, img2):
    # Example homographies (must be float32 or float64)
    H1 = np.array([[1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, 1]], dtype=np.float32)  # Explicit float32
    
    H2 = np.array([[1, 0, 100], 
                   [0, 1, 0], 
                   [0, 0, 1]], dtype=np.float32)
    
    # Warp images
    h, w = img1.shape
    img1_rect = cv2.warpPerspective(img1, H1, (w, h))
    img2_rect = cv2.warpPerspective(img2, H2, (w, h))
    
    return img1_rect, img2_rect

# Update the main execution to include rectification
if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('IMG_3098.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('IMG_3099.jpg', cv2.IMREAD_GRAYSCALE)
    # Reduce image size if needed (add at start of main)
    img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images! Check file paths.")
    
    # Compute fundamental matrix and points
    F, pts1, pts2 = eight_point_algorithm(img1, img2)
    
    print("Fundamental Matrix:")
    print(F)
    
    # Draw epipolar lines visualization before rectification
    draw_epipolar_lines(img1, img2, pts1, pts2, F)
    
    # Compute rectification homographies
    H1, H2 = compute_rectification_homographies(F, 
                                              np.hstack([pts1, np.ones((len(pts1), 1))]), 
                                              np.hstack([pts2, np.ones((len(pts2), 1))]), 
                                              img1.shape)
    
    # Rectify images
    #img1_rect, img2_rect = simple_rectify(img1, img2 )
    img1_rect, img2_rect = safe_rectify(img1, img2, F, pts1, pts2)
    #img1_rect, img2_rect = rectify_images(img1, img2, H1, H2)
    
    # Draw epipolar lines after rectification (should be horizontal)
    draw_rectified_epipolar_lines(img1_rect, img2_rect, pts1, pts2, H1, H2, F)
    
    # Display the rectified images side by side
    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(img1_rect, cmap='gray')
    plt.title('Rectified Image 1')
    plt.subplot(122), plt.imshow(img2_rect, cmap='gray')
    plt.title('Rectified Image 2')
    plt.tight_layout()
    plt.show()
