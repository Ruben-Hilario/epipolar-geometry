# Fundamental Matrix Estimation using Normalized 8-Point Algorithm

This repository contains a Python script demonstrating the computation of the Fundamental Matrix between two images using a from-scratch implementation of the **normalized 8-point algorithm**. It utilizes OpenCV for feature detection (ORB) and matching, but computes the Fundamental Matrix manually using NumPy, including point normalization and Singular Value Decomposition (SVD). The script also provides an enhanced visualization showing bidirectional epipolar lines.

**Note:** This implementation uses the pure 8-point algorithm after normalization. It does **not** include robust estimation techniques like RANSAC (unlike `cv2.findFundamentalMat`). Therefore, its accuracy is sensitive to outliers among the matched feature points.

## Features

* Loads two grayscale images.
* Detects ORB features and computes descriptors.
* Matches features using `cv2.BFMatcher` with KNN matching and Lowe's ratio test.
* Implements **point normalization** for numerical stability.
* Implements the **normalized 8-point algorithm** using SVD to solve for the Fundamental Matrix (F).
* Enforces the rank-2 constraint on the computed F matrix.
* Visualizes *all* good feature matches found between the images.
* Provides an **enhanced visualization** showing points on each image along with the corresponding epipolar lines derived from the other image (bidirectional visualization).
* Prints the computed 3x3 Fundamental Matrix to the console.

## Requirements

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)

## Installation

1.  **Clone the repository (optional):**
    ```bash
    git clone https://github.com/Ruben-Hilario/epipolar-geometry.git
    cd epipolar-geometry
    ```
2.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy matplotlib
    ```

## Usage

1.  **Prepare Images:**
    * Place the two images you want to process into the **same directory** as the Python script.
    * **Important:** The script currently looks for `IMG_3098.jpg` and `IMG_3099.jpg` by default. You **must** either rename your images to match these exact names or modify the `cv2.imread` lines within the `if __name__ == "__main__":` block at the end of the script to point to your files.

2.  **Run the script:**
    ```bash
    python eight_point_algorithm.py
    ```
## Input

* Two image files (e.g., `IMG_3098.jpg` and `IMG_3099.jpg`) located in the **same directory** as the Python script. The script loads them in grayscale.

## Output

1.  **Console Output:** Prints the computed 3x3 Fundamental Matrix `F`.
2.  **Match Visualization Window:** A Matplotlib window displaying the two input images side-by-side with lines connecting *all* the feature matches considered "good" after the ratio test.
3.  **Epipolar Lines Visualization Window:** A Matplotlib window containing two subplots:
    * **Left Subplot (Image 1):** Shows points detected in Image 1 (blue dots) and the epipolar lines (cyan lines) corresponding to the matched points from Image 2.
    * **Right Subplot (Image 2):** Shows points detected in Image 2 (blue dots) and the epipolar lines (cyan lines) corresponding to the matched points from Image 1.

## Code Explanation

1.  **`normalize_points(points)`:**
    * Takes homogeneous points as input.
    * Calculates the centroid and mean distance of the points from the centroid.
    * Constructs a similarity transformation matrix `T` that translates the points so their centroid is at the origin and scales them so their average distance from the origin is `sqrt(2)`. This improves the conditioning of the matrix used in the 8-point algorithm.
    * Returns the normalized points and the transformation matrix `T`.

2.  **`compute_fundamental_matrix(points1, points2)`:**
    * Takes two sets of corresponding homogeneous points.
    * Normalizes both sets of points using `normalize_points`.
    * Constructs the `W` matrix (also known as `A` matrix) where each row corresponds to the epipolar constraint equation `x'^T F x = 0` for a pair of normalized points `x` and `x'`.
    * Performs Singular Value Decomposition (SVD) on `W`. The solution for the vectorized `F` is the column of `V` corresponding to the smallest singular value (the last column of `V.T` or `V[-1]` in NumPy).
    * Reshapes the solution vector into a 3x3 matrix `F_normalized`.
    * Enforces the rank-2 constraint of the Fundamental Matrix by performing SVD on `F_normalized`, zeroing out the smallest singular value, and reconstructing `F_normalized`.
    * Denormalizes the matrix using the transformation matrices `T1` and `T2`: `F = T2.T @ F_normalized @ T1`.
    * Returns the final Fundamental Matrix `F`.

3.  **`draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=20)`:**
    * Takes the original images, the *inhomogeneous* matched points, the computed `F`, and the number of lines to draw.
    * Selects a random subset of point pairs for visualization clarity.
    * For selected points in Image 2 (`pts2_selected`), it computes the corresponding epipolar lines in Image 1 using `line = F.T @ pt_homogeneous` and draws them on Image 1.
    * For selected points in Image 1 (`pts1_selected`), it computes the corresponding epipolar lines in Image 2 using `line = F @ pt_homogeneous` and draws them on Image 2.
    * Draws the selected points themselves on both images.
    * Displays the two images side-by-side using Matplotlib, ensuring correct color conversion for display (`cv2.COLOR_BGR2RGB`).

4.  **`eight_point_algorithm(img1, img2)`:**
    * Acts as the main pipeline function.
    * Performs ORB feature detection and description.
    * Performs feature matching (BFMatcher + KNN + Ratio Test).
    * Extracts the coordinates of good matches into `pts1` and `pts2` (inhomogeneous).
    * Converts points to homogeneous coordinates (`pts1_h`, `pts2_h`).
    * Calls `compute_fundamental_matrix` with the homogeneous points to get `F`.
    * Optionally (and currently enabled) displays the initial feature matches.
    * Returns the computed `F` and the original *inhomogeneous* points `pts1`, `pts2`.

5.  **Main Execution Block (`if __name__ == "__main__":`)**:
    * Loads the two grayscale images from the current directory, checking for loading errors.
    * Calls `eight_point_algorithm` to run the pipeline.
    * Prints the resulting `F` matrix.
    * Calls `draw_epipolar_lines` to show the final bidirectional visualization.

## Notes

* Remember to modify the hardcoded image filenames in the script (`IMG_3098.jpg`, `IMG_3099.jpg`) or rename your files accordingly. The script expects these files in the **same directory** it is run from.
* The quality of the computed Fundamental Matrix heavily depends on the accuracy and distribution of the initial feature matches obtained from ORB and the ratio test.
* Since this version lacks RANSAC, outliers (incorrect matches) can significantly degrade the result. Consider using images with clear features and distinct viewpoints for best results with this basic algorithm.
* Parameters like `nfeatures` in `ORB_create` and the ratio test threshold (`0.6`) can be tuned.