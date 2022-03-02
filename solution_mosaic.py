import cv2
import numpy as np
import timeit

from common_lab_utils import *

class HomographyEstimator:
    """Estimates a homography from point correspondences using RANSAC."""

    def __init__(self, p=0.99, distance_threshold=5.0, max_iterations=10000):
        """
        Constructs the estimator.

        :param p: The desired probability of getting a good sample.
        :param distance_threshold: The maximum error a good sample can have as defined by the two-sided reprojection error.
        :param max_iterations: The absolute maximum iterations allowed, ignoring p if necessary.
        """
        self._p = p
        self._distance_threshold = distance_threshold
        self._max_iterations = max_iterations

    def estimate(self, pts1, pts2):
        """
        Estimate a homography from point correspondences.

        :param pts1: Set of corresponding points from image 1.
        :param pts2: Set of corresponding points from image 2.
        :return: The estimated homography.
        """
        if len(pts1) != len(pts2):
            raise ValueError("Point correspondence matrices did not have same size")

        # Find inliers
        is_inlier = self._ransac_estimator(pts1, pts2)

        if len(is_inlier) < 4:
            return None

        # Estimate homography from set of inliers
        inliers_1 = pts1[is_inlier]
        inliers_2 = pts2[is_inlier]

        return HomographyEstimate(normalized_dlt_estimator(inliers_1, inliers_2), len(is_inlier), is_inlier)

    def _ransac_estimator(self, pts1, pts2):
        """Finds a set of inliers for estimating a homography."""

        # Initialize maximum number of iterations.
        num_iterations = self._max_iterations
        iteration = 0
        test_inliers = []
        best_inliers = []
        best_num_inliers = 0

        num_samples = 4
        while iteration < num_iterations:
            iteration += 1

            # Sample 4 random points
            rand_selection = randomly_select_points(pts1, num_samples)
            samples_1 = pts1[rand_selection]
            samples_2 = pts2[rand_selection]

            # Determine test homography
            test_h = dlt_estimator(samples_1, samples_2)
            test_h_inv = np.invert(test_H)

            # Count number of inliers
            test_num_inliers = 0
            for i in range(0, len(pts1)):
                if compute_reprojection_error(pts1[i], pts2[i], test_H, test_H_inv) < self._distance_threshold:
                    test_inliers.append(i)
                    test_num_inliers += 1

            # Update homography if test homography has the most inliers so far
            if test_num_inliers > 4 and test_num_inliers > best_num_inliers:
                # Update homography with larges inlier set
                best_inliers = test_inliers
                best_num_inliers = test_num_inliers

                # Adaptively update number of iterations.
                inlier_ratio = best_num_inliers / len(pts1)
                if inlier_ratio == 1.0:
                    break

                num_iterations = np.minimum(
                    int(np.log(1.0 - self._p) / np.log(1.0 - inlier_ratio ** num_samples)),
                    self._max_iterations
                )

        return best_inliers

    def dlt_estimator(self, pts1, pts2):
        """Estimates a homography from point correspondences using DLT."""

        # Construct the equation matrix
        x = lambda pt1, pt2 : np.array([
            [0, 0, 0, -pt1[0], -pt1[1], -1, pt2[1]*pt1[0], pt2[1]*pt1[1], pt2[1]],
            [pt1[0], pt1[1], 1, 0, 0, 0, -pt2[0]*pt1[0], -pt2[0]*pt1[1], -pt2[0]]
        ])

        #pts1 = np.array([[1, 1], [2, 2], [3, 3]])
        #pts2 = a = np.array([[1,1],[2,2],[3,3]])

        a = np.concatenate(list(map(x, pts1, pts2)), axis=0)

        # Solve using SVD


def run_mosaic_solution():
    # Connect to the camera.
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Set up windows
    window_match = 'Lab: Image mosaics from feature matching'
    window_mosaic = 'Mosaic Result'
    cv2.namedWindow(window_match, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_mosaic, cv2.WINDOW_NORMAL)

    detector = cv2.SIFT_create()
    desc_extractor = cv2.SIFT_create()
    cv2.BFMatcher_create(desc_extractor.defaultNorm())

    # Create homography estimator


    while True:
        # Read next frame.
        success, frame = cap.read()
        if not success:
            print(f"The video source {video_source} stopped")
            break

        # Convert frame to gray scale image.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints
        # Measure how long the processing takes.
        start = timeit.default_timer()
        keypoints = np.asarray(detector.detect(gray_frame))
        print(f"keypoints: {type(keypoints[0])}")
        end = timeit.default_timer()
        duration_corners = end - start

        # Keep the highest scoring points.
        best = retain_best(keypoints, 500)
        best_keypoints = keypoints[best]

        # Show the results
        draw_keypoint_detections(frame, best_keypoints, duration_corners, Colour.red)
        cv2.imshow(window_match, frame)
        cv2.imshow(window_mosaic, gray_frame)

        # Update the GUI and wait a short time for input from the keyboard.
        key = cv2.waitKey(1)

        # React to keyboard commands.
        if key == ord('q'):
            print("Quitting")
            break

    # Stop video source.
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    run_mosaic_solution()
