import cv2
import numpy as np
from numpy import linalg
import timeit

from common_lab_utils import HomographyEstimate, homogeneous, hnormalized, \
    retain_best, extract_matching_points, randomly_select_points, \
    colours, draw_keypoint_detections, draw_keypoint_matches, draw_estimation_details


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
    window_mosaic = 'Mosaic result'
    cv2.namedWindow(window_match, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_mosaic, cv2.WINDOW_NORMAL)

    # Set up a similarity transform.
    frame_cols = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_rows = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    img_size = np.array((frame_cols, frame_rows), dtype=int)

    # TODO 6: Question: What does this similarity transform do?
    S = np.array([
        [0.5, 0.0, 0.25 * frame_cols],
        [0.0, 0.5, 0.25 * frame_rows],
        [0.0, 0.0, 1.0]
    ])

    # TODO 1: Experiment with blob and corner feature detectors.
    # TODO 3: Experiment with feature matching
    # Set up objects for detection, description and matching.
    detector = cv2.ORB_create(nfeatures=1000)
    desc_extractor = cv2.ORB_create()
    matcher = cv2.BFMatcher_create(desc_extractor.defaultNorm())

    # Create homography estimator
    estimator = HomographyEstimator()

    # Reference image for mosaic.
    ref_image = None
    ref_keypoints = None
    ref_descriptors = None

    while True:
        # Read next frame.
        success, curr_image = cap.read()
        if not success:
            print(f"The video source {video_source} stopped")
            break

        # Convert frame to gray scale image.
        gray_frame = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        vis_img = np.copy(curr_image)

        # Detect keypoints
        # Measure how long the processing takes.
        start = timeit.default_timer()
        curr_keypoints = np.asarray(detector.detect(gray_frame))
        end = timeit.default_timer()
        duration_detection = end - start

        # Uncomment this to keep the highest scoring points
        # (for methods that do not have this possibility as standard and produce a lot of detections).
        # best = retain_best(curr_keypoints, 1000)
        # curr_keypoints = curr_keypoints[best]

        if ref_descriptors is None:
            # No reference image, draw keypoints.
            # (Press space to create a reference image).
            draw_keypoint_detections(vis_img, curr_keypoints, duration_detection, colours.red)
        else:
            # We have a reference, try to match features!
            # Measure how long the matching takes.
            start = timeit.default_timer()

            # Match descriptors with ratio test.
            curr_keypoints, frame_descriptors = desc_extractor.compute(gray_frame, curr_keypoints)
            matches = matcher.knnMatch(frame_descriptors, ref_descriptors, k=2)
            good_matches = extract_good_ratio_matches(matches, max_ratio=0.8)

            end = timeit.default_timer()
            duration_matching = end - start

            # Draw matching result
            vis_img = draw_keypoint_matches(
                curr_image,
                curr_keypoints,
                ref_image,
                ref_keypoints,
                good_matches,
                duration_detection,
                duration_matching
            )

            if len(good_matches) >= 10:
                # Extract pixel coordinates for corresponding points.
                matching_pts1, matching_pts2 = extract_matching_points(curr_keypoints, ref_keypoints, good_matches)

                # Estimate homography
                # Measure how long the estimation takes.
                start = timeit.default_timer()

                estimate = estimator.estimate(matching_pts1, matching_pts2)

                end = timeit.default_timer()
                duration_estimation = end - start

                # TODO 7: Transform the reference image according to the similarity S, and insert into the mosaic.
                mosaic = cv2.warpPerspective(ref_image, S, img_size)

                if estimate is not None:
                    H = estimate.homography

                    # TODO 8: Transform the current frame according to S and the computed homography.
                    frame_warp = cv2.warpPerspective(curr_image, S @ H, img_size)

                    # TODO 9: Compute a mask for the transformed image
                    mask = np.ones(np.flip(img_size), dtype=np.uint8)
                    mask_warp = cv2.warpPerspective(mask, S @ H, img_size)
                    mask_warp = cv2.erode(mask_warp, np.ones((3, 3)))

                    # TODO 10: Insert the current frame into the mosaic
                    cv2.copyTo(frame_warp, mask_warp, dst=mosaic)

                    # Draw estimation duration
                    draw_estimation_details(vis_img, duration_estimation, estimate.num_inliers)

                if mosaic is not None:
                    cv2.imshow(window_mosaic, mosaic)

        # Show the results
        cv2.imshow(window_match, vis_img)

        # Update the GUI and wait a short time for input from the keyboard.
        key = cv2.waitKey(1)

        # React to keyboard commands.
        if key == ord('q'):
            print("Quit")
            break

        elif key == ord(' '):
            # Set reference image for mosaic and compute descriptors.
            print("Set reference image")
            ref_image = np.copy(curr_image)
            ref_keypoints, ref_descriptors = desc_extractor.compute(gray_frame, curr_keypoints)

        elif key == ord('r'):
            # Reset
            # Make all reference data empty
            print("Reset")
            ref_image = None
            ref_keypoints = None
            ref_descriptors = None

    # Stop video source.
    cv2.destroyAllWindows()
    cap.release()


def extract_good_ratio_matches(matches, max_ratio):
    """
    Extracts a set of good matches according to the ratio test.

    :param matches: Input set of matches, the best and the second best match for each putative correspondence.
    :param max_ratio: Maximum acceptable ratio between the best and the next best match.
    :return: The set of matches that pass the ratio test.
    """
    if len(matches) == 0:
        return ()

    # TODO 2: Implement the ratio test.
    matches_arr = np.asarray(matches)
    distances = np.array([m.distance for m in matches_arr.ravel()]).reshape(matches_arr.shape)
    good = distances[:, 0] < distances[:, 1] * max_ratio

    # Return a tuple of good DMatch objects.
    return tuple(matches_arr[good, 0])


class HomographyEstimator:
    """Estimates a homography from point correspondences using RANSAC."""

    def __init__(self, p=0.99, distance_threshold=3.0, max_iterations=500):
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

        # TODO 4: Understand how we estimate the homography.
        if len(pts1) != len(pts2):
            raise ValueError("Point correspondence matrices did not have same size")

        pts1 = np.asarray(pts1).transpose()
        pts2 = np.asarray(pts2).transpose()

        # Find inliers
        is_inlier, num_inliers = self._ransac_estimator(pts1, pts2)

        if num_inliers < 4:
            return None

        # Estimate homography from set of inliers
        inliers_1 = pts1[:, is_inlier]
        inliers_2 = pts2[:, is_inlier]

        H = self._normalized_dlt_estimator(inliers_1, inliers_2)
        if H is None:
            return None

        return HomographyEstimate(H, num_inliers, is_inlier)

    def _ransac_estimator(self, pts1, pts2):
        """Finds a set of inliers for estimating a homography."""

        # Initialize maximum number of iterations.
        num_iterations = self._max_iterations
        iteration = 0
        best_inliers = []
        best_num_inliers = 0

        num_samples = 4
        while iteration < num_iterations:
            iteration += 1

            # Sample 4 random point correspondences
            samples_1, samples_2 = randomly_select_points(pts1, pts2, num_samples)

            # Determine test homography
            test_H = self._dlt_estimator(samples_1, samples_2)
            if test_H is None:
                continue

            try:
                test_H_inv = linalg.inv(test_H)
            except linalg.LinAlgError:
                continue

            # Count number of inliers
            reprojection_error = self._compute_reprojection_error(pts1, pts2, test_H, test_H_inv)

            test_inliers = reprojection_error < self._distance_threshold
            test_num_inliers = np.count_nonzero(test_inliers)

            # Update homography if test homography has the most inliers so far
            if test_num_inliers > 4 and test_num_inliers > best_num_inliers:
                # Update homography with larges inlier set
                best_inliers = test_inliers
                best_num_inliers = test_num_inliers

                # Adaptively update number of iterations.
                inlier_ratio = best_num_inliers / pts1.shape[1]
                if inlier_ratio == 1.0:
                    break

                num_iterations = np.minimum(
                    int(np.log(1.0 - self._p) / np.log(1.0 - inlier_ratio ** num_samples)),
                    self._max_iterations
                )

        return best_inliers, best_num_inliers

    def _compute_reprojection_error(self, pt1, pt2, H, H_inv):
        """Computes the two-sided reprojection error for a given homography."""

        # TODO 5: Compute the two-sided reprojection error.
        # Map points onto each other using the homography
        pt1_in_2 = hnormalized(H @ homogeneous(pt1))
        pt2_in_1 = hnormalized(H_inv @ homogeneous(pt2))

        # Compute the two-sided reprojection error \sigma_i.
        reprojection_error = np.linalg.norm(pt1 - pt2_in_1, axis=0) + np.linalg.norm(pt2 - pt1_in_2, axis=0)

        return reprojection_error

    def _dlt_estimator(self, pts1, pts2):
        """Estimates a homography from point correspondences using DLT."""

        def build_equation_set(pt1, pt2):
            return np.array([
                [0., 0., 0., -pt1[0], -pt1[1], -1., pt2[1] * pt1[0], pt2[1] * pt1[1], pt2[1]],
                [pt1[0], pt1[1], 1., 0., 0., 0., -pt2[0] * pt1[0], -pt2[0] * pt1[1], -pt2[0]]
            ])

        # Construct the equation matrix
        A = np.concatenate([eqs for eqs in map(build_equation_set, pts1.transpose(), pts2.transpose())], axis=0)

        # Solve using SVD
        try:
            _, _, Vh = linalg.svd(A, full_matrices=True)
        except linalg.LinAlgError:
            print("Warning: SVD computation did not converge")
            return None

        return Vh[-1, :].reshape((3, 3))

    def _normalized_dlt_estimator(self, pts1, pts2):
        """Estimates a homography from point correspondences using the normalized DLT."""

        # Normalize points
        S1 = self._find_normalizing_similarity(pts1)
        S2 = self._find_normalizing_similarity(pts2)
        pts1_normalized = hnormalized(S1 @ homogeneous(pts1))
        pts2_normalized = hnormalized(S2 @ homogeneous(pts2))

        # Estimate the homography
        H = self._dlt_estimator(pts1_normalized, pts2_normalized)
        if H is None:
            return None

        # Transform back to the original frame
        H = linalg.inv(S2) @ H @ S1

        if H[2, 2] == 0:
            return None

        # Standardize H
        H /= H[2, 2]

        return H

    def _find_normalizing_similarity(self, pts):
        """Finds a normalizing similarity transform for a set of points."""

        # Centroid of points
        center = np.mean(pts, axis=1)

        # Compute the mean distance from centroid over all pts
        r_mean = np.mean(np.linalg.norm(pts - center[:, np.newaxis], axis=0))

        # The normalizing similarity matrix S
        scale = np.sqrt(2.) / r_mean

        S = np.array([
            [scale, 0, -scale * center[0]],
            [0, scale, -scale * center[1]],
            [0, 0, 1]
        ])

        return S


if __name__ == "__main__":
    run_mosaic_solution()
