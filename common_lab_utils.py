import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class HomographyEstimate:
    homography: np.ndarray = np.array([], dtype=np.float32)
    num_inliers: int = 0
    inliers: np.ndarray = np.array([], dtype=int)


class DotDict(dict):
    """
    dot.notation access to dictionary attributes.

    https://stackoverflow.com/a/23689767/14325545
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


colours = DotDict({
    'green': (0, 255, 0),
    'red': (0, 0, 255)
})


font = DotDict({
    'face': cv2.FONT_HERSHEY_PLAIN,
    'scale': 1.0
})


def retain_best(keypoints, num_to_keep):
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return best


def randomly_select_points(pts1, pts2, num_points):
    n = pts1.shape[1]
    idx = np.random.choice(n, size=num_points, replace=False)
    return pts1[:,idx], pts2[:,idx]


def extract_good_ratio_matches(matches, max_ratio):
    npmatches = np.asarray(matches)
    distances = np.array([m.distance for m in npmatches.ravel()]).reshape(npmatches.shape)
    good = distances[:, 0] < distances[:, 1] * max_ratio
    return tuple(npmatches[good, 0])


def extract_matching_points(keypoints1, keypoints2, matches):
    keypoints1 = np.asarray(keypoints1)
    keypoints2 = np.asarray(keypoints2)

    query_idx = [m.queryIdx for m in matches]
    train_idx = [m.trainIdx for m in matches]

    matching_pts1 = [k.pt for k in keypoints1[query_idx]]
    matching_pts2 = [k.pt for k in keypoints2[train_idx]]

    return matching_pts1, matching_pts2


def draw_keypoint_detections(img, keypoints, duration, colour=None):
    cv2.drawKeypoints(img, keypoints, img, colour)
    cv2.putText(img, f"detection: {duration:.2f}", (10, 20), font.face, font.scale, colours.green)


def draw_keypoint_matches(img1, keypoints1, img2, keypoints2, matches, duration_detection, duration_matching):
    vis_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
    cv2.putText(vis_img, f"detection: {duration_detection:.2f}", (10, 20), font.face, font.scale, colours.red)
    cv2.putText(vis_img, f"matching:  {duration_matching:.2f}", (10, 40), font.face, font.scale, colours.red)
    return vis_img


def draw_estimation_details(vis_img, duration, num_inliers):
    cv2.putText(vis_img, f"estimation: {duration:.2f}", (10, 60), font.face, font.scale, colours.red)
    cv2.putText(vis_img, f"inliers:      {num_inliers:}", (10, 80), font.face, font.scale, colours.red)


def multiply_homogeneous(m, pts):
    n = pts.shape[1]
    pts_h = np.r_[pts, [np.ones(n)]]

    hnormalized = lambda m: m[:-1, :] / m[-1, :]

    return hnormalized(m @ pts_h)