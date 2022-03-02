import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class HomographyEstimate:
    homography: np.ndarray = np.array([], dtype=np.float32)
    num_inliers: int = 0
    inliers: np.ndarray = np.array([], dtype=int)

@dataclass
class Font:
    face = cv2.FONT_HERSHEY_PLAIN
    scale = 1.0


@dataclass
class Colour:
    green = (0, 255, 0)
    red = (0, 0, 255)


def draw_keypoint_detections(img, keypoints, duration, colour=None):
    cv2.drawKeypoints(img, keypoints, img, colour)
    cv2.putText(img, f"detection: {duration:.2f}", (10, 20), Font.face, Font.scale, Colour.green)


def retain_best(keypoints, num_to_keep):
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return best


def randomly_select_points(points, num_points):
    return points[np.random.choice(len(points), size=num_points, replace=False)]
