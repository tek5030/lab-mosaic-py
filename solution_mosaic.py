import cv2
import numpy as np

# from common_lab_utils import \


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

    detector = cv2.xfeatures2d.SIFT_create()
    desc_extractor = cv2.xfeatures2d.SIFT_create()
    cv2.BFMatcher_create(desc_extractor.defaultNorm())

    while True:
        # Read next frame.
        success, frame = cap.read()
        if not success:
            print(f"The video source {video_source} stopped")
            break

        # Convert frame to gray scale image.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show the results
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
