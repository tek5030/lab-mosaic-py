# Step 1: Get an overview
We will as usual start by presenting an overview of the method and the contents of this project.

## Algorithm overview
The main steps in today's lab are:
- Apply feature detectors and descriptors available in OpenCV to detect and describe keypoints.
  - Detect keypoints in the current frame.
  - Press *\<space\>* to set the current frame as the reference frame.

- Match keypoints between the reference frame and new current frames using OpenCV, and extract good matches by applying the ratio test.

- Use the point correspondences to estimate a homography between two images, using RANSAC and the (normalized) DLT:
  - Find a large inlier set by applying a DLT-estimator repeatedly on a minimal set of correspondences using RANSAC.
  - Apply normalized DLT on the inlier set.
  
- Use the estimated homography to combine the two images in an image mosaic:
  - Warp a downscaled reference image into the mosaic image.
  - Warp the current frame into the mosaic image based on the estimated homography.

## Introduction to the project source files
We have chosen to distribute the code on the following modules:
- [**lab_mosaic.py**](../lab_mosaic.py)
  
  Contains the main loop of the program and all exercises. 
  Your task will be to finish the code in this module. 
  
- [**common_lab_utils.py**](../common_lab_utils.py)

  This module contains utility functions and classes that we will use both in the lab and in the solution.
  Please take a quick look through the code.
 
- [**solution_mosaic.py**](../solution_mosaic.py)

  This is our proposed solution to the lab.
  Please try to solve the lab with help from others instead of just jumping straight to the solution ;)
 
  
  Please continue to [the next step](2-features-in-opencv.md) to get started with features!