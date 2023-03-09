# Step 2: Features in OpenCV
There are several feature methods available in the [features2d] and [xfeatures2d] modules in OpenCV. 
Some of these are detectors, some compute descriptors and some do both. 
They all extend the abstact [`cv::Feature2D`] base class.

First, please take a look at the documentation above to get an overview.
Then, lets try it out!


## How to construct
In C++ you construct detectors and descriptor extractors by calling their static `create()` factory method.
In Python, this method becomes `<feature class>_create()`.
You can also typically set different parameters using this method (read the documentation).

The following example will create a [FAST] feature detector and a [LATCH] descriptor extractor:

```python
    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.xfeatures2d.LATCH_create()
```

In order to use the descriptors for matching, we will also need a *matcher*. 
The following example constructs a *brute force matcher* based on the default metric used by the descriptor:

```python
matcher = cv2.BFMatcher_create(desc_extractor.defaultNorm())
```

## How to use
We detect keypoints by calling the detector's `detect()` method.

```python
keypoints = detector.detect(gray_frame)
```

The detected keypoints are returned as a tuple of `cv2.KeyPoint`s. 

Take a look at [`cv::KeyPoint`]. 
What fields does it contain?

We compute descriptors for each keypoint by calling the descriptor extractor's `compute()` method:

```python
curr_keypoints, frame_descriptors = desc_extractor.compute(gray_frame, curr_keypoints)
```

This will return the descriptors (as matrices) and possibly updated keypoints (see [`compute()`]).

We will use the matcher to match descriptors between the current frame and a reference frame. 
In order to apply the *ratio test* (from the lectures), we will need to extract the two best matches for each keypoint. 
We can do this with the matcher's `knnMatch()` method:

```python
matches = matcher.knnMatch(frame_descriptors, ref_descriptors, k=2)
```

The result is returned as a tuple of tuples, each with two `cv2.DMatch` objects (the best and the second best matches).

Take a look at [`cv::DMatch`].
What fields does it contain?

## `run_mosaic_lab()`
Now, take look at `run_mosaic_lab()` in [lab_mosaic.py](../lab_mosaic.py), and find where each of the steps above are performed in the code.
This is a pretty advanced program, so ask the instructors if you have trouble understanding what is going on in this function.

Then, please continue to the [next step](3-experiment-with-feature-matching.md).

[features2d]:  https://docs.opencv.org/4.5.5/da/d9b/group__features2d.html
[xfeatures2d]: https://docs.opencv.org/4.5.5/d1/db4/group__xfeatures2d.html
[`cv::Feature2D`]: https://docs.opencv.org/4.5.5/d0/d13/classcv_1_1Feature2D.html
[`compute()`]:     https://docs.opencv.org/4.5.5/d0/d13/classcv_1_1Feature2D.html#ad18e1027ffc8d9cffbb2d59c1d05b89e
[`cv::KeyPoint`]:  https://docs.opencv.org/4.5.5/d2/d29/classcv_1_1KeyPoint.html
[`cv::DMatch`]:    https://docs.opencv.org/4.5.5/d4/de0/classcv_1_1DMatch.html

[FAST]: https://www.edwardrosten.com/work/fast.html
[LATCH]: https://talhassner.github.io/home/publication/2016_WACV_2