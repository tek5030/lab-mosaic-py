# Step 5: Creating an image mosaic
We will in this last part use the computed homography to combine the images in a common pixel coordinate system (a common image).

## 6. What does the similarity *S* do?
In `run_mosaic_lab()` at [lab_mosaic.py](../lab_mosaic.py), we have defined the similarity *S* as:

```python
    S = np.array([
        [0.5, 0.0, 0.25 * frame_cols],
        [0.0, 0.5, 0.25 * frame_rows],
        [0.0, 0.0, 1.0]
    ])
```

- What does this transformation do when applied to an image point?

This similarity will define the transformation from the reference image to the *mosaic image*.

## 7. Transform the reference image and insert it into the mosaic
Use the similarity *S* above to warp the reference image into the mosaic image:

```python
mosaic = cv2.warpPerspective(ref_image, ?, img_size)
```

## 8. Transform the current frame
Use the similarity *S* and the computed homography *H* to warp the current frame into mosaic image coordinates:

```python
frame_warp = cv2.warpPerspective(curr_image, ?, img_size)
```

What happens if you set `mosaic = frame_warp`?


## 9. Compute a mask for the transformed current frame
We will use a mask to define which pixels in the mosaic that should be updated with the current frame. 
We will do this by taking an image with all ones (all the pixels in the original current frame), and warping it in the exact same way as we did with the current frame. 
This should give us a warped mask that defines where in the mosaic image we want to insert the warped current frame:

```python
mask = np.ones(np.flip(img_size), dtype=np.uint8)
mask_warp = cv2.warpPerspective(mask, ?, img_size)
```

## 10. Insert the current frame and remove the edge effects
We can use `cv2.copyTo()` to insert the warped current frame into the mosaic:

```python
cv2.copyTo(frame_warp, mask_warp, dst=mosaic)
```

Do this and run.
Cool, right?

Notice that there are some unwanted effects around the edges of the current frame. 
This is because of interpolations when warping the mask. 
How can we remove these effects with `cv2.erode()`? Try!


## Then...
Now you are finished with the lab! 
But if you still have some time left, or want to continue with this lab at home, there is other cool stuff you can try:
- Try `cv2.findHomography()` instead of our method.
- Use the estimated homography to search for more, weaker correspondences. Then recompute!
- Use the keypoint detector you implemented in [lab-corners-py](https://github.com/tek5030/lab-corners-py).
- Apply blending to the mosaic, like we did in [lab-image-blending-py](https://github.com/tek5030/lab-image-blending-py).
- Expand the program to make a mosaic of more than two images.
