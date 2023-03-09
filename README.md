# Image mosaics from feature matching

Welcome to this lab in the computer vision course [TEK5030] at the University of Oslo.

In this lab we will experiment with feature detection and matching to extract point correspondences between two images. 
We will then use these correspondences to estimate a homography between the images, which will enable us to create a live video mosaic!

![Screenshot from the lab](lab-guide/img/screenshot_lab4.png)

Start by cloning this repository on your machine. 
Then open the lab project in your editor.

The lab is carried out by following these steps:

1. [Get an overview](lab-guide/1-get-an-overview.md)
2. [Features in OpenCV](lab-guide/2-features-in-opencv.md)
3. [Experiment with feature matching](lab-guide/3-experiment-with-feature-matching.md)
4. [Homography estimation](lab-guide/4-homography-estimation.md)
5. [Creating an image mosaic](lab-guide/5-creating-an-image-mosaic.md) 

Please start the lab by going to the [first step](lab-guide/1-get-an-overview.md).





## Prerequisites

Here is a quick reference if you need to set up a Python virtual environment manually:

```bash
python3.8 -m venv venv  # any python version > 3.8 is OK
source venv/bin/activate.
# expect to see (venv) at the beginning of your prompt.
pip install -U pip  # <-- Important step for Ubuntu 18.04!
pip install -r requirements.txt
```

Please consult the [resource pages] if you need more help with the setup.

[TEK5030]: https://www.uio.no/studier/emner/matnat/its/TEK5030/
[resource pages]: https://tek5030.github.io