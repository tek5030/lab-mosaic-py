#!/usr/bin/env bash

# OpenCV on Wheels: https://github.com/opencv/opencv-python.git

build_opencv() (
  set -euxo pipefail

  [[ ! -d venv ]] && \
    python3.8 -m venv venv

  [[ ! -d opencv-python ]] && \
    git clone -b 62 --recursive https://github.com/opencv/opencv-python.git --depth 1

  source venv/bin/activate
  pip install -U pip
  pip install wheel

  cd opencv-python

  export ENABLE_CONTRIB=1
  export OPENCV_ENABLE_NONFREE=1
  export CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" -DENABLE_CONTRIB=1 -DOPENCV_ENABLE_NONFREE=1 -DWITH_CUDA=1 -DOpenGL_GL_PREFERENCE=GLVND -DCMAKE_CUDA_ARCHITECTURES=50-real -DCUDA_ARCH_BIN=5.0'

  pip wheel . --verbose 2>&1 | tee $(date "+%Y-%m-%d_%H%M%S").log
)

build_opencv

