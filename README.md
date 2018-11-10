# Stereo Visual Odometry using Supervised Detector
```
export OMP_NUM_THREADS=16
g++ main.cpp -std=c++14 -I tiny-dnn/ -lpthread `pkg-config --libs --cflags opencv` -fopenmp -g -O2 -ltbb -o main
```
