# Real-Time Video Processing Pipeline
> Hybrid parallel architecture using MPI · OpenMP · CUDA

A four-stage 1080p video preprocessing pipeline built for the kind of throughput that sequential code simply cannot deliver. Raw surveillance footage goes in — noise removed, motion isolated, edges extracted — at **78 FPS** on a single GPU, **8.82× faster** than a single thread.

## What it does

```
Input Video
  → Motion Blur Detection   (OpenMP)   filter bad frames before wasting compute
  → Gaussian Blur           (CUDA)     remove sensor noise, 2M threads in parallel  
  → Background Subtraction  (CUDA)     isolate moving objects from static background
  → Sobel Edge Detection    (CUDA)     extract object boundaries for downstream AI
Output Video
```

## Why it's built this way

Every pixel in a video frame is independent — no pixel needs to wait for another. This makes video preprocessing **embarrassingly parallel**. MPI distributes frames across processes, OpenMP parallelizes rows within each process, and CUDA assigns one thread per pixel on the GPU. Three levels of parallelism, each targeting a different layer of the hardware.

## Built with

`C++` `CUDA 11.5` `OpenMPI` `OpenMP` `OpenCV 4.5` `WSL2 Ubuntu 22.04`

---

*UCS645 Parallel and Distributed Computing · TIET Patiala · Team SegFault*
