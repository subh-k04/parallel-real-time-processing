#include "gpu_pipeline.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

static unsigned char* d_bg_curr = NULL;
static unsigned char* d_bg_prev = NULL;
static unsigned char* d_bg_out = NULL;
static unsigned char* d_sobel_in = NULL;
static unsigned char* d_sobel_out = NULL;

extern "C" void gpu_pipeline_alloc(int width, int height) {
    size_t size = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_bg_curr, size);
    cudaMalloc((void**)&d_bg_prev, size);
    cudaMalloc((void**)&d_bg_out, size);
    cudaMalloc((void**)&d_sobel_in, size);
    cudaMalloc((void**)&d_sobel_out, size);
}

extern "C" void gpu_pipeline_free() {
    if (d_bg_curr) cudaFree(d_bg_curr);
    if (d_bg_prev) cudaFree(d_bg_prev);
    if (d_bg_out) cudaFree(d_bg_out);
    if (d_sobel_in) cudaFree(d_sobel_in);
    if (d_sobel_out) cudaFree(d_sobel_out);
}

__global__ void bg_sub_kernel(unsigned char* current, unsigned char* previous, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int diff = abs((int)current[idx] - (int)previous[idx]);
        output[idx] = (unsigned char)diff;
        
        previous[idx] = current[idx];
    }
}

extern "C" void gpu_background_subtraction(unsigned char* current, unsigned char* previous, unsigned char* output, int width, int height) {
    size_t size = width * height * sizeof(unsigned char);

    cudaMemcpy(d_bg_curr, current, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bg_prev, previous, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    bg_sub_kernel<<<gridDim, blockDim>>>(d_bg_curr, d_bg_prev, d_bg_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_bg_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(previous, d_bg_prev, size, cudaMemcpyDeviceToHost);
}

__constant__ int d_sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__global__ void sobel_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int gx = 0, gy = 0, k_idx = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;

                if (nx < 0) nx = 0; else if (nx >= width) nx = width - 1;
                if (ny < 0) ny = 0; else if (ny >= height) ny = height - 1;

                unsigned char pixel = input[ny * width + nx];
                gx += pixel * d_sobel_x[k_idx];
                gy += pixel * d_sobel_y[k_idx];
                k_idx++;
            }
        }

        int mag = (int)sqrtf((float)(gx * gx + gy * gy));
        if (mag > 255) mag = 255;
        output[y * width + x] = (unsigned char)mag;
    }
}

extern "C" void gpu_sobel_edge_detection(unsigned char* input, unsigned char* output, int width, int height) {
    size_t size = width * height * sizeof(unsigned char);

    cudaMemcpy(d_sobel_in, input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    sobel_kernel<<<gridDim, blockDim>>>(d_sobel_in, d_sobel_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_sobel_out, size, cudaMemcpyDeviceToHost);
}
