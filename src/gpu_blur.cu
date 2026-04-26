#include "gpu_blur.h"
#include <cuda_runtime.h>

static unsigned char* d_blur_input = NULL;
static unsigned char* d_blur_output = NULL;

extern "C" void gpu_alloc(int width, int height) {
    size_t size = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_blur_input, size);
    cudaMalloc((void**)&d_blur_output, size);
}

extern "C" void gpu_free() {
    if (d_blur_input) cudaFree(d_blur_input);
    if (d_blur_output) cudaFree(d_blur_output);
}

__constant__ float d_kernel[25] = {
    0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f,
    0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f,
    0.0219f, 0.0983f, 0.1621f, 0.0983f, 0.0219f,
    0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f,
    0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f
};

__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int kernel_idx = 0;

        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int neighbor_x = x + kx;
                int neighbor_y = y + ky;

                if (neighbor_x < 0) neighbor_x = 0;
                else if (neighbor_x >= width) neighbor_x = width - 1;

                if (neighbor_y < 0) neighbor_y = 0;
                else if (neighbor_y >= height) neighbor_y = height - 1;

                unsigned char pixel = input[neighbor_y * width + neighbor_x];
                sum += pixel * d_kernel[kernel_idx];
                kernel_idx++;
            }
        }

        output[y * width + x] = (unsigned char)sum;
    }
}

extern "C" void gpu_gaussian_blur(unsigned char* input, unsigned char* output, int width, int height) {
    size_t size = width * height * sizeof(unsigned char);
    cudaMemcpy(d_blur_input, input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    gaussian_blur_kernel<<<gridDim, blockDim>>>(d_blur_input, d_blur_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_blur_output, size, cudaMemcpyDeviceToHost);
}
