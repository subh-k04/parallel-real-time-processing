#include "cpu_blur.h"
#include <omp.h>

const float kernel[25] = {
    0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f,
    0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f,
    0.0219f, 0.0983f, 0.1621f, 0.0983f, 0.0219f,
    0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f,
    0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f
};

void cpu_gaussian_blur(unsigned char* input, unsigned char* output, int width, int height) {
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            int kernel_idx = 0;

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;

                    // Handle boundary pixels by clamping to nearest edge
                    if (neighbor_x < 0) neighbor_x = 0;
                    else if (neighbor_x >= width) neighbor_x = width - 1;

                    if (neighbor_y < 0) neighbor_y = 0;
                    else if (neighbor_y >= height) neighbor_y = height - 1;

                    unsigned char pixel = input[neighbor_y * width + neighbor_x];
                    sum += pixel * kernel[kernel_idx];
                    kernel_idx++;
                }
            }

            output[y * width + x] = (unsigned char)sum;
        }
    }
}

float laplacian_variance(unsigned char* frame, int width, int height) {
    long long sum = 0;
    long long sq_sum = 0;
    int n = (width - 2) * (height - 2);

    // reduction clause prevents race conditions when accumulating sum and sq_sum across multiple threads
    #pragma omp parallel for reduction(+:sum, sq_sum)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int val = 
                frame[(y - 1) * width + x] +
                frame[(y + 1) * width + x] +
                frame[y * width + (x - 1)] +
                frame[y * width + (x + 1)] -
                4 * frame[y * width + x];
            
            sum += val;
            sq_sum += val * val;
        }
    }

    float mean = (float)sum / n;
    return ((float)sq_sum / n) - (mean * mean);
}
