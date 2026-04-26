#ifndef CPU_BLUR_H
#define CPU_BLUR_H

#ifdef __cplusplus
extern "C" {
#endif

void cpu_gaussian_blur(unsigned char* input, unsigned char* output, int width, int height);

float laplacian_variance(unsigned char* frame, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // CPU_BLUR_H
