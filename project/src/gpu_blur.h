#ifndef GPU_BLUR_H
#define GPU_BLUR_H

#ifdef __cplusplus
extern "C" {
#endif

void gpu_alloc(int width, int height);
void gpu_free();

void gpu_gaussian_blur(unsigned char* input, unsigned char* output, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // GPU_BLUR_H
