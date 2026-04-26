#ifndef GPU_PIPELINE_H
#define GPU_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

void gpu_pipeline_alloc(int width, int height);
void gpu_pipeline_free();

void gpu_background_subtraction(unsigned char* current, unsigned char* previous, unsigned char* output, int width, int height);
void gpu_sobel_edge_detection(unsigned char* input, unsigned char* output, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // GPU_PIPELINE_H
