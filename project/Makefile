CXX = mpicxx
NVCC = nvcc
CXXFLAGS = -fopenmp
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
CUDA_LIBS = -lcuda -lcudart

# -DUSE_CUDA allows the C++ code to conditionally call GPU functions
GPUFLAGS = -fopenmp -DUSE_CUDA

cpu: src/main.cpp src/cpu_blur.c
	$(CXX) $(CXXFLAGS) -o blur_cpu src/main.cpp src/cpu_blur.c $(OPENCV_FLAGS)

gpu: src/gpu_blur.cu src/gpu_pipeline.cu src/main.cpp src/cpu_blur.c
	$(NVCC) -c src/gpu_blur.cu -o gpu_blur.o
	$(NVCC) -c src/gpu_pipeline.cu -o gpu_pipeline.o
	$(CXX) $(GPUFLAGS) -o blur_gpu src/main.cpp src/cpu_blur.c gpu_blur.o gpu_pipeline.o $(CUDA_LIBS) $(OPENCV_FLAGS)

sequential: src/sequential.cpp
	$(CXX) src/sequential.cpp -o blur_sequential $(OPENCV_FLAGS)

clean:
	rm -f blur_cpu blur_gpu blur_sequential blur_cpu.exe blur_gpu.exe blur_sequential.exe gpu_blur.o gpu_pipeline.o output/*.mp4
