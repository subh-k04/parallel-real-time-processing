#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "cpu_blur.h"
#include "gpu_blur.h"
#include "gpu_pipeline.h"

void cpu_background_subtraction(unsigned char* current, unsigned char* previous, unsigned char* output, int width, int height) {
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            output[idx] = (unsigned char)std::abs((int)current[idx] - (int)previous[idx]);
            previous[idx] = current[idx]; // Update previous
        }
    }
}

void cpu_sobel_edge_detection(unsigned char* input, unsigned char* output, int width, int height) {
    const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gx = 0, gy = 0, k_idx = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx < 0) nx = 0; else if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0; else if (ny >= height) ny = height - 1;
                    
                    unsigned char p = input[ny * width + nx];
                    gx += p * sobel_x[k_idx];
                    gy += p * sobel_y[k_idx];
                    k_idx++;
                }
            }
            int mag = std::sqrt(gx * gx + gy * gy);
            output[y * width + x] = (unsigned char)(mag > 255 ? 255 : mag);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::VideoCapture cap;
    cv::VideoWriter writer;
    int width = 0, height = 0, total_frames = 0;
    double start_time = 0.0;

    if (rank == 0) {
        cap.open("input/input.mp4");
        if (!cap.isOpened()) MPI_Abort(MPI_COMM_WORLD, 1);
        
        width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = (int)cap.get(cv::CAP_PROP_FPS);
        total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

        writer.open("output/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height), false);
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_frames, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = height / size;
    int remainder = height % size;
    int local_height = rows_per_process + (rank == size - 1 ? remainder : 0);
    int local_elements = local_height * width;

    unsigned char* local_input = new unsigned char[local_elements];
    unsigned char* local_blurred = new unsigned char[local_elements];
    unsigned char* local_bgsub = new unsigned char[local_elements];
    unsigned char* local_previous = new unsigned char[local_elements](); // zero-initialized
    unsigned char* local_output = new unsigned char[local_elements];

#ifdef USE_CUDA
    gpu_alloc(width, local_height);
    gpu_pipeline_alloc(width, local_height);
#endif

    for (int f = 0; f < total_frames; f++) {
        cv::Mat frame, gray_frame;
        if (rank == 0) {
            cap >> frame;
            if (!frame.empty()) cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        }

        if (rank == 0) {
            memcpy(local_input, gray_frame.data, local_elements * sizeof(unsigned char));
            int offset = local_elements;
            for (int i = 1; i < size; i++) {
                int send_elements = (rows_per_process + (i == size - 1 ? remainder : 0)) * width;
                MPI_Send(gray_frame.data + offset, send_elements, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
                offset += send_elements;
            }
        } else {
            MPI_Recv(local_input, local_elements, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Stage 1 - Motion blur detection (skip frames with variance < 100.0)
        float variance = laplacian_variance(local_input, width, local_height);
        
        if (variance >= 100.0f) {
            // Stage 2 - Gaussian blur
#ifdef USE_CUDA
            gpu_gaussian_blur(local_input, local_blurred, width, local_height);
#else
            cpu_gaussian_blur(local_input, local_blurred, width, local_height);
#endif

            // Stage 3 - Background subtraction
            if (f == 0) {
                memcpy(local_previous, local_blurred, local_elements);
                memset(local_bgsub, 0, local_elements); // First frame outputs blank for bg sub
            } else {
#ifdef USE_CUDA
                gpu_background_subtraction(local_blurred, local_previous, local_bgsub, width, local_height);
#else
                cpu_background_subtraction(local_blurred, local_previous, local_bgsub, width, local_height);
#endif
            }

            // Stage 4 - Sobel edge detection
#ifdef USE_CUDA
            gpu_sobel_edge_detection(local_bgsub, local_output, width, local_height);
#else
            cpu_sobel_edge_detection(local_bgsub, local_output, width, local_height);
#endif
        } else {
            // Frame is blurry, send back original strip
            memcpy(local_output, local_input, local_elements);
        }

        if (rank == 0) {
            cv::Mat out_gray(height, width, CV_8UC1);
            memcpy(out_gray.data, local_output, local_elements * sizeof(unsigned char));

            int offset = local_elements;
            for (int i = 1; i < size; i++) {
                int recv_elements = (rows_per_process + (i == size - 1 ? remainder : 0)) * width;
                MPI_Recv(out_gray.data + offset, recv_elements, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += recv_elements;
            }

            // Add real-time FPS text to the stitched image before writing
            double elapsed = MPI_Wtime() - start_time;
            double current_fps = (f + 1) / elapsed;
            cv::putText(out_gray, "FPS: " + std::to_string((int)current_fps), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
            writer.write(out_gray);
        } else {
            MPI_Send(local_output, local_elements, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }

    delete[] local_input; delete[] local_blurred;
    delete[] local_bgsub; delete[] local_previous; delete[] local_output;

#ifdef USE_CUDA
    gpu_free();
    gpu_pipeline_free();
#endif

    if (rank == 0) {
        cap.release();
        writer.release();
        double total_time = MPI_Wtime() - start_time;
        std::cout << "Total Frames: " << total_frames << std::endl;
        std::cout << "Total Time: " << total_time << " s" << std::endl;
        std::cout << "Average FPS: " << total_frames / total_time << std::endl;
    }

    MPI_Finalize();
    return 0;
}
