#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

const float kernel[25] = {
    0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f,
    0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f,
    0.0219f, 0.0983f, 0.1621f, 0.0983f, 0.0219f,
    0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f,
    0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f
};

const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

float laplacian_variance(unsigned char* frame, int width, int height) {
    long long sum = 0;
    long long sq_sum = 0;
    int n = (width - 2) * (height - 2);

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

void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            int kernel_idx = 0;

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    // Clamping boundary coordinates to the nearest valid edge pixel
                    if (nx < 0) nx = 0; else if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0; else if (ny >= height) ny = height - 1;

                    sum += input[ny * width + nx] * kernel[kernel_idx++];
                }
            }
            output[y * width + x] = (unsigned char)sum;
        }
    }
}

void background_subtraction(unsigned char* current, unsigned char* previous, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            output[idx] = (unsigned char)std::abs((int)current[idx] - (int)previous[idx]);
            previous[idx] = current[idx];
        }
    }
}

void sobel_edge_detection(unsigned char* input, unsigned char* output, int width, int height) {
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

int main() {
    cv::VideoCapture cap("input/input.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening input video!" << std::endl;
        return 1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer_blur("output/stage2_blur.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height), false);
    cv::VideoWriter writer_edges("output/stage4_edges.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height), false);
    
    int frame_size = width * height;
    unsigned char* local_blurred = new unsigned char[frame_size];
    unsigned char* local_bgsub = new unsigned char[frame_size];
    unsigned char* local_previous = new unsigned char[frame_size]();
    unsigned char* local_output = new unsigned char[frame_size];

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < total_frames; f++) {
        cv::Mat frame, gray_frame;
        cap >> frame;
        if (frame.empty()) break;
        
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        unsigned char* local_input = gray_frame.data;

        float variance = laplacian_variance(local_input, width, height);

        if (variance >= 100.0f) {
            gaussian_blur(local_input, local_blurred, width, height);

            if (f == 0) {
                memcpy(local_previous, local_blurred, frame_size);
                memset(local_bgsub, 0, frame_size);
            } else {
                background_subtraction(local_blurred, local_previous, local_bgsub, width, height);
            }

            sobel_edge_detection(local_bgsub, local_output, width, height);
        } else {
            memcpy(local_blurred, local_input, frame_size);
            memcpy(local_bgsub, local_input, frame_size);
            memcpy(local_output, local_input, frame_size);
        }

        cv::Mat out_stage2(height, width, CV_8UC1, local_blurred);
        cv::Mat out_stage4(height, width, CV_8UC1, local_output);
        
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        double current_fps = (f + 1) / elapsed.count();
        std::string fps_text = "FPS: " + std::to_string((int)current_fps);
        
        cv::putText(out_stage2, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
        cv::putText(out_stage4, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
        
        writer_blur.write(out_stage2);
        writer_edges.write(out_stage4);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end_time - start_time;

    std::cout << "Sequential Processing Complete!" << std::endl;
    std::cout << "Total Frames: " << total_frames << std::endl;
    std::cout << "Total Time: " << total_time.count() << " seconds" << std::endl;
    std::cout << "Average FPS: " << total_frames / total_time.count() << std::endl;

    delete[] local_blurred;
    delete[] local_bgsub;
    delete[] local_previous;
    delete[] local_output;
    cap.release();
    writer_blur.release();
    writer_edges.release();

    return 0;
}
