    // Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <chrono>

#define NUM_CHANNELS 1
#define BLOCK_SIZE 512

// CUDA kernel for parallel reduction to find the minimum and maximum values in the image
__global__ void min_max_reduction(uint8_t* img, int* min_max, int size) {
    __shared__ int sdata[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    int min_val = 255;
    int max_val = 0;

    while (i < size) {
        int val = img[i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);

        if (i + blockDim.x < size) {
            int val2 = img[i + blockDim.x];
            min_val = min(min_val, val2);
            max_val = max(max_val, val2);
        }
        i += gridSize;
    }

    sdata[tid] = min_val;
    sdata[tid + blockDim.x] = max_val;

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
            sdata[tid + blockDim.x] = max(sdata[tid + blockDim.x], sdata[tid + blockDim.x + s]);
        }
        __syncthreads();
    }

    // Write the results of this block to global memory
    if (tid == 0) {
        min_max[blockIdx.x] = sdata[0];
        min_max[blockIdx.x + gridDim.x] = sdata[blockDim.x];
    }
}

// CUDA kernel to subtract a value from all pixels in the image
__global__ void sub_kernel(uint8_t* img, uint8_t sub_value, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        img[i] -= sub_value;
    }
}

// CUDA kernel to scale pixel values in the image
__global__ void scale_kernel(uint8_t* img, float constant, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        img[i] = img[i] * constant;
    }
}

// CPU function to find the minimum and maximum values in the image
void find_min_max_cpu(const uint8_t* img, int size, int& min_val, int& max_val) {
    min_val = 255;
    max_val = 0;

    for (int i = 0; i < size; ++i) {
        int val = img[i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
}

// CPU function to subtract a value from all pixels in the image and scale them
void process_image_cpu(uint8_t* img, int size, uint8_t sub_value, float scale_constant) {
    for (int i = 0; i < size; ++i) {
        img[i] = static_cast<uint8_t>((img[i] - sub_value) * scale_constant);
    }
}

// Dummy kernel to warm up the GPU
__global__ void dummy_kernel() {}

int main() {
    // List of image files to be processed
    const char* images[] = {
        "./samples/640x426.bmp",
        "./samples/1280x843.bmp",
        "./samples/1920x1280.bmp",
        "./samples/5184x3456.bmp",
    };

    // Warm up the GPU
    dummy_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    for (int k = 0; k < 4; ++k) {
        // Load image
        int width, height, channels;
        uint8_t* image = stbi_load(images[k], &width, &height, &channels, STBI_grey);

        if (!image) {
            printf("Failed to load image: %s\n", images[k]);
            continue;
        }

        int image_size = width * height;

        // Benchmark CPU implementation
        auto start_cpu = std::chrono::steady_clock::now();

        // CPU implementation to find min and max values
        int min_cpu, max_cpu;
        find_min_max_cpu(image, image_size, min_cpu, max_cpu);

        // Calculate scale constant
        float scale_constant_cpu = 255.0f / (max_cpu - min_cpu);

        // Subtract min value and scale pixels
        process_image_cpu(image, image_size, min_cpu, scale_constant_cpu);

        auto end_cpu = std::chrono::steady_clock::now();
        std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        printf("CPU Time for image %d: %f seconds\n", k + 1, cpu_duration.count());

        // Allocate memory on the GPU for image data and other variables
        uint8_t* d_image;
        int* d_min_max;
        cudaMalloc(&d_image, image_size * sizeof(uint8_t));
        cudaMalloc(&d_min_max, 2 * sizeof(int));

        // Copy image data from host to device
        cudaMemcpy(d_image, image, image_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

        // Benchmark GPU implementation
        auto start_gpu = std::chrono::steady_clock::now();

        // GPU implementation: find min-max, subtract, and scale
        int grid_size = (image_size + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);
        min_max_reduction << <grid_size, BLOCK_SIZE >> > (d_image, d_min_max, image_size);

        // Calculate scale constant on GPU
        int min_max_host[2];
        cudaMemcpy(min_max_host, d_min_max, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        float scale_constant_gpu = 255.0f / (min_max_host[1] - min_max_host[0]);

        // Subtract min value and scale pixels on GPU
        sub_kernel << <grid_size, BLOCK_SIZE >> > (d_image, min_max_host[0], image_size);
        scale_kernel << <grid_size, BLOCK_SIZE >> > (d_image, scale_constant_gpu, image_size);

        auto end_gpu = std::chrono::steady_clock::now();
        std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
        printf("GPU Time for image %d: %f seconds\n", k + 1, gpu_duration.count());

        // Copy processed image data from device to host
        cudaMemcpy(image, d_image, image_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // Write processed image data to a BMP file
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "./out_img_%d.bmp", k + 1);
        stbi_write_bmp(output_file, width, height, 1, image);

        // Free memory on the GPU and host
        cudaFree(d_image);
        cudaFree(d_min_max);
        stbi_image_free(image);
    }

    return 0;
}
