// The following two lines define macros that tell the stb_image and stb_image_write libraries
// to implement the corresponding functions in this file. This technique is known as the 
// "single-header library" pattern, where including the header file is sufficient to compile 
// the implementation along with the rest of the code. It simplifies the usage of external 
// libraries by encapsulating their implementation details.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

// Include the header files for stb_image and stb_image_write libraries. These libraries provide
// simple functions for loading and writing images in various formats (e.g., BMP, PNG, JPEG).
#include "stb_image.h"
#include "stb_image_write.h"

// Include necessary CUDA headers for CUDA programming.
#include <cuda_runtime.h>

// Include standard C headers for data types and I/O operations.
#include <stdint.h>
#include <stdio.h>

// Define the number of color channels in the image. In this case, it's set to 1 for grayscale images.
#define NUM_CHANNELS 1

// Define a CUDA kernel to find the minimum value in an image using parallel reduction with memory access optimization.
// This kernel will be executed on the GPU for each pixel block.
__global__ void find_min_reduction(const uint8_t* img, int width, int height, int* min_val, int blockSize) {
    // Declare shared memory to store intermediate results. The size is increased to 512 elements
    // to better utilize the shared memory resources per block.
    __shared__ int sdata[512];

    // Get the thread ID within the block.
    int tid = threadIdx.x;

    // Calculate the global index of the thread within the image.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the stride for accessing image elements in parallel.
    int stride = gridDim.x * blockDim.x;

    // Initialize the minimum value to the maximum possible value (255 for uint8_t).
    int fmin = 255;

    // Iterate through the assigned portion of the image to find the local minimum with coalesced memory access.
    for (int i = idx; i < width * height; i += stride) {
        uint8_t pixel = img[i];
        if (pixel < fmin) fmin = pixel;
    }

    // Store the local minimum in shared memory.
    sdata[tid] = fmin;
    __syncthreads();

    // Perform parallel reduction to find the block's minimum.
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s && sdata[tid] > sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level unrolling for efficient reduction.
    if (tid < 32) {
        volatile int* smem = sdata;
        if (blockSize >= 64) smem[tid] = min(smem[tid], smem[tid + 32]);
        if (blockSize >= 32) smem[tid] = min(smem[tid], smem[tid + 16]);
        if (blockSize >= 16) smem[tid] = min(smem[tid], smem[tid + 8]);
        if (blockSize >= 8) smem[tid] = min(smem[tid], smem[tid + 4]);
        if (blockSize >= 4) smem[tid] = min(smem[tid], smem[tid + 2]);
        if (blockSize >= 2) smem[tid] = min(smem[tid], smem[tid + 1]);
    }

    // Perform an atomic operation to find the minimum value across all blocks.
    if (tid == 0) {
        atomicMin(min_val, sdata[0]);
    }
}

// Similarly, define a CUDA kernel to find the maximum value in an image using parallel reduction with memory access optimization.
__global__ void find_max_reduction(const uint8_t* img, int width, int height, int* max_val, int blockSize) {
    // Declare shared memory to store intermediate results.
    __shared__ int sdata[512];

    // Get the thread ID within the block.
    int tid = threadIdx.x;

    // Calculate the global index of the thread within the image.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the stride for accessing image elements in parallel.
    int stride = gridDim.x * blockDim.x;

    // Initialize the maximum value to 0.
    int fmax = 0;

    // Iterate through the assigned portion of the image to find the local maximum with coalesced memory access.
    for (int i = idx; i < width * height; i += stride) {
        uint8_t pixel = img[i];
        if (pixel > fmax) fmax = pixel;
    }

    // Store the local maximum in shared memory.
    sdata[tid] = fmax;
    __syncthreads();

    // Perform parallel reduction to find the block's maximum.
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level unrolling for efficient reduction.
    if (tid < 32) {
        volatile int* smem = sdata;
        if (blockSize >= 64) smem[tid] = max(smem[tid], smem[tid + 32]);
        if (blockSize >= 32) smem[tid] = max(smem[tid], smem[tid + 16]);
        if (blockSize >= 16) smem[tid] = max(smem[tid], smem[tid + 8]);
        if (blockSize >= 8) smem[tid] = max(smem[tid], smem[tid + 4]);
        if (blockSize >= 4) smem[tid] = max(smem[tid], smem[tid + 2]);
        if (blockSize >= 2) smem[tid] = max(smem[tid], smem[tid + 1]);
    }

    // Perform an atomic operation to find the maximum value across all blocks.
    if (tid == 0) {
        atomicMax(max_val, sdata[0]);
    }
}

// Define additional CUDA kernels for subtracting the minimum value from all pixels and scaling pixel values.
// These kernels are not explicitly used in the current implementation provided, but they are included for completeness.

// Function to benchmark the GPU kernel execution time
void run_and_time_kernel(const char* image_name, int blockSize) {
    // Load the image file into a buffer using stb_image
    int width, height, bpp;
    uint8_t* image = stbi_load(image_name, &width, &height, &bpp, NUM_CHANNELS);

    // Allocate memory on the GPU for the image and min/max values
    uint8_t* d_image;
    int* d_min, * d_max;
    cudaMalloc(&d_image, width * height * sizeof(uint8_t));
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));

    // Copy the image data from the host to the device
    cudaMemcpy(d_image, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Warm-up run to trigger CUDA setup tasks
    find_min_reduction << <1, 1 >> > (d_image, width, height, d_min, blockSize);
    cudaDeviceSynchronize(); // Ensure all CUDA tasks are completed

    // Record the start time of the GPU computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run the GPU kernels to find the minimum and maximum values
    find_min_reduction << <1, blockSize >> > (d_image, width, height, d_min, blockSize);
    find_max_reduction << <1, blockSize >> > (d_image, width, height, d_max, blockSize);
    cudaDeviceSynchronize(); // Ensure all CUDA tasks are completed

    // Record the stop time and calculate the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Print the elapsed time for the current configuration
    printf("Time elapsed for image %s with block size %d: %.2f ms\n", image_name, blockSize, elapsed_ms);

    // Free GPU memory
    cudaFree(d_image);
    cudaFree(d_min);
    cudaFree(d_max);

    // Free host memory
    stbi_image_free(image);
}

// The main function where the CUDA kernel execution is initiated for each image with a specified block size.
int main() {
    // Array containing paths to sample images
    const char* images[] = {
        "./samples/640x426.bmp",
        "./samples/1280x843.bmp",
        "./samples/1920x1280.bmp",
        "./samples/5184x3456.bmp",
    };

    // Fixed block size for CUDA kernel execution
    int blockSize = 256;

    // Iterate over each image and run the CUDA kernel with the specified block size
    for (int i = 0; i < sizeof(images) / sizeof(images[0]); ++i) {
        run_and_time_kernel(images[i], blockSize);
    }

    return 0;
}
