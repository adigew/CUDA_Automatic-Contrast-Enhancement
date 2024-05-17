// Necessary preprocessor directives for stb_image
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define NUM_CHANNELS 1

// CPU function for finding minimum and maximum pixel values in an image
void min_max_of_img_host(uint8_t* img, uint8_t* min, uint8_t* max, int width, int height) {
    int max_tmp = 0;
    int min_tmp = 255;
    for (int n = 0; n < width * height; n++) {
        max_tmp = (img[n] > max_tmp) ? img[n] : max_tmp;
        min_tmp = (img[n] < min_tmp) ? img[n] : min_tmp;
    }
    *max = max_tmp;
    *min = min_tmp;
}

// CPU function to subtract a value from all pixels in an image
void sub_host(uint8_t* img, uint8_t sub_value, int width, int height) {
    for (int n = 0; n < width * height; n++) {
        img[n] -= sub_value;
    }
}

// CPU function to scale pixel values given a constant
void scale_host(uint8_t* img, float constant, int width, int height) {
    for (int n = 0; n < width * height; n++) {
        img[n] = img[n] * constant; // Note the implicit type conversion
    }
}

int main() {
    // List of image file paths to process
    const char* images[] = {
        "./samples/640x426.bmp",
        "./samples/1280x843.bmp",
        "./samples/1920x1280.bmp",
        "./samples/5184x3456.bmp"
    };

    // Loop through each image path
    for (int i = 0; i < sizeof(images) / sizeof(images[0]); ++i) {
        const char* image_path = images[i];
        int width; //image width
        int height; //image height
        int bpp;  //bytes per pixel if the image was RGB (not used)

        uint8_t min_host, max_host;

        // Load the image file into memory
        uint8_t* image = stbi_load(image_path, &width, &height, &bpp, NUM_CHANNELS);

        // Print sanity check information for the loaded image
        printf("Bytes per pixel: %d \n", bpp / 3);
        printf("Height: %d \n", height);
        printf("Width: %d \n", width);

        // Start measuring time for the current image
        clock_t start = clock();

        // Find the minimum and maximum pixel values
        min_max_of_img_host(image, &min_host, &max_host, width, height);

        // Calculate the scaling factor for contrast enhancement
        float scale_constant = 255.0f / (max_host - min_host);

        // Subtract the minimum pixel value from all pixels
        sub_host(image, min_host, width, height);

        // Scale all pixel values based on the calculated factor
        scale_host(image, scale_constant, width, height);

        // Stop measuring time and calculate the elapsed time
        clock_t end = clock();
        double elapsed_time_ms = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;

        // Print the elapsed time for the current image
        printf("Time elapsed for processing %s: %.2f ms\n\n", image_path, elapsed_time_ms);

        // Save the processed image to a BMP file
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "./out_%s.bmp", image_path);
        stbi_write_bmp(output_file, width, height, 1, image);

        // Deallocate memory for the current image
        stbi_image_free(image);
    }

    return 0;
}
