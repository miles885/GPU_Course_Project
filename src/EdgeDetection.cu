#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "ImageProc.h"
#include "ImageUtils.h"

/**
 * Parses the command-line arguments
 *
 * @param argc The number of command-line arguments
 * @param argv The command-line arguments
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t parseCmdArgs(int32_t argc, char ** argv, std::string & fileName)
{
    if(argc != 2)
    {
        printf("Usage:\n");
        printf("edge_detection.exe fileName\n");

        return EXIT_FAILURE;
    }
    else
    {
        fileName = std::string(argv[1]);
    }

    return EXIT_SUCCESS;
}

/**
 * Converts pixel data to grayscale and applies a filter using the CPU or GPU
 *
 * @param format      The file format
 * @param imageWidth  The width of the image to write
 * @param imageHeight The height of the image to write
 * @param filter      The type of image filter to use
 * @param pixelData   The channel-separated pixel data
 * @param useCPU      Flag denoting whether to use the CPU or GPU
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t applyFilterGray(const FREE_IMAGE_FORMAT & format,
                        uint32_t imageWidth,
                        uint32_t imageHeight,
                        uint32_t bitsPerPixel,
                        ImageFilter filter, 
                        const BYTE * pixelData, 
                        bool useCPU)
{
    uint32_t imageSize = imageWidth * imageHeight;

    // Allocate grayscale pixel memory
    BYTE * grayPixelData;
    
    if(useCPU)
    {
        grayPixelData = new BYTE[imageSize];
    }
    else
    {
        checkCudaErrors(cudaMallocHost((void **) &grayPixelData, sizeof(BYTE) * imageSize, cudaHostAllocDefault));
    }

    // Convert RGB pixel data to grayscale
    int32_t status = rgbToGray(imageWidth, imageHeight, bitsPerPixel, pixelData, grayPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filter
    status = applyFilter(format, imageWidth, imageHeight, filter, grayPixelData, useCPU);

    // Cleanup
    if(useCPU)
    {
        delete [] grayPixelData;
    }
    else
    {
        checkCudaErrors(cudaFreeHost(grayPixelData));
    }

    return EXIT_SUCCESS;
}

/**
 * Entry point to the application
 *
 * @param argc The number of command-line arguments
 * @param argv The command-line arguments
 *
 * @return Exit code indicating success or failure
 */
int32_t main(int32_t argc, char ** argv)
{
    /*
     * Parse command-line arguments
     */
    std::string fileName;

    int32_t status = parseCmdArgs(argc, argv, fileName);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /*
     * Load the image data
     */
    FREE_IMAGE_FORMAT format;
    FIBITMAP * bitmap = NULL;

    status = loadImage(fileName, format, &bitmap);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /*
     * Get the image info
     */
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t bitsPerPixel;

    status = getImageInfo(&bitmap, imageWidth, imageHeight, bitsPerPixel);

    if(status == EXIT_FAILURE)
    {
        FreeImage_Unload(bitmap);
        return EXIT_FAILURE;
    }

    /*
     * Load the pixel data
     */
    BYTE * pixelData;
    
    // Check if 8 or 24 bits per pixel
    if(bitsPerPixel == 8 || bitsPerPixel == 24)
    {
        pixelData = new BYTE[imageWidth * imageHeight * (bitsPerPixel / 8)];
    }
    // Check if 32 bits per pixel
    else if(bitsPerPixel == 32)
    {
        pixelData = new BYTE[imageWidth * imageHeight * 3];
    }
    // Unsupported pixel format
    else
    {
        std::cerr << "Unsupported pixel format!" << std::endl;
        
        return EXIT_FAILURE;
    }

    status = loadPixelData(&bitmap, imageWidth, imageHeight, bitsPerPixel, pixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /*
     * Apply Sobel filter to RGB pixel values
     */
    // Apply Sobel filter using CPU
    status = applyFilterGray(format, imageWidth, imageHeight, bitsPerPixel, SOBEL, pixelData, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter using GPU

    /*
     * Apply Sobel filter to HSV channels
     */
    // Apply Sobel filter using CPU

    // Apply Sobel filter using GPU

    /*
     * Cleanup
     */
    FreeImage_Unload(bitmap);

    delete [] pixelData;

    return EXIT_SUCCESS;
}