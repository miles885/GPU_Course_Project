#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "ImageUtils.h"
#include "Sobel.h"

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
 * Apply a Sobel filter to grayscale values
 *
 * @param format       The file format
 * @param imageWidth   The width of the image to write
 * @param imageHeight  The height of the image to write
 * @param bitsPerPixel The bits per pixel of the image to write
 * @param pixelData    The channel separated RGB pixel data
 * @param useCPU       Flag denoting whether to use CPU or GPU
 *
 * @return Flag denoting success or failure
 */
int32_t applySobelFilterGrayscale(const FREE_IMAGE_FORMAT & format, 
                                  uint32_t imageWidth, 
                                  uint32_t imageHeight, 
                                  uint32_t bitsPerPixel, 
                                  const BYTE * pixelData, 
                                  bool useCPU)
{
    // Convert pixel data to grayscale
    uint32_t imageSize = imageWidth * imageHeight;

    BYTE * h_grayPixelData = new BYTE[imageSize];
    BYTE * h_outputPixelData = new BYTE[imageSize];

    if(bitsPerPixel == 8)
    {
        memcpy(h_grayPixelData, pixelData, imageWidth * imageHeight * sizeof(BYTE));
    }
    else
    {
        for(uint32_t y = 0; y < imageHeight; y++)
        {
            for(uint32_t x = 0; x < imageWidth; x++)
            {
                BYTE r = pixelData[(y * imageWidth) + x];
                BYTE g = pixelData[(y * imageWidth) + imageSize + x];
                BYTE b = pixelData[(y * imageWidth) + (imageSize * 2) + x];

                h_grayPixelData[(y * imageWidth) + x] = (r + g + b) / 3;
            }
        }
    }

    // Apply Sobel filter using CPU
    if(useCPU)
    {
        applySobelFilterCPU(imageWidth, imageHeight, h_grayPixelData, h_outputPixelData);
    }
    // Apply Sobel filter using GPU
    else
    {
        //TODO: Setup device memory
        //TODO: Execute kernel
        //TODO: Cleanup device memory
    }

    // Output results
    std::string outputFileName = "grayscale_output.png";

    int32_t status = saveImage(outputFileName, format, imageWidth, imageHeight, 8, h_outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Cleanup
    delete [] h_grayPixelData;
    delete [] h_outputPixelData;

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
     * Apply Sobel filter to grayscale values
     */
    // Apply Sobel filter using CPU
    status = applySobelFilterGrayscale(format, imageWidth, imageHeight, bitsPerPixel, pixelData, true);

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