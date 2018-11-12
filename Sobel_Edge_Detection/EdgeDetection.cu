#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
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
        return EXIT_FAILURE;
    }

    /*
     * Load the pixel data
     */
    BYTE * pixelData = new BYTE[imageWidth * imageHeight * (bitsPerPixel / 8)];

    status = loadPixelData(&bitmap, imageWidth, imageHeight, bitsPerPixel, pixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /*
     * Apply sobel filter to grayscale image
     */
    for(uint32_t y = 0; y < imageHeight; y++)
    {
        for(uint32_t x = 0; x < imageWidth; x++)
        {
            //pixelData[(y * imageHeight) + x] = max(0, pixelData[(y * imageHeight) + x] - 50);
        }
    }

    std::string outputFileName = "test.png";

    status = saveImage(outputFileName, format, imageWidth, imageHeight, bitsPerPixel, pixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /*
     * Apply sobel filter to hue channel of HSV image
     */

    /*
     * Cleanup
     */
    FreeImage_Unload(bitmap);

    delete [] pixelData;
}