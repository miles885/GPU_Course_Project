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
 * Converts pixel data to grayscale or HSV and 
 * applies a filter using the CPU or GPU
 *
 * @param imageWidth     The width of the image to write
 * @param imageHeight    The height of the image to write
 * @param bitsPerPixel   The bits per pixel of the image to write
 * @param pixelData      The channel-separated pixel data
 * @param filter         The type of image filter to use
 * @param outputFilename The output file name
 * @param outputFormat   The output file format
 * @param pixelType      The type of pixel data to apply the filter to
 * @param useCPU         Flag denoting whether to use the CPU or GPU
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t applyFilter(uint32_t imageWidth,
                    uint32_t imageHeight,
                    uint32_t bitsPerPixel,
                    const BYTE * pixelData, 
                    ImageFilter filter,
                    const char * outputFilename, 
                    const FREE_IMAGE_FORMAT & outputFormat, 
                    PixelType pixelType, 
                    bool useCPU)
{
    uint32_t imageSize = imageWidth * imageHeight;

    // Allocate grayscale pixel memory
    BYTE * inputPixelData;
    BYTE * outputPixelData;
    
    if(useCPU)
    {
        inputPixelData = new BYTE[imageSize];
        outputPixelData = new BYTE[imageSize];
    }
    else
    {
        checkCudaErrors(cudaMallocHost((void **) &inputPixelData, sizeof(BYTE) * imageSize, cudaHostAllocDefault));
        checkCudaErrors(cudaMallocHost((void **) &outputPixelData, sizeof(BYTE) * imageSize, cudaHostAllocDefault));
    }

    
    int32_t status = 0;

    // Check to see if converting RGB pixel data to grayscale
    if(pixelType == GRAY)
    {
        status = rgbToGray(imageWidth, imageHeight, bitsPerPixel, pixelData, inputPixelData);
    }
    // Converting RGB pixel data to HSV
    else
    {
        status = rgbToHSV(imageWidth, imageHeight, bitsPerPixel, pixelType, pixelData, inputPixelData);
    }

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filter
    status = applyFilter(imageWidth, imageHeight, filter, inputPixelData, outputPixelData, useCPU);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Output results
    status = saveImage(outputFilename, outputFormat, imageWidth, imageHeight, 8, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Cleanup
    if(useCPU)
    {
        delete [] inputPixelData;
        delete [] outputPixelData;
    }
    else
    {
        checkCudaErrors(cudaFreeHost(inputPixelData));
        checkCudaErrors(cudaFreeHost(outputPixelData));
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
    /**************************************************************************
     * Parse command-line arguments
     **************************************************************************/
    std::string fileName;

    int32_t status = parseCmdArgs(argc, argv, fileName);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Print device settings
     **************************************************************************/
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

    printf("***** Device Settings *****\n");
    printf("Global memory: %zu\n", prop.totalGlobalMem);
    printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Total constant memory: %zu\n", prop.totalConstMem);
    printf("Registers per block: %zu\n\n", prop.regsPerBlock);

    /**************************************************************************
     * Load the image data
     **************************************************************************/
    FREE_IMAGE_FORMAT format;
    FIBITMAP * bitmap = NULL;

    status = loadImage(fileName, format, &bitmap);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Get the image info
     **************************************************************************/
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t bitsPerPixel;

    printf("***** Image Info *****\n");

    status = getImageInfo(&bitmap, imageWidth, imageHeight, bitsPerPixel);

    if(status == EXIT_FAILURE)
    {
        FreeImage_Unload(bitmap);
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Load the pixel data
     **************************************************************************/
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

    /**************************************************************************
     * Output image data before filters are applied
     **************************************************************************/
    // Output grayscale image
    BYTE * outputPixelData = new BYTE[imageWidth * imageHeight];

    status = rgbToGray(imageWidth, imageHeight, bitsPerPixel, pixelData, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    status = saveImage("output_grayscale.png", format, imageWidth, imageHeight, 8, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Output hue image
    status = rgbToHSV(imageWidth, imageHeight, bitsPerPixel, HUE, pixelData, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    status = saveImage("output_hue.png", format, imageWidth, imageHeight, 8, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Output saturation image
    status = rgbToHSV(imageWidth, imageHeight, bitsPerPixel, SATURATION, pixelData, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    status = saveImage("output_saturation.png", format, imageWidth, imageHeight, 8, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Output saturation image
    status = rgbToHSV(imageWidth, imageHeight, bitsPerPixel, VALUE, pixelData, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    status = saveImage("output_value.png", format, imageWidth, imageHeight, 8, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    delete [] outputPixelData;

    /**************************************************************************
     * Apply filters to RGB pixel values on CPU
     **************************************************************************/
    // Apply Roberts filter
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_gray_output_CPU.png", format, GRAY, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_gray_output_CPU.png", format, GRAY, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_gray_output_CPU.png", format, GRAY, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Apply filters to RGB pixel values on GPU
     **************************************************************************/
    // Apply Roberts filter
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_gray_output_GPU.png", format, GRAY, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_gray_output_GPU.png", format, GRAY, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_gray_output_GPU.png", format, GRAY, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Apply filters to HSV channels on CPU
     **************************************************************************/
    // Apply Roberts filter to hue channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_hue_output_CPU.png", format, HUE, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Roberts filter to saturation channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_sat_output_CPU.png", format, SATURATION, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Roberts filter to value channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_value_output_CPU.png", format, VALUE, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter to hue channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_hue_output_CPU.png", format, HUE, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter to saturation channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_sat_output_CPU.png", format, SATURATION, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter to value channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_value_output_CPU.png", format, VALUE, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter to hue channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_hue_output_CPU.png", format, HUE, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter to saturation channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_sat_output_CPU.png", format, SATURATION, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter to value channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_value_output_CPU.png", format, VALUE, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Apply filters to HSV channels on GPU
     **************************************************************************/
    // Apply Roberts filter to hue channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_hue_output_GPU.png", format, HUE, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Roberts filter to saturation channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_sat_output_GPU.png", format, SATURATION, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Roberts filter to value channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, ROBERTS, "roberts_value_output_GPU.png", format, VALUE, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter to hue channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_hue_output_GPU.png", format, HUE, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter to saturation channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_sat_output_GPU.png", format, SATURATION, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter to value channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, SOBEL, "sobel_value_output_GPU.png", format, VALUE, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter to hue channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_hue_output_GPU.png", format, HUE, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter to saturation channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_sat_output_GPU.png", format, SATURATION, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter to value channel
    status = applyFilter(imageWidth, imageHeight, bitsPerPixel, pixelData, PREWITT, "prewitt_value_output_GPU.png", format, VALUE, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /*
     * Cleanup
     */
    FreeImage_Unload(bitmap);

    delete [] pixelData;

    return EXIT_SUCCESS;
}