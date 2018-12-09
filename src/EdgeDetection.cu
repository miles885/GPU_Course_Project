#include <algorithm>
#include <iomanip>
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
 * Outputs the pixel data before filters are applied
 *
 * @param imageWidth     The width of the image to write
 * @param imageHeight    The height of the image to write
 * @param bitsPerPixel   The bits per pixel of the image to write
 * @param pixelData      The channel-separated pixel data
 * @param outputFormat   The output file format
 *
 * @return Flag denoting success or failure
 */
int32_t outputPixelData(uint32_t imageWidth, 
                        uint32_t imageHeight, 
                        uint32_t bitsPerPixel, 
                        BYTE * pixelData, 
                        const FREE_IMAGE_FORMAT & outputFormat)
{
    // Output grayscale image
    BYTE * outputPixelData = new BYTE[imageWidth * imageHeight];

    int32_t status = rgbToGray(imageWidth, imageHeight, bitsPerPixel, pixelData, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    status = saveImage("output_grayscale.png", outputFormat, imageWidth, imageHeight, 8, outputPixelData);

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

    status = saveImage("output_hue.png", outputFormat, imageWidth, imageHeight, 8, outputPixelData);

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

    status = saveImage("output_saturation.png", outputFormat, imageWidth, imageHeight, 8, outputPixelData);

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

    status = saveImage("output_value.png", outputFormat, imageWidth, imageHeight, 8, outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    delete[] outputPixelData;

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
 * @param useGlobalMem   Flag denoting whether to use global or shared memory on the GPU
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t convAndFilterImage(uint32_t imageWidth,
                           uint32_t imageHeight,
                           uint32_t bitsPerPixel,
                           const BYTE * pixelData, 
                           ImageFilter filter,
                           const char * outputFilename, 
                           const FREE_IMAGE_FORMAT & outputFormat, 
                           PixelType pixelType, 
                           bool useCPU, 
                           bool useGlobalMem)
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
    status = filterImage(imageWidth, imageHeight, filter, inputPixelData, outputPixelData, useCPU, useGlobalMem);

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
 * Applies each filter type by converting pixel data to 
 * grayscale or HSV and applies a filter using the CPU or GPU
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
 * @param useGlobalMem   Flag denoting whether to use global or shared memory on the GPU
 *
 * @return Flag denoting success or failure
 */
int32_t applyAllFilters(uint32_t imageWidth,
                        uint32_t imageHeight,
                        uint32_t bitsPerPixel,
                        const BYTE * pixelData, 
                        const FREE_IMAGE_FORMAT & outputFormat, 
                        PixelType pixelType,
                        bool useCPU,
                        bool useGlobalMem)
{
    // Set the output filename
    std::string robertsFilename = "ROBERTS";
    std::string sobelFilename = "SOBEL";
    std::string prewittFilename = "PREWITT";

    switch(pixelType)
    {
        case GRAY:
            robertsFilename += "_GRAY";
            sobelFilename += "_GRAY";
            prewittFilename += "_GRAY";
            break;
        case HUE:
            robertsFilename += "_HUE";
            sobelFilename += "_HUE";
            prewittFilename += "_HUE";
            break;
        case SATURATION:
            robertsFilename += "_SATURATION";
            sobelFilename += "_SATURATION";
            prewittFilename += "_SATURATION";
            break;
        case VALUE:
            robertsFilename += "_VALUE";
            sobelFilename += "_VALUE";
            prewittFilename += "_VALUE";
            break;
        default:
            std::cerr << "Invalid pixel type!" << std::endl;
            return EXIT_FAILURE;
    }

    if(useCPU)
    {
        robertsFilename += "_CPU";
        sobelFilename += "_CPU";
        prewittFilename += "_CPU";
    }
    else
    {
        robertsFilename += "_GPU";
        sobelFilename += "_GPU";
        prewittFilename += "_GPU";

        if(useGlobalMem)
        {
            robertsFilename += "_GLOBAL";
            sobelFilename += "_GLOBAL";
            prewittFilename += "_GLOBAL";
        }
        else
        {
            robertsFilename += "_SHARED";
            sobelFilename += "_SHARED";
            prewittFilename += "_SHARED";
        }
    }

    // Apply Roberts filter
    std::cout << std::setw(29) << robertsFilename;
    robertsFilename += ".png";

    int32_t status = convAndFilterImage(imageWidth, 
                                        imageHeight, 
                                        bitsPerPixel, 
                                        pixelData, 
                                        ROBERTS, 
                                        robertsFilename.c_str(), 
                                        outputFormat, 
                                        pixelType, 
                                        useCPU, 
                                        useGlobalMem);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Sobel filter
    std::cout << std::setw(29) << sobelFilename;
    sobelFilename += ".png";

    status = convAndFilterImage(imageWidth, 
                                imageHeight, 
                                bitsPerPixel, 
                                pixelData, 
                                SOBEL, 
                                sobelFilename.c_str(),
                                outputFormat, 
                                pixelType, 
                                useCPU, 
                                useGlobalMem);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply Prewitt filter
    std::cout << std::setw(29) << prewittFilename;
    prewittFilename += ".png";

    status = convAndFilterImage(imageWidth, 
                                imageHeight, 
                                bitsPerPixel, 
                                pixelData, 
                                PREWITT, 
                                prewittFilename.c_str(),
                                outputFormat, 
                                pixelType, 
                                useCPU, 
                                useGlobalMem);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
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

    std::cout << "***** Image Info *****" << std::endl;

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
    status = outputPixelData(imageWidth, imageHeight, bitsPerPixel, pixelData, format);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    /**************************************************************************
     * Apply filters to RGB pixel values on CPU
     **************************************************************************/
    std::cout << "***** RGB CPU Results *****" << std::endl;

    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, GRAY, true, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    /**************************************************************************
     * Apply filters to RGB pixel values on GPU
     **************************************************************************/
    std::cout << "***** RGB GPU Results *****" << std::endl;

    // Use global memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, GRAY, false, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    // Use shared memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, GRAY, false, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    /**************************************************************************
     * Apply filters to HSV channels on CPU
     **************************************************************************/
    std::cout << "***** HSV CPU Results *****" << std::endl;

    // Apply filters on hue channel
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, HUE, true, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filters on saturation channel
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, SATURATION, true, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filters on value channel
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, VALUE, true, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    /**************************************************************************
     * Apply filters to HSV channels on GPU
     **************************************************************************/
    std::cout << "***** HSV GPU Results *****" << std::endl;

    // Apply filters on hue channel using global memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, HUE, false, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filters on saturation channel using global memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, SATURATION, false, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filters on value channel using global memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, VALUE, false, true);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    // Apply filters on hue channel using shared memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, HUE, false, false);

    if (status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filters on saturation channel using shared memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, SATURATION, false, false);

    if (status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Apply filters on value channel using shared memory
    status = applyAllFilters(imageWidth, imageHeight, bitsPerPixel, pixelData, format, VALUE, false, false);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    /*
     * Cleanup
     */
    FreeImage_Unload(bitmap);

    delete [] pixelData;

    return EXIT_SUCCESS;
}