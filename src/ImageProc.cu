#include "ImageProc.h"

#include <iostream>

#include "helper_cuda.h"

//NOTE: A pseudocode implementation of the sobel algorithm was
//      found at https://en.wikipedia.org/wiki/Sobel_operator
//      under the "Pseduocode implementation" section

/**
 * Creates an image with highlighted edges by
 * applying a filter to the pixel data
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param filterX         The x dimension filter data
 * @param filterY         The y dimension filter data
 * @param pixelData       The input array of pixel data
 * @param outputPixelData The output array of pixel data
 *
 * @return None
 */
 __host__
void applyFilterCPU(uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    const int32_t * filterX, 
                    const int32_t * filterY, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData)
{
    for(uint32_t y = 1; y < (imageHeight - 1); y++)
    {
        for(uint32_t x = 1; x < (imageWidth - 1); x++)
        {
            uint32_t topRowOffset = ((y - 1) * imageWidth) + x;
            uint32_t midRowOffset = (y * imageWidth) + x;
            uint32_t botRowOffset = ((y + 1) * imageWidth) + x;

            int32_t topRow[3] = {pixelData[topRowOffset - 1], pixelData[topRowOffset], pixelData[topRowOffset + 1]};
            int32_t midRow[3] = {pixelData[midRowOffset - 1], pixelData[midRowOffset], pixelData[midRowOffset + 1]};
            int32_t botRow[3] = {pixelData[botRowOffset - 1], pixelData[botRowOffset], pixelData[botRowOffset + 1]};

            int32_t pixelX = (filterX[0] * topRow[0]) + (filterX[1] * topRow[1]) + (filterX[2] * topRow[2]) +
                             (filterX[3] * midRow[0]) + (filterX[4] * midRow[1]) + (filterX[5] * midRow[2]) +
                             (filterX[6] * botRow[0]) + (filterX[7] * botRow[1]) + (filterX[8] * botRow[2]);
            
            int32_t pixelY = (filterY[0] * topRow[0]) + (filterY[1] * topRow[1]) + (filterY[2] * topRow[2]) +
                             (filterY[3] * midRow[0]) + (filterY[4] * midRow[1]) + (filterY[5] * midRow[2]) +
                             (filterY[6] * botRow[0]) + (filterY[7] * botRow[1]) + (filterY[8] * botRow[2]);
            
            // Calculate magnitude
            int32_t mag = sqrt((pixelX * pixelX) + (pixelY * pixelY));

            // Set output pixel value
            //TODO: Use some pixel threshold for better results?
            outputPixelData[(y * imageWidth) + x] = mag;
        }
    }
}

/**
 * Creates an image with highlighted edges by
 * applying a filter to the pixel data
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param filterX         The x dimension filter data
 * @param filterY         The y dimension filter data
 * @param pixelData       The input array of pixel data
 * @param outputPixelData The output array of pixel data
 *
 * @return None
 */
__global__
void applyFilterGPU(uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    const int32_t * filterX, 
                    const int32_t * filterY, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData)
{
    // Retrieve the thread index
    const uint32_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

    //TODO: Put filter data into shared memory

    // Check to make sure not at edge of image
    if(x > 0 && x < imageWidth && y > 0 && y < imageHeight)
    {
        uint32_t topRowOffset = ((y - 1) * imageWidth) + x;
        uint32_t midRowOffset = (y * imageWidth) + x;
        uint32_t botRowOffset = ((y + 1) * imageWidth) + x;
        
        int32_t topRow[3] = {pixelData[topRowOffset - 1], pixelData[topRowOffset], pixelData[topRowOffset + 1]};
        int32_t midRow[3] = {pixelData[midRowOffset - 1], pixelData[midRowOffset], pixelData[midRowOffset + 1]};
        int32_t botRow[3] = {pixelData[botRowOffset - 1], pixelData[botRowOffset], pixelData[botRowOffset + 1]};

        int32_t pixelX = (filterX[0] * topRow[0]) + (filterX[1] * topRow[1]) + (filterX[2] * topRow[2]) +
                         (filterX[3] * midRow[0]) + (filterX[4] * midRow[1]) + (filterX[5] * midRow[2]) +
                         (filterX[6] * botRow[0]) + (filterX[7] * botRow[1]) + (filterX[8] * botRow[2]);
        
        int32_t pixelY = (filterY[0] * topRow[0]) + (filterY[1] * topRow[1]) + (filterY[2] * topRow[2]) +
                         (filterY[3] * midRow[0]) + (filterY[4] * midRow[1]) + (filterY[5] * midRow[2]) +
                         (filterY[6] * botRow[0]) + (filterY[7] * botRow[1]) + (filterY[8] * botRow[2]);
        
        // Calculate magnitude (must use float version with Cuda)
        int32_t mag = sqrt((float) (pixelX * pixelX) + (float) (pixelY * pixelY));

        // Set output pixel value
        //TODO: Use some pixel threshold for better results?
        outputPixelData[(y * imageWidth) + x] = mag;
    }
}

/**
 * Apply a filter to a set of pixel values using the CPU or GPU
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param filter          The type of image filter to use
 * @param pixelData       The single channel pixel data
 * @param outputPixelData The filtered pixel data
 * @param useCPU          Flag denoting whether to use the CPU or GPU
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t applyFilter(uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    ImageFilter filter, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData, 
                    bool useCPU)
{
    uint32_t imageSize = imageWidth * imageHeight;

    // Host memory
    const int32_t * h_filterX;
    const int32_t * h_filterY;

    // Set filter data
    switch(filter)
    {
        case ROBERTS:
            h_filterX = ROBERTS_X;
            h_filterY = ROBERTS_Y;

            break;
        case SOBEL:
            h_filterX = SOBEL_X;
            h_filterY = SOBEL_Y;

            break;
        case PREWITT:
            h_filterX = PREWITT_X;
            h_filterY = PREWITT_Y;

            break;
        default:
            std::cerr << "Invalid filter type!" << std::endl;
            break;
    }

    // Check to see if using the CPU
    if(useCPU)
    {
        // Apply filter
        applyFilterCPU(imageWidth, imageHeight, h_filterX, h_filterY, pixelData, outputPixelData);
    }
    // Using the GPU
    else
    {
        uint32_t imageSizeBytes = sizeof(BYTE) * imageSize;

        // Allocate device memory
        int32_t * d_filterX;
        int32_t * d_filterY;

        BYTE * d_pixelData;
        BYTE * d_outputPixelData;

        checkCudaErrors(cudaMalloc((void **) &d_filterX, sizeof(int32_t) * 9));
        checkCudaErrors(cudaMalloc((void **) &d_filterY, sizeof(int32_t) * 9));

        checkCudaErrors(cudaMalloc((void **) &d_pixelData, imageSizeBytes));
        checkCudaErrors(cudaMalloc((void **) &d_outputPixelData, imageSizeBytes));

        // Copy host memory to device
        checkCudaErrors(cudaMemcpy(d_filterX, h_filterX, sizeof(int32_t) * 9, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_filterY, h_filterY, sizeof(int32_t) * 9, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(d_pixelData, pixelData, imageSizeBytes, cudaMemcpyHostToDevice));

        // Apply filter (kernel)
        dim3 blockSize(16, 16);
        dim3 gridSize(ceil((double) imageWidth / blockSize.x), ceil((double) imageHeight / blockSize.y));

        applyFilterGPU<<<gridSize, blockSize>>>(imageWidth, imageHeight, d_filterX, d_filterY, d_pixelData, d_outputPixelData);
        getLastCudaError("");

        // Copy device memory to host
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(outputPixelData, d_outputPixelData, imageSizeBytes, cudaMemcpyDeviceToHost));

        // Cleanup
        checkCudaErrors(cudaFree(d_pixelData));
        checkCudaErrors(cudaFree(d_outputPixelData));
    }

    return EXIT_SUCCESS;
 }