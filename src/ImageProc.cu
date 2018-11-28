#include "ImageProc.h"

#include <chrono>
#include <iostream>

#include "helper_cuda.h"

//NOTE: A pseudocode implementation of the sobel algorithm was
//      found at https://en.wikipedia.org/wiki/Sobel_operator
//      under the "Pseduocode implementation" section

/**
 * Creates a CUDA event at the current time
 *
 * @param None
 *
 * @return time The cuda event for the current time
 */
__host__
cudaEvent_t getTime()
{
    cudaEvent_t time;

    cudaEventCreate(&time);
    cudaEventRecord(time);

    return time;
}

/**
 * Applies a filter to the pixels located 
 * around the specified x and y coordinate
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param filterX         The x dimension filter data
 * @param filterY         The y dimension filter data
 * @param x               The x coordinate of the pixel
 * @param y               The y coordinate of the pixel
 * @param pixelData       The input array of pixel data
 * @param outputPixelData The output array of pixel data
 *
 * @return None
 */
__host__
__device__
void applyFilter(uint32_t imageWidth, 
                 uint32_t imageHeight, 
                 const int32_t * filterX, 
                 const int32_t * filterY, 
                 int32_t x, 
                 int32_t y, 
                 const BYTE * pixelData, 
                 BYTE * outputPixelData)
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
            applyFilter(imageWidth, imageHeight, filterX, filterY, x, y, pixelData, outputPixelData);
        }
    }
}

/**
 * Creates an image with highlighted edges by applying
 * a filter to the pixel data using global memory
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
void applyFilterGlobalGPU(uint32_t imageWidth, 
                          uint32_t imageHeight, 
                          const int32_t * filterX, 
                          const int32_t * filterY, 
                          const BYTE * pixelData, 
                          BYTE * outputPixelData)
{
    // Retrieve the thread index
    const uint32_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Check to make sure not at edge of image
    if(x > 0 && x < imageWidth && y > 0 && y < imageHeight)
    {
        // Apply the filter
        applyFilter(imageWidth, imageHeight, filterX, filterY, x, y, pixelData, outputPixelData);
    }
}

/**
* Creates an image with highlighted edges by applying
* a filter to the pixel data using shared memory
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
void applyFilterSharedGPU(uint32_t imageWidth,
                          uint32_t imageHeight,
                          const int32_t * filterX, 
                          const int32_t * filterY, 
                          const BYTE * pixelData, 
                          BYTE * outputPixelData)
{
    // Retrieve the thread index
    const uint32_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Put filter data into shared memory
    __shared__ int32_t shFilterX[FILTER_SIZE];
    __shared__ int32_t shFilterY[FILTER_SIZE];

    if(threadIdx.x < FILTER_SIZE)
    {
        shFilterX[threadIdx.x] = filterX[threadIdx.x];
        shFilterY[threadIdx.x] = filterY[threadIdx.x];
    }

    __syncthreads();

    // Check to make sure not at edge of image
    if(x > 0 && x < imageWidth && y > 0 && y < imageHeight)
    {
        // Apply the filter
        applyFilter(imageWidth, imageHeight, shFilterX, shFilterY, x, y, pixelData, outputPixelData);
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
 * @param useGlobalMem    Flag denoting whether to use global or shared memory on the GPU
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t filterImage(uint32_t imageWidth,
                    uint32_t imageHeight, 
                    ImageFilter filter, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData, 
                    bool useCPU, 
                    bool useGlobalMem)
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
            return EXIT_FAILURE;
    }

    // Check to see if using the CPU
    if(useCPU)
    {
        // Start a timer
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        // Apply filter
        applyFilterCPU(imageWidth, imageHeight, h_filterX, h_filterY, pixelData, outputPixelData);

        // Stop the timer and output the duration
        std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = stop - start;

        std::cout << " ellapsed time (ms): " << dur.count() * 1e3 << std::endl;
    }
    // Using the GPU
    else
    {
        uint32_t imageSizeBytes = sizeof(BYTE) * imageSize;

        // Capture the start time
        cudaEvent_t startTime = getTime();

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
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize(ceil((double) imageWidth / blockSize.x), ceil((double) imageHeight / blockSize.y));

        if(useGlobalMem)
        {
            applyFilterGlobalGPU<<<gridSize, blockSize>>>(imageWidth, imageHeight, d_filterX, d_filterY, d_pixelData, d_outputPixelData);
        }
        else
        {
            applyFilterSharedGPU<<<gridSize, blockSize>>>(imageWidth, imageHeight, d_filterX, d_filterY, d_pixelData, d_outputPixelData);
        }

        getLastCudaError("");

        // Copy device memory to host
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(outputPixelData, d_outputPixelData, imageSizeBytes, cudaMemcpyDeviceToHost));

        // Capture the end time
        cudaEvent_t stopTime = getTime();
        cudaEventSynchronize(stopTime);

        // Print the ellapsed time
        float ellapsedTime = 0;

        cudaEventElapsedTime(&ellapsedTime, startTime, stopTime);

        std::cout << " ellapsed time (ms): " << ellapsedTime << std::endl;

        // Cleanup
        checkCudaErrors(cudaFree(d_filterX));
        checkCudaErrors(cudaFree(d_filterY));

        checkCudaErrors(cudaFree(d_pixelData));
        checkCudaErrors(cudaFree(d_outputPixelData));
    }

    return EXIT_SUCCESS;
 }