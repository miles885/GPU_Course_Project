#include "ImageProc.h"

#include "helper_cuda.h"

//NOTE: The sobel algorithm and kernels were found on
//      https://en.wikipedia.org/wiki/Sobel_operator
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
    for(uint32_t y = 1; y <= (imageHeight - 2); y++)
    {
        for(uint32_t x = 1; x <= (imageWidth - 2); x++)
        {
            int32_t topRow[3] = {pixelData[((y - 1) * imageWidth) + (x - 1)], pixelData[((y - 1) * imageWidth) + x], pixelData[((y - 1) * imageWidth) + (x + 1)]};
            int32_t midRow[3] = {pixelData[(y * imageWidth) + (x - 1)],       pixelData[(y * imageWidth) + x],       pixelData[(y * imageWidth) + (x + 1)]};
            int32_t botRow[3] = {pixelData[((y + 1) * imageWidth) + (x - 1)], pixelData[((y + 1) * imageWidth) + x], pixelData[((y + 1) * imageWidth) + (x + 1)]};

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

}

/**
 * Apply a filter to a set of pixel values using the CPU or GPU
 *
 * @param format      The file format
 * @param imageWidth  The width of the image to write
 * @param imageHeight The height of the image to write
 * @param filter      The type of image filter to use
 * @param pixelData   The single channel pixel data
 * @param useCPU      Flag denoting whether to use the CPU or GPU
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t applyFilter(const FREE_IMAGE_FORMAT & format, 
                    uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    ImageFilter filter, 
                    const BYTE * pixelData, 
                    bool useCPU)
{
    uint32_t imageSize = imageWidth * imageHeight;

    // Host memory
    BYTE * h_outputPixelData;
    int32_t * h_filterX;
    int32_t * h_filterY;

    // Set filter data
    switch (filter)
    {
        case SOBEL:
            h_filterX = const_cast<int32_t *>(SOBEL_X);
            h_filterY = const_cast<int32_t *>(SOBEL_Y);
            break;
        default:
            break;
    }

    // Check to see if using the CPU
    if(useCPU)
    {
        // Allocate memory
        h_outputPixelData = new BYTE[imageSize];

        // Apply filter
        applyFilterCPU(imageWidth, imageHeight, h_filterX, h_filterY, pixelData, h_outputPixelData);
    }
    // Using the GPU
    else
    {
        uint32_t imageSizeBytes = sizeof(BYTE) * imageSize;

        // Allocate host memory
        checkCudaErrors(cudaMallocHost((void **) &h_outputPixelData, imageSizeBytes, cudaHostAllocDefault));

        // Allocate device memory
        BYTE * d_pixelData;
        BYTE * d_outputPixelData;

        checkCudaErrors(cudaMalloc((void **) &d_pixelData, imageSizeBytes));
        checkCudaErrors(cudaMalloc((void **) &d_outputPixelData, imageSizeBytes));

        // Copy pixel data to device
        cudaMemcpy(d_pixelData, pixelData, imageSizeBytes, cudaMemcpyHostToDevice);

        //TODO: Execute kernel

        // Cleanup
        checkCudaErrors(cudaFree(d_pixelData));
        checkCudaErrors(cudaFree(d_outputPixelData));
    }

    // Output results
    std::string outputFileName = "sobel_output.png";

    int32_t status = saveImage(outputFileName, format, imageWidth, imageHeight, 8, h_outputPixelData);

    if(status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    // Cleanup
    if(useCPU)
    {
        delete[] h_outputPixelData;
    }
    else
    {
        cudaFreeHost(h_outputPixelData);
    }

    return EXIT_SUCCESS;
 }