#include "Sobel.h"

#include "helper_cuda.h"

//NOTE: The sobel algorithm and kernels were found on
//      https://en.wikipedia.org/wiki/Sobel_operator
//      under the "Pseduocode implementation" section

/**
 * Creates an image that highlights edges by
 * applying a Sobel filter to the specified image
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param pixelData       The input array of pixel data
 * @param outputPixelData The output array of pixel data
 *
 * @return None
 */
 __host__
void applySobelFilterCPU(uint32_t imageWidth, uint32_t imageHeight, const BYTE * pixelData, BYTE * outputPixelData)
{
    for(uint32_t y = 1; y <= (imageHeight - 2); y++)
    {
        for(uint32_t x = 1; x <= (imageWidth - 2); x++)
        {
            uint32_t topRow[3] = {pixelData[((y - 1) * imageWidth) + (x - 1)], pixelData[((y - 1) * imageWidth) + x], pixelData[((y - 1) * imageWidth) + (x + 1)]};
            uint32_t midRow[3] = {pixelData[(y * imageWidth) + (x - 1)],       pixelData[(y * imageWidth) + x],       pixelData[(y * imageWidth) + (x + 1)]};
            uint32_t botRow[3] = {pixelData[((y + 1) * imageWidth) + (x - 1)], pixelData[((y + 1) * imageWidth) + x], pixelData[((y + 1) * imageWidth) + (x + 1)]};

            uint32_t pixelX = (GRAD_X[0][0] * topRow[0]) + (GRAD_X[0][1] * topRow[1]) + (GRAD_X[0][2] * topRow[2]) +
                              (GRAD_X[1][0] * midRow[0]) + (GRAD_X[1][1] * midRow[1]) + (GRAD_X[1][2] * midRow[2]) +
                              (GRAD_X[2][0] * botRow[0]) + (GRAD_X[2][1] * botRow[1]) + (GRAD_X[2][2] * botRow[2]);
            
            uint32_t pixelY = (GRAD_Y[0][0] * topRow[0]) + (GRAD_Y[0][1] * topRow[1]) + (GRAD_Y[0][2] * topRow[2]) +
                              (GRAD_Y[1][0] * midRow[0]) + (GRAD_Y[1][1] * midRow[1]) + (GRAD_Y[1][2] * midRow[2]) +
                              (GRAD_Y[2][0] * botRow[0]) + (GRAD_Y[2][1] * botRow[1]) + (GRAD_Y[2][2] * botRow[2]);
            
            // Calculate magnitude
            uint32_t mag = sqrt((pixelX * pixelX) + (pixelY * pixelY));

            // Set output pixel value
            //TODO: Use some pixel threshold for better results?
            outputPixelData[(y * imageWidth) + x] = mag;
        }
    }
}

/**
 * Creates an image that highlights edges by
 * applying a Sobel filter to the specified image
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param pixelData       The input array of pixel data
 * @param outputPixelData The output array of pixel data
 *
 * @return None
 */
 __global__
void applySobelFilterGPU(uint32_t imageWidth, uint32_t imageHeight, const BYTE * pixelData, BYTE * outputPixelData)
{

}