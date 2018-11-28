#ifndef SOBEL_H
#define SOBEL_H

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ImageUtils.h"

//NOTE: The following filters were found on page 86 of
//
// Book:      A Simplified Approach to Image Processing: Classical and Modern Techniques in C, 1st Edition
// Authors:   Randy Crane
// ISBN-10:   0132264161
// ISBN-13:   978-0132264167
// Publisher: Prentice Hall PTR
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Smaller effective area makes it more susceptible to noise
const int32_t ROBERTS_X[9] = {0, 0, -1, 
                              0, 1,  0, 
                              0, 0,  0};
const int32_t ROBERTS_Y[9] = {-1, 0, 0, 
                               0, 1, 0, 
                               0, 0, 0};

// More sensitive to diagonal edges
const int32_t SOBEL_X[9] = {1, 0, -1, 
                            2, 0, -2, 
                            1, 0, -1};
const int32_t SOBEL_Y[9] = {-1, -2, -1, 
                             0,  0,  0, 
                             1,  2,  1};

// More sensitive to vertical and horizontal edges
const int32_t PREWITT_X[9] = {1, 0, -1, 
                              1, 0, -1, 
                              1, 0, -1};
const int32_t PREWITT_Y[9] = {-1, -1, -1, 
                               0,  0,  0, 
                               1,  1,  1};

enum ImageFilter
{
    ROBERTS = 0,
    SOBEL = 1,
    PREWITT = 2
};

__host__
__device__
void applyFilter(uint32_t imageWidth, 
                 uint32_t imageHeight, 
                 const int32_t * filterX, 
                 const int32_t * filterY, 
                 int32_t x, 
                 int32_t y, 
                 const BYTE * pixelData, 
                 BYTE * outputPixelData);

__host__
void applyFilterCPU(uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    const int32_t * filterX, 
                    const int32_t * filterY, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData);

__global__
void applyFilterGlobalGPU(uint32_t imageWidth, 
                          uint32_t imageHeight, 
                          const int32_t * filterX, 
                          const int32_t * filterY, 
                          const BYTE * pixelData, 
                          BYTE * outputPixelData);

__global__
void applyFilterSharedGPU(uint32_t imageWidth,
                          uint32_t imageHeight,
                          const int32_t * filterX, 
                          const int32_t * filterY, 
                          const BYTE * pixelData, 
                          BYTE * outputPixelData);

__host__
int32_t filterImage(uint32_t imageWidth,
                    uint32_t imageHeight, 
                    ImageFilter filter, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData, 
                    bool useCPU, 
                    bool useGlobalMem);

#endif
