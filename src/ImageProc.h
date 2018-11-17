#ifndef SOBEL_H
#define SOBEL_H

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ImageUtils.h"

const int32_t SOBEL_X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
const int32_t SOBEL_Y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

enum ImageFilter
{
    SOBEL = 0
};

__host__
void applyFilterCPU(uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    const int32_t * filterX, 
                    const int32_t * filterY, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData);

__global__
void applyFilterGPU(uint32_t imageWidth, 
                    uint32_t imageHeight, 
                    const int32_t * filterX, 
                    const int32_t * filterY, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData);

__host__
int32_t applyFilter(uint32_t imageWidth,
                    uint32_t imageHeight, 
                    ImageFilter filter, 
                    const BYTE * pixelData, 
                    BYTE * outputPixelData, 
                    bool useCPU);

#endif
