#ifndef SOBEL_H
#define SOBEL_H

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ImageUtils.h"

const int32_t GRAD_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
const int32_t GRAD_Y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__host__
void applySobelFilterCPU(uint32_t imageWidth, uint32_t imageHeight, const BYTE * pixelData, BYTE * outputPixelData);

__global__
void applySobelFilterGPU(uint32_t imageWidth, uint32_t imageHeight, const BYTE * pixelData, BYTE * outputPixelData);

#endif
